/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package picker

import (
	"context"
	"encoding/json"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	RandomPickerType = "random-picker"
)

// compile-time type validation
var _ framework.Picker = &RandomPicker{}

// RandomPickerFactory defines the factory function for RandomPicker.
func RandomPickerFactory(name string, rawParameters json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	parameters := pickerParameters{MaxNumOfEndpoints: DefaultMaxNumOfEndpoints}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' picker - %w", RandomPickerType, err)
		}
	}

	return NewRandomPicker(parameters.MaxNumOfEndpoints).WithName(name), nil
}

// NewRandomPicker initializes a new RandomPicker and returns its pointer.
func NewRandomPicker(maxNumOfEndpoints int) *RandomPicker {
	if maxNumOfEndpoints <= 0 {
		maxNumOfEndpoints = DefaultMaxNumOfEndpoints // on invalid configuration value, fallback to default value
	}

	return &RandomPicker{
		typedName:         plugins.TypedName{Type: RandomPickerType, Name: RandomPickerType},
		maxNumOfEndpoints: maxNumOfEndpoints,
	}
}

// RandomPicker picks random pod(s) from the list of candidates.
type RandomPicker struct {
	typedName         plugins.TypedName
	maxNumOfEndpoints int
}

// WithName sets the name of the picker.
func (p *RandomPicker) WithName(name string) *RandomPicker {
	p.typedName.Name = name
	return p
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *RandomPicker) TypedName() plugins.TypedName {
	return p.typedName
}

// Pick selects random pod(s) from the list of candidates.
func (p *RandomPicker) Pick(ctx context.Context, _ *types.CycleState, scoredPods []*types.ScoredPod) *types.ProfileRunResult {
	log.FromContext(ctx).V(logutil.DEBUG).Info("Selecting pods from candidates randomly", "max-num-of-endpoints", p.maxNumOfEndpoints,
		"num-of-candidates", len(scoredPods), "scored-pods", scoredPods)

	// Shuffle in-place
	shuffleScoredPods(scoredPods)

	// if we have enough pods to return keep only the relevant subset
	if p.maxNumOfEndpoints < len(scoredPods) {
		scoredPods = scoredPods[:p.maxNumOfEndpoints]
	}

	targetPods := make([]types.Pod, len(scoredPods))
	for i, scoredPod := range scoredPods {
		targetPods[i] = scoredPod
	}

	return &types.ProfileRunResult{TargetPods: targetPods}
}
