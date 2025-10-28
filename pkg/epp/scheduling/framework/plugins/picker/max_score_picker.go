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
	"slices"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	MaxScorePickerType = "max-score-picker"
)

// compile-time type validation
var _ framework.Picker = &MaxScorePicker{}

// MaxScorePickerFactory defines the factory function for MaxScorePicker.
func MaxScorePickerFactory(name string, rawParameters json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	parameters := pickerParameters{MaxNumOfEndpoints: DefaultMaxNumOfEndpoints}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' picker - %w", MaxScorePickerType, err)
		}
	}

	return NewMaxScorePicker(parameters.MaxNumOfEndpoints).WithName(name), nil
}

// NewMaxScorePicker initializes a new MaxScorePicker and returns its pointer.
func NewMaxScorePicker(maxNumOfEndpoints int) *MaxScorePicker {
	if maxNumOfEndpoints <= 0 {
		maxNumOfEndpoints = DefaultMaxNumOfEndpoints // on invalid configuration value, fallback to default value
	}

	return &MaxScorePicker{
		typedName:         plugins.TypedName{Type: MaxScorePickerType, Name: MaxScorePickerType},
		maxNumOfEndpoints: maxNumOfEndpoints,
	}
}

// MaxScorePicker picks pod(s) with the maximum score from the list of candidates.
type MaxScorePicker struct {
	typedName         plugins.TypedName
	maxNumOfEndpoints int // maximum number of endpoints to pick
}

// WithName sets the picker's name
func (p *MaxScorePicker) WithName(name string) *MaxScorePicker {
	p.typedName.Name = name
	return p
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *MaxScorePicker) TypedName() plugins.TypedName {
	return p.typedName
}

// Pick selects the pod with the maximum score from the list of candidates.
func (p *MaxScorePicker) Pick(ctx context.Context, cycleState *types.CycleState, scoredPods []*types.ScoredPod) *types.ProfileRunResult {
	log.FromContext(ctx).V(logutil.DEBUG).Info("Selecting pods from candidates sorted by max score", "max-num-of-endpoints", p.maxNumOfEndpoints,
		"num-of-candidates", len(scoredPods), "scored-pods", scoredPods)

	// Shuffle in-place - needed for random tie break when scores are equal
	shuffleScoredPods(scoredPods)

	slices.SortStableFunc(scoredPods, func(i, j *types.ScoredPod) int { // highest score first
		if i.Score > j.Score {
			return -1
		}
		if i.Score < j.Score {
			return 1
		}
		return 0
	})

	// if we have enough pods to return keep only the "maxNumOfEndpoints" highest scored pods
	if p.maxNumOfEndpoints < len(scoredPods) {
		scoredPods = scoredPods[:p.maxNumOfEndpoints]
	}

	targetPods := make([]types.Pod, len(scoredPods))
	for i, scoredPod := range scoredPods {
		targetPods[i] = scoredPod
	}

	return &types.ProfileRunResult{TargetPods: targetPods}
}
