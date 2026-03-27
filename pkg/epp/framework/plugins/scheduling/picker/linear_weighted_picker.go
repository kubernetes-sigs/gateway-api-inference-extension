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
	"math/rand"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

const (
	LinearWeightedPickerType = "linear-weighted-picker"
)

// compile-time type validation
var _ framework.Picker = &LinearWeightedPicker{}

// LinearWeightedPickerFactory defines the factory function for LinearWeightedPicker.
func LinearWeightedPickerFactory(name string, rawParameters json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	parameters := pickerParameters{MaxNumOfEndpoints: DefaultMaxNumOfEndpoints}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' picker - %w", LinearWeightedPickerType, err)
		}
	}

	return NewLinearWeightedPicker(parameters.MaxNumOfEndpoints).WithName(name), nil
}

// NewLinearWeightedPicker initializes a new LinearWeightedPicker and returns its pointer.
func NewLinearWeightedPicker(maxNumOfEndpoints int) *LinearWeightedPicker {
	if maxNumOfEndpoints <= 0 {
		maxNumOfEndpoints = DefaultMaxNumOfEndpoints
	}

	return &LinearWeightedPicker{
		typedName:         fwkplugin.TypedName{Type: LinearWeightedPickerType, Name: LinearWeightedPickerType},
		maxNumOfEndpoints: maxNumOfEndpoints,
	}
}

// LinearWeightedPicker picks endpoint(s) using linear CDF-based weighted random selection.
// Each endpoint's probability of being selected is directly proportional to its score.
// Endpoints with score <= 0 are excluded from selection.
type LinearWeightedPicker struct {
	typedName         fwkplugin.TypedName
	maxNumOfEndpoints int
}

// WithName sets the picker's name.
func (p *LinearWeightedPicker) WithName(name string) *LinearWeightedPicker {
	p.typedName.Name = name
	return p
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *LinearWeightedPicker) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// Pick selects endpoint(s) using linear weighted random selection.
// The probability of selecting each endpoint is proportional to its score.
func (p *LinearWeightedPicker) Pick(ctx context.Context, _ *framework.CycleState, scoredEndpoints []*framework.ScoredEndpoint) *framework.ProfileRunResult {
	logger := log.FromContext(ctx)

	if len(scoredEndpoints) == 0 {
		return &framework.ProfileRunResult{}
	}

	// Work on a copy to avoid mutating the caller's slice.
	candidates := make([]*framework.ScoredEndpoint, len(scoredEndpoints))
	copy(candidates, scoredEndpoints)

	// Shuffle for tie-breaking when scores are equal.
	shuffleScoredEndpoints(candidates)

	logger.V(logutil.DEBUG).Info("Selecting endpoints by linear weighted random",
		"max-num-of-endpoints", p.maxNumOfEndpoints,
		"num-of-candidates", len(scoredEndpoints))

	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	targetEndpoints := make([]framework.Endpoint, 0, p.maxNumOfEndpoints)

	for range p.maxNumOfEndpoints {
		if len(candidates) == 0 {
			break
		}

		// Compute total weight across positive-score endpoints.
		total := 0.0
		for _, se := range candidates {
			if se.Score > 0 {
				total += se.Score
			}
		}

		// If all scores are zero or negative, pick uniformly at random.
		if total <= 0 {
			idx := r.Intn(len(candidates))
			targetEndpoints = append(targetEndpoints, candidates[idx])
			candidates = removeIndex(candidates, idx)
			continue
		}

		// Linear CDF selection: random value in [0, total), walk the cumulative sum.
		val := r.Float64() * total
		selected := len(candidates) - 1 // fallback to last
		for i, se := range candidates {
			if se.Score <= 0 {
				continue
			}
			val -= se.Score
			if val < 0 {
				selected = i
				break
			}
		}

		targetEndpoints = append(targetEndpoints, candidates[selected])
		candidates = removeIndex(candidates, selected)
	}

	logger.V(logutil.DEBUG).Info("Linear weighted picker selected endpoints",
		"count", len(targetEndpoints))

	return &framework.ProfileRunResult{TargetEndpoints: targetEndpoints}
}

// removeIndex removes element at index i without preserving order.
func removeIndex(s []*framework.ScoredEndpoint, i int) []*framework.ScoredEndpoint {
	s[i] = s[len(s)-1]
	return s[:len(s)-1]
}
