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
	"math"
	"math/rand"
	"slices"
	"sort"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/util/logging"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/scheduling"
	types "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
)

const (
	WeightedRandomPickerType = "weighted-random-picker"
)

// weightedScoredEndpoint represents a scored endpoint with its A-Res sampling key
type weightedScoredEndpoint struct {
	*types.ScoredEndpoint
	key float64
}

// compile-time type validation
var _ framework.Picker = &WeightedRandomPicker{}

// WeightedRandomPickerFactory defines the factory function for WeightedRandomPicker.
func WeightedRandomPickerFactory(name string, rawParameters json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	parameters := pickerParameters{MaxNumOfEndpoints: DefaultMaxNumOfEndpoints}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' picker - %w", WeightedRandomPickerType, err)
		}
	}

	return NewWeightedRandomPicker(parameters.MaxNumOfEndpoints).WithName(name), nil
}

// NewWeightedRandomPicker initializes a new WeightedRandomPicker and returns its pointer.
func NewWeightedRandomPicker(maxNumOfEndpoints int) *WeightedRandomPicker {
	if maxNumOfEndpoints <= 0 {
		maxNumOfEndpoints = DefaultMaxNumOfEndpoints // on invalid configuration value, fallback to default value
	}

	return &WeightedRandomPicker{
		typedName:         plugins.TypedName{Type: WeightedRandomPickerType, Name: WeightedRandomPickerType},
		maxNumOfEndpoints: maxNumOfEndpoints,
		randomPicker:      NewRandomPicker(maxNumOfEndpoints),
	}
}

// WeightedRandomPicker picks endpoint(s) from the list of candidates based on weighted random sampling using A-Res algorithm.
// Reference: https://utopia.duth.gr/~pefraimi/research/data/2007EncOfAlg.pdf.
//
// The picker at its core is picking endpoints randomly, where the probability of the endpoint to get picked is derived
// from its weighted score.
// Algorithm:
// - Uses A-Res (Algorithm for Reservoir Sampling): keyᵢ = Uᵢ^(1/wᵢ)
// - Selects k items with largest keys for mathematically correct weighted sampling
// - More efficient than traditional cumulative probability approach
//
// Key characteristics:
// - Mathematically correct weighted random sampling
// - Single pass algorithm with O(n + k log k) complexity
type WeightedRandomPicker struct {
	typedName         plugins.TypedName
	maxNumOfEndpoints int
	randomPicker      *RandomPicker // fallback for zero weights
}

// WithName sets the name of the picker.
func (p *WeightedRandomPicker) WithName(name string) *WeightedRandomPicker {
	p.typedName.Name = name
	return p
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *WeightedRandomPicker) TypedName() plugins.TypedName {
	return p.typedName
}

// Pick selects the endpoint(s) randomly from the list of candidates, where the probability of the endpoint to get picked is derived
// from its weighted score.
func (p *WeightedRandomPicker) Pick(ctx context.Context, cycleState *types.CycleState, scoredEndpoints []*types.ScoredEndpoint) *types.ProfileRunResult {
	// Check if there is at least one endpoint with Score > 0, if not let random picker run
	if slices.IndexFunc(scoredEndpoints, func(scoredEndpoint *types.ScoredEndpoint) bool { return scoredEndpoint.Score > 0 }) == -1 {
		log.FromContext(ctx).V(logutil.DEBUG).Info("All scores are zero, delegating to RandomPicker for uniform selection")
		return p.randomPicker.Pick(ctx, cycleState, scoredEndpoints)
	}

	log.FromContext(ctx).V(logutil.DEBUG).Info("Selecting endpoints from candidates by random weighted picker", "max-num-of-endpoints", p.maxNumOfEndpoints,
		"num-of-candidates", len(scoredEndpoints), "scored-endpoints", scoredEndpoints)

	randomGenerator := rand.New(rand.NewSource(time.Now().UnixNano()))

	// A-Res algorithm: keyᵢ = Uᵢ^(1/wᵢ)
	weightedEndpoints := make([]weightedScoredEndpoint, len(scoredEndpoints))

	for i, scoredEndpoint := range scoredEndpoints {
		// Handle zero score
		if scoredEndpoint.Score <= 0 {
			// Assign key=0 for zero-score endpoints (effectively excludes them from selection)
			weightedEndpoints[i] = weightedScoredEndpoint{ScoredEndpoint: scoredEndpoint, key: 0}
			continue
		}

		// If we're here the scoredEndpoint.Score > 0. Generate a random number U in (0,1)
		u := randomGenerator.Float64()
		if u == 0 {
			u = 1e-10 // Avoid log(0)
		}

		weightedEndpoints[i] = weightedScoredEndpoint{ScoredEndpoint: scoredEndpoint, key: math.Pow(u, 1.0/scoredEndpoint.Score)} // key = U^(1/weight)
	}

	// Sort by key in descending order (largest keys first)
	sort.Slice(weightedEndpoints, func(i, j int) bool {
		return weightedEndpoints[i].key > weightedEndpoints[j].key
	})

	// Select top k endpoints
	selectedCount := min(p.maxNumOfEndpoints, len(weightedEndpoints))

	targetEndpoints := make([]types.Endpoint, selectedCount)
	for i := range selectedCount {
		targetEndpoints[i] = weightedEndpoints[i].ScoredEndpoint
	}

	return &types.ProfileRunResult{TargetEndpoints: targetEndpoints}
}
