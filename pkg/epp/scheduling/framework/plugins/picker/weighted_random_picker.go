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
	"sort"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	WeightedRandomPickerType = "weighted-random-picker"
)

// weightedScoredPod represents a scored pod with its A-Res sampling key
type weightedScoredPod struct {
	*types.ScoredPod
	key float64
}

var _ framework.Picker = &WeightedRandomPicker{}

func WeightedRandomPickerFactory(name string, rawParameters json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	parameters := pickerParameters{
		MaxNumOfEndpoints: DefaultMaxNumOfEndpoints,
	}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' picker - %w", WeightedRandomPickerType, err)
		}
	}

	return NewWeightedRandomPicker(parameters.MaxNumOfEndpoints).WithName(name), nil
}

func NewWeightedRandomPicker(maxNumOfEndpoints int) *WeightedRandomPicker {
	if maxNumOfEndpoints <= 0 {
		maxNumOfEndpoints = DefaultMaxNumOfEndpoints
	}

	return &WeightedRandomPicker{
		typedName:         plugins.TypedName{Type: WeightedRandomPickerType, Name: WeightedRandomPickerType},
		maxNumOfEndpoints: maxNumOfEndpoints,
		randomPicker:      NewRandomPicker(maxNumOfEndpoints),
	}
}

type WeightedRandomPicker struct {
	typedName         plugins.TypedName
	maxNumOfEndpoints int
	randomPicker      *RandomPicker // fallback for zero weights
}

func (p *WeightedRandomPicker) WithName(name string) *WeightedRandomPicker {
	p.typedName.Name = name
	return p
}

func (p *WeightedRandomPicker) TypedName() plugins.TypedName {
	return p.typedName
}

// WeightedRandomPicker performs weighted random sampling using A-Res algorithm.
// Reference: https://utopia.duth.gr/~pefraimi/research/data/2007EncOfAlg.pdf
// Algorithm:
// - Uses A-Res (Algorithm for Reservoir Sampling): keyᵢ = Uᵢ^(1/wᵢ)
// - Selects k items with largest keys for mathematically correct weighted sampling
// - More efficient than traditional cumulative probability approach
//
// Key characteristics:
// - Mathematically correct weighted random sampling
// - Single pass algorithm with O(n + k log k) complexity
func (p *WeightedRandomPicker) Pick(ctx context.Context, cycleState *types.CycleState, scoredPods []*types.ScoredPod) *types.ProfileRunResult {
	log.FromContext(ctx).V(logutil.DEBUG).Info(fmt.Sprintf("Selecting maximum '%d' pods from %d candidates using weighted random sampling: %+v",
		p.maxNumOfEndpoints, len(scoredPods), scoredPods))

	// Check if all weights are zero or negative
	allZeroWeights := true
	for _, scoredPod := range scoredPods {
		if scoredPod.Score > 0 {
			allZeroWeights = false
			break
		}
	}

	// Delegate to RandomPicker for uniform selection when all weights are zero
	if allZeroWeights {
		log.FromContext(ctx).V(logutil.DEBUG).Info("All weights are zero, delegating to RandomPicker for uniform selection")
		return p.randomPicker.Pick(ctx, cycleState, scoredPods)
	}

	randomGenerator := rand.New(rand.NewSource(time.Now().UnixNano()))

	// A-Res algorithm: keyᵢ = Uᵢ^(1/wᵢ)
	weightedPods := make([]weightedScoredPod, 0, len(scoredPods))

	for _, scoredPod := range scoredPods {
		weight := float64(scoredPod.Score)

		// Handle zero or negative weights
		if weight <= 0 {
			// Assign very small key for zero-weight pods (effectively excludes them)
			weightedPods = append(weightedPods, weightedScoredPod{
				ScoredPod: scoredPod,
				key:       0,
			})
			continue
		}

		// Generate random number U in (0,1)
		u := randomGenerator.Float64()
		if u == 0 {
			u = 1e-10 // Avoid log(0)
		}

		// Calculate key = U^(1/weight)
		key := math.Pow(u, 1.0/weight)

		weightedPods = append(weightedPods, weightedScoredPod{
			ScoredPod: scoredPod,
			key:       key,
		})
	}

	// Sort by key in descending order (largest keys first)
	sort.Slice(weightedPods, func(i, j int) bool {
		return weightedPods[i].key > weightedPods[j].key
	})

	// Select top k pods
	selectedCount := min(p.maxNumOfEndpoints, len(weightedPods))

	scoredPods = make([]*types.ScoredPod, selectedCount)
	for i := range selectedCount {
		scoredPods[i] = weightedPods[i].ScoredPod
	}

	targetPods := make([]types.Pod, len(scoredPods))
	for i, scoredPod := range scoredPods {
		targetPods[i] = scoredPod
	}

	return &types.ProfileRunResult{TargetPods: targetPods}
}
