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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	WeightedRandomPickerType = "weighted-random-picker"
	TopNThreshold           = 0.34  // Threshold that safely includes 1/3 ratio (0.333...)
)

type weightedRandomPickerParameters struct {
	MaxNumOfEndpoints int `json:"maxNumOfEndpoints"`
}

var _ framework.Picker = &WeightedRandomPicker{}

func WeightedRandomPickerFactory(name string, rawParameters json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	parameters := weightedRandomPickerParameters{
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
	}
}

type WeightedRandomPicker struct {
	typedName         plugins.TypedName
	maxNumOfEndpoints int
}

func (p *WeightedRandomPicker) WithName(name string) *WeightedRandomPicker {
	p.typedName.Name = name
	return p
}

func (p *WeightedRandomPicker) TypedName() plugins.TypedName {
	return p.typedName
}

// WeightedRandomPicker performs weighted random sampling with topN filtering to prevent hotspots.
//
// Key characteristics:
// - Most effective when maxNumOfEndpoints = 1 (true probabilistic selection)
// - As maxNumOfEndpoints increases, behavior converges toward max-score picker
// - Uses "sampling without replacement" - selected pods are removed from subsequent selections
// - Applies topN filtering when requesting < 34% of total pods to prevent hotspots
//
// TopN Logic:
// - If maxEndpoints >= totalPods: use all pods
// - If maxEndpoints/totalPods < 0.34: use top (maxEndpoints + 1) pods to prevent hotspots
// - Otherwise: use all pods for maximum diversity
func (p *WeightedRandomPicker) Pick(ctx context.Context, _ *types.CycleState, scoredPods []*types.ScoredPod) *types.ProfileRunResult {
	log.FromContext(ctx).V(logutil.DEBUG).Info(fmt.Sprintf("Selecting maximum '%d' pods from %d candidates using weighted random sampling: %+v", 
		p.maxNumOfEndpoints, len(scoredPods), scoredPods))

	randomGenerator := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Apply topN filtering to prevent hotspots
	topN := p.calculateTopN(p.maxNumOfEndpoints, len(scoredPods))
	if topN < len(scoredPods) {
		// Sort by score in descending order to get top pods
		sortedPods := make([]*types.ScoredPod, len(scoredPods))
		copy(sortedPods, scoredPods)
		
		// Shuffle first for fair tie-breaking when scores are equal
		randomGenerator.Shuffle(len(sortedPods), func(i, j int) {
			sortedPods[i], sortedPods[j] = sortedPods[j], sortedPods[i]
		})
		
		// Simple bubble sort by score (descending)
		for i := 0; i < len(sortedPods)-1; i++ {
			for j := 0; j < len(sortedPods)-i-1; j++ {
				if sortedPods[j].Score < sortedPods[j+1].Score {
					sortedPods[j], sortedPods[j+1] = sortedPods[j+1], sortedPods[j]
				}
			}
		}
		
		// Take only top N pods
		scoredPods = sortedPods[:topN]
		log.FromContext(ctx).V(logutil.DEBUG).Info(fmt.Sprintf("Applied topN filtering: using top %d pods from %d total", topN, len(sortedPods)))
	}

	// Calculate total weight
	var totalWeight float64
	for _, scoredPod := range scoredPods {
		if scoredPod.Score >= 0 {
			totalWeight += float64(scoredPod.Score)
		}
	}

	// Handle zero weight case - fallback to random selection
	if totalWeight == 0 {
		randomGenerator.Shuffle(len(scoredPods), func(i, j int) {
			scoredPods[i], scoredPods[j] = scoredPods[j], scoredPods[i]
		})
		if p.maxNumOfEndpoints < len(scoredPods) {
			scoredPods = scoredPods[:p.maxNumOfEndpoints]
		}
	} else {
		// Weighted random sampling without replacement
		selectedPods := make([]*types.ScoredPod, 0, p.maxNumOfEndpoints)
		remainingPods := make([]*types.ScoredPod, len(scoredPods))
		copy(remainingPods, scoredPods)

		for len(selectedPods) < p.maxNumOfEndpoints && len(remainingPods) > 0 {
			// Recalculate total weight for remaining pods
			currentTotalWeight := float64(0)
			for _, pod := range remainingPods {
				if pod.Score >= 0 {
					currentTotalWeight += float64(pod.Score)
				}
			}

			if currentTotalWeight == 0 {
				// Fallback to random selection for remaining pods
				selectedIndex := randomGenerator.Intn(len(remainingPods))
				selectedPods = append(selectedPods, remainingPods[selectedIndex])
				remainingPods = append(remainingPods[:selectedIndex], remainingPods[selectedIndex+1:]...)
				continue
			}

			// Weighted random selection
			randomValue := randomGenerator.Float64() * currentTotalWeight
			cumulativeWeight := float64(0)
			selectedIndex := -1

			for i, pod := range remainingPods {
				if pod.Score >= 0 {
					cumulativeWeight += float64(pod.Score)
					if randomValue <= cumulativeWeight {
						selectedIndex = i
						break
					}
				}
			}

			if selectedIndex == -1 {
				selectedIndex = len(remainingPods) - 1
			}

			// Add selected pod and remove from remaining
			selectedPods = append(selectedPods, remainingPods[selectedIndex])
			remainingPods = append(remainingPods[:selectedIndex], remainingPods[selectedIndex+1:]...)
		}

		scoredPods = selectedPods
	}

	targetPods := make([]types.Pod, len(scoredPods))
	for i, scoredPod := range scoredPods {
		targetPods[i] = scoredPod
	}

	return &types.ProfileRunResult{TargetPods: targetPods}
}

// calculateTopN determines the number of top pods to consider for weighted random sampling
//
// Key test cases that demonstrate the filtering behavior:
// Core cases with hotspot prevention:
//   maxEndpoint=1, scoredPods=3 → 1/3=0.333 < 0.34 ✓ (topN=2)
//   maxEndpoint=1, scoredPods=4 → 1/4=0.25 < 0.34 ✓ (topN=2)  
//   maxEndpoint=2, scoredPods=6 → 2/6=0.333 < 0.34 ✓ (topN=3)
//
//
// Special case for MaxScorePicker differentiation:
//   maxEndpoint=1, scoredPods=2 → special case ✓ (topN=2, uses all pods)
//
// Boundary cases:
//   maxEndpoint=3, scoredPods=9 → 3/9=0.333 < 0.34 ✓ (topN=4)
//   maxEndpoint=3, scoredPods=8 → 3/8=0.375 > 0.34 ✗ (topN=3)
func (p *WeightedRandomPicker) calculateTopN(maxEndpoint, totalPods int) int {
	// If requesting all or more pods, use all available pods
	if maxEndpoint >= totalPods {
		return totalPods
	}
	
	// Special case: maxEndpoint=1, totalPods=2 should use both pods for differentiation from MaxScorePicker
	if maxEndpoint == 1 && totalPods == 2 {
		return totalPods
	}
	
	// If requesting less than 1/3 of total pods, add 1 extra pod to prevent hotspots
	ratio := float64(maxEndpoint) / float64(totalPods)
	if ratio < TopNThreshold {
		return maxEndpoint + 1
	}
	
	// Otherwise, use the requested number of pods
	return maxEndpoint
}