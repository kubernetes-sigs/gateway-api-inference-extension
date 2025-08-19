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

type NormalizationType string

const (
	NormalizationNone    NormalizationType = "none"
	NormalizationCapping NormalizationType = "capping"
	NormalizationLog     NormalizationType = "logarithmic"
	NormalizationSqrt    NormalizationType = "sqrt"
)

type weightedRandomPickerParameters struct {
	MaxNumOfEndpoints int               `json:"maxNumOfEndpoints"`
	NormalizationType NormalizationType `json:"normalizationType,omitempty"`
	MaxRatio          float64           `json:"maxRatio,omitempty"`
}

var _ framework.Picker = &WeightedRandomPicker{}

func WeightedRandomPickerFactory(name string, rawParameters json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	parameters := weightedRandomPickerParameters{
		MaxNumOfEndpoints: DefaultMaxNumOfEndpoints,
		NormalizationType: NormalizationNone,
		MaxRatio:          3.0,
	}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' picker - %w", WeightedRandomPickerType, err)
		}
	}

	return NewWeightedRandomPicker(parameters.MaxNumOfEndpoints, parameters.NormalizationType, parameters.MaxRatio).WithName(name), nil
}

func NewWeightedRandomPicker(maxNumOfEndpoints int, options ...interface{}) *WeightedRandomPicker {
	if maxNumOfEndpoints <= 0 {
		maxNumOfEndpoints = DefaultMaxNumOfEndpoints
	}

	// Set defaults
	normalizationType := NormalizationNone
	maxRatio := 3.0

	// Parse options in order: normalizationType, maxRatio
	for _, option := range options {
		switch v := option.(type) {
		case NormalizationType:
			normalizationType = v
		case float64:
			maxRatio = v
		default:
			// Ignore unknown types for forward compatibility
		}
	}

	// Validate maxRatio
	if maxRatio <= 1.0 {
		maxRatio = 3.0
	}

	return &WeightedRandomPicker{
		typedName:         plugins.TypedName{Type: WeightedRandomPickerType, Name: WeightedRandomPickerType},
		maxNumOfEndpoints: maxNumOfEndpoints,
		normalizationType: normalizationType,
		maxRatio:          maxRatio,
	}
}

type WeightedRandomPicker struct {
	typedName         plugins.TypedName
	maxNumOfEndpoints int
	normalizationType NormalizationType
	maxRatio          float64
}

func (p *WeightedRandomPicker) WithName(name string) *WeightedRandomPicker {
	p.typedName.Name = name
	return p
}

func (p *WeightedRandomPicker) TypedName() plugins.TypedName {
	return p.typedName
}

func (p *WeightedRandomPicker) Pick(ctx context.Context, _ *types.CycleState, scoredPods []*types.ScoredPod) *types.ProfileRunResult {
	log.FromContext(ctx).V(logutil.DEBUG).Info(fmt.Sprintf("Selecting maximum '%d' pods from %d candidates using weighted random sampling with normalization '%s': %+v", p.maxNumOfEndpoints,
		len(scoredPods), p.normalizationType, scoredPods))

	if len(scoredPods) == 0 {
		return &types.ProfileRunResult{TargetPods: []types.Pod{}}
	}

	// Apply normalization only when needed for performance
	if p.normalizationType != NormalizationNone {
		normalizeScores(scoredPods, p.normalizationType, p.maxRatio)
	}

	randomGenerator := rand.New(rand.NewSource(time.Now().UnixNano()))

	var totalWeight float64
	for _, scoredPod := range scoredPods {
		if scoredPod.Score >= 0 {
			totalWeight += float64(scoredPod.Score)
		}
	}

	if totalWeight == 0 {
		randomGenerator.Shuffle(len(scoredPods), func(i, j int) {
			scoredPods[i], scoredPods[j] = scoredPods[j], scoredPods[i]
		})
		if p.maxNumOfEndpoints < len(scoredPods) {
			scoredPods = scoredPods[:p.maxNumOfEndpoints]
		}
	} else {
		selectedPods := make([]*types.ScoredPod, 0, p.maxNumOfEndpoints)
		remainingPods := make([]*types.ScoredPod, len(scoredPods))
		copy(remainingPods, scoredPods)

		for len(selectedPods) < p.maxNumOfEndpoints && len(remainingPods) > 0 {
			currentTotalWeight := float64(0)
			for _, pod := range remainingPods {
				if pod.Score >= 0 {
					currentTotalWeight += float64(pod.Score)
				}
			}

			if currentTotalWeight == 0 {
				selectedIndex := randomGenerator.Intn(len(remainingPods))
				selectedPods = append(selectedPods, remainingPods[selectedIndex])
				remainingPods = append(remainingPods[:selectedIndex], remainingPods[selectedIndex+1:]...)
				continue
			}

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

func normalizeScores(scoredPods []*types.ScoredPod, normType NormalizationType, maxRatio float64) {
	if normType == NormalizationNone || len(scoredPods) == 0 {
		return
	}

	switch normType {
	case NormalizationCapping:
		applyCappingNormalizationOptimized(scoredPods, maxRatio)
	case NormalizationLog:
		applyLogNormalization(scoredPods)
	case NormalizationSqrt:
		applySqrtNormalization(scoredPods)
	}
}

func applyLogNormalization(scoredPods []*types.ScoredPod) {
	for _, pod := range scoredPods {
		if pod.Score > 0 {
			pod.Score = math.Log(1 + pod.Score)
		}
	}
}

func applySqrtNormalization(scoredPods []*types.ScoredPod) {
	for _, pod := range scoredPods {
		if pod.Score > 0 {
			pod.Score = math.Sqrt(pod.Score)
		}
	}
}

func applyCappingNormalizationOptimized(scoredPods []*types.ScoredPod, maxRatio float64) {
	if maxRatio <= 1.0 || len(scoredPods) == 0 {
		return
	}

	// Single pass: find min and apply capping simultaneously
	minScore := math.Inf(1)

	// First pass: find minimum positive score
	for _, pod := range scoredPods {
		if pod.Score > 0 && pod.Score < minScore {
			minScore = pod.Score
		}
	}

	if math.IsInf(minScore, 1) || minScore <= 0 {
		return
	}

	maxAllowed := minScore * maxRatio

	// Second pass: apply capping
	for _, pod := range scoredPods {
		if pod.Score > maxAllowed {
			pod.Score = maxAllowed
		}
	}
}
