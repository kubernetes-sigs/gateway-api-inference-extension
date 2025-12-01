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

package weightedrandompicker

import (
	"context"
	"math"
	"testing"

	k8stypes "k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

func TestPickWeightedRandomPicker(t *testing.T) {
	const (
		testIterations = 10000
		tolerance      = 0.05 // Verify within tolerance ±5%
	)

	pod1 := &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}}
	pod2 := &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}}
	pod3 := &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}}
	pod4 := &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod4"}}}
	pod5 := &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod5"}}}

	// A-Res algorithm uses U^(1/w) transformation which introduces statistical variance
	// beyond simple proportional sampling. Generous tolerance is required to prevent
	// flaky tests in CI environments, especially for multi-tier weights.
	tests := []struct {
		name    string
		input   []*types.ScoredPod
		maxPods int // maxNumOfEndpoints for this test
	}{
		{
			name: "High weight dominance test",
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 10}, // Lower weight
				{Pod: pod2, Score: 90}, // Higher weight (should dominate)
			},
			maxPods: 1,
		},
		{
			name: "Equal weights test - A-Res uniform distribution",
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 100}, // Equal weights (higher values for better numerical precision)
				{Pod: pod2, Score: 100}, // Equal weights should yield uniform distribution
				{Pod: pod3, Score: 100}, // Equal weights in A-Res
			},
			maxPods: 1,
		},
		{
			name: "Zero weight exclusion test - A-Res edge case",
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 30}, // Normal weight, should be selected
				{Pod: pod2, Score: 0},  // Zero weight, never selected in A-Res
			},
			maxPods: 1,
		},
		{
			name: "Multi-tier weighted test - A-Res complex distribution",
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 100}, // Highest weight
				{Pod: pod2, Score: 90},  // High weight
				{Pod: pod3, Score: 50},  // Medium weight
				{Pod: pod4, Score: 30},  // Low weight
				{Pod: pod5, Score: 20},  // Lowest weight
			},
			maxPods: 1,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			picker := NewWeightedRandomPicker(test.maxPods)

			// Summarize the total score of all pods
			totalScore := 0.0
			for _, pod := range test.input {
				totalScore += pod.Score
			}

			// Calculate expected probabilities based on scores
			expectedProbabilities := make(map[string]float64)
			for _, pod := range test.input {
				podName := pod.GetPod().NamespacedName.Name
				if totalScore > 0 {
					expectedProbabilities[podName] = pod.Score / totalScore
				} else {
					expectedProbabilities[podName] = 0.0
				}
			}

			// Initialize selection counters for each pod
			selectionCounts := make(map[string]int)
			for _, pod := range test.input {
				podName := pod.GetPod().NamespacedName.Name
				selectionCounts[podName] = 0
			}

			// Run multiple iterations to gather statistical data
			for range testIterations {
				result := picker.Pick(context.Background(), types.NewCycleState(), test.input)

				// Count selections for probability analysis
				selectedPodName := result.TargetPods[0].GetPod().NamespacedName.Name
				selectionCounts[selectedPodName]++
			}

			// Verify probability distribution
			for podName, expectedProb := range expectedProbabilities {
				actualCount := selectionCounts[podName]
				actualProb := float64(actualCount) / float64(testIterations)

				if math.Abs(actualProb-expectedProb) > tolerance {
					t.Errorf("Pod %s: expected probability %.3f ±%.1f%%, got %.3f (count: %d/%d)",
						podName, expectedProb, tolerance*100, actualProb, actualCount, testIterations)
				} else {
					t.Logf("Pod %s: expected %.3f, got %.3f (count: %d/%d) ✓",
						podName, expectedProb, actualProb, actualCount, testIterations)
				}
			}
		})
	}
}
