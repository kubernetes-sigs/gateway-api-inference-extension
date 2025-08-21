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
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

func TestPickMaxScorePicker(t *testing.T) {
	pod1 := &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}}
	pod2 := &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}}
	pod3 := &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}}

	tests := []struct {
		name               string
		picker             framework.Picker
		input              []*types.ScoredPod
		output             []types.Pod
		tieBreakCandidates int // tie break is random, specify how many candidate with max score
	}{
		{
			name:   "Single max score",
			picker: NewMaxScorePicker(1),
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 10},
				{Pod: pod2, Score: 25},
				{Pod: pod3, Score: 15},
			},
			output: []types.Pod{
				&types.ScoredPod{Pod: pod2, Score: 25},
			},
		},
		{
			name:   "Multiple max scores, all are equally scored",
			picker: NewMaxScorePicker(2),
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 50},
				{Pod: pod2, Score: 50},
				{Pod: pod3, Score: 30},
			},
			output: []types.Pod{
				&types.ScoredPod{Pod: pod1, Score: 50},
				&types.ScoredPod{Pod: pod2, Score: 50},
			},
			tieBreakCandidates: 2,
		},
		{
			name:   "Multiple results sorted by highest score, more pods than needed",
			picker: NewMaxScorePicker(2),
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 20},
				{Pod: pod2, Score: 25},
				{Pod: pod3, Score: 30},
			},
			output: []types.Pod{
				&types.ScoredPod{Pod: pod3, Score: 30},
				&types.ScoredPod{Pod: pod2, Score: 25},
			},
		},
		{
			name:   "Multiple results sorted by highest score, less pods than needed",
			picker: NewMaxScorePicker(4), // picker is required to return 4 pods at most, but we have only 3.
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 20},
				{Pod: pod2, Score: 25},
				{Pod: pod3, Score: 30},
			},
			output: []types.Pod{
				&types.ScoredPod{Pod: pod3, Score: 30},
				&types.ScoredPod{Pod: pod2, Score: 25},
				&types.ScoredPod{Pod: pod1, Score: 20},
			},
		},
		{
			name:   "Multiple results sorted by highest score, num of pods exactly needed",
			picker: NewMaxScorePicker(3), // picker is required to return 3 pods at most, we have only 3.
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 30},
				{Pod: pod2, Score: 25},
				{Pod: pod3, Score: 30},
			},
			output: []types.Pod{
				&types.ScoredPod{Pod: pod1, Score: 30},
				&types.ScoredPod{Pod: pod3, Score: 30},
				&types.ScoredPod{Pod: pod2, Score: 25},
			},
			tieBreakCandidates: 2,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := test.picker.Pick(context.Background(), types.NewCycleState(), test.input)
			got := result.TargetPods

			if test.tieBreakCandidates > 0 {
				testMaxScoredPods := test.output[:test.tieBreakCandidates]
				gotMaxScoredPods := got[:test.tieBreakCandidates]
				diff := cmp.Diff(testMaxScoredPods, gotMaxScoredPods, cmpopts.SortSlices(func(a, b types.Pod) bool {
					return a.String() < b.String() // predictable order within the pods with equal scores
				}))
				if diff != "" {
					t.Errorf("Unexpected output (-want +got): %v", diff)
				}
				test.output = test.output[test.tieBreakCandidates:]
				got = got[test.tieBreakCandidates:]
			}

			if diff := cmp.Diff(test.output, got); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}

func TestPickWeightedRandomPicker(t *testing.T) {
	pod1 := &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}}
	pod2 := &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}}
	pod3 := &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}}
	pod4 := &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod4"}}}
	pod5 := &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod5"}}}

	tests := []struct {
		name                 string
		input                []*types.ScoredPod
		maxPods              int                 // maxNumOfEndpoints for this test
		iterations           int
		expectedProbabilities map[string]float64 // pod name -> expected probability
		tolerancePercent     float64             // acceptable deviation percentage
		expectExactLength    int                 // expected exact length per iteration (0 means skip this check)
	}{
		{
			name: "All pods requested - basic functionality",
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 10},
				{Pod: pod2, Score: 20},
				{Pod: pod3, Score: 30},
			},
			maxPods:           5, // Request more than available
			iterations:        10,
			expectExactLength: 3, // All 3 pods returned (maxPods >= totalPods, so use all)
		},
		{
			name: "High weight dominance test",
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 10}, // 10/100 = 10%
				{Pod: pod2, Score: 90}, // 90/100 = 90%
			},
			maxPods:    1, // special case: maxEndpoint=1, totalPods=2 → topN=2 (uses all)
			iterations: 2000,
			expectedProbabilities: map[string]float64{
				"pod1": 0.10,
				"pod2": 0.90,
			},
			tolerancePercent: 20.0, // ±20% (more tolerant for statistical variance)
		},
		{
			name: "Equal weights test with topN filtering",
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 50}, // Equal scores (will be in top 2 after sorting)
				{Pod: pod2, Score: 50}, // Equal scores (will be in top 2 after sorting)
				{Pod: pod3, Score: 50}, // Equal scores (will be filtered out)
			},
			maxPods:    1, // ratio = 1/3 = 0.333 < 0.34, so topN = 1+1 = 2
			iterations: 1500,
			expectedProbabilities: map[string]float64{
				"pod1": 0.333, // Each pod has equal chance: 2/3 * 1/2 ≈ 0.333
				"pod2": 0.333, // Each pod has equal chance: 2/3 * 1/2 ≈ 0.333
				"pod3": 0.333, // Each pod has equal chance: 2/3 * 1/2 ≈ 0.333
			},
			tolerancePercent: 15.0, // ±15%
		},
		{
			name: "Progressive weight test",
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 20}, // 20/60 = 33.3%
				{Pod: pod2, Score: 40}, // 40/60 = 66.7%
			},
			maxPods:    1, // special case: maxEndpoint=1, totalPods=2 → topN=2 (uses all)
			iterations: 1200,
			expectedProbabilities: map[string]float64{
				"pod1": 0.333,
				"pod2": 0.667,
			},
			tolerancePercent: 20.0, // ±20% (more tolerant for statistical variance)
		},
		{
			name: "Zero weight exclusion test",
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 30}, // 30/30 = 100%
				{Pod: pod2, Score: 0},  // 0/30 = 0%
			},
			maxPods:    1,
			iterations: 500,
			expectedProbabilities: map[string]float64{
				"pod1": 1.0,
				"pod2": 0.0,
			},
			tolerancePercent: 5.0, // ±5%
		},
		{
			name: "Top N filtering - only top 2 pods with threshold",
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 100}, // Highest probability
				{Pod: pod2, Score: 90},  // Second highest probability
				{Pod: pod3, Score: 50},  // Should be filtered out
				{Pod: pod4, Score: 30},  // Should be filtered out
				{Pod: pod5, Score: 10},  // Should be filtered out
			},
			maxPods:    1, // ratio = 1/5 = 0.2 < 0.34, so topN = 1+1 = 2
			iterations: 1000,
			expectedProbabilities: map[string]float64{
				"pod1": 0.526, // 100/(100+90) ≈ 52.6%
				"pod2": 0.474, // 90/(100+90) ≈ 47.4%
				"pod3": 0.0,   // Should never be selected
				"pod4": 0.0,   // Should never be selected
				"pod5": 0.0,   // Should never be selected
			},
			tolerancePercent: 15.0,
		},
		{
			name: "Boundary test: ratio exactly at threshold (1/3 ≈ 0.33)",
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 100}, // Highest probability
				{Pod: pod2, Score: 90},  // Second highest probability  
				{Pod: pod3, Score: 50},  // Should be included (ratio < 0.34)
			},
			maxPods:    1, // ratio = 1/3 = 0.333 < 0.34, so topN = 1+1 = 2
			iterations: 1000,
			expectedProbabilities: map[string]float64{
				"pod1": 0.526, // 100/(100+90) ≈ 52.6%
				"pod2": 0.474, // 90/(100+90) ≈ 47.4%
				"pod3": 0.0,   // Should never be selected due to filtering
			},
			tolerancePercent: 15.0,
		},
		{
			name: "No filtering: ratio above threshold (2/3 ≈ 0.67)",
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 100}, // Highest probability
				{Pod: pod2, Score: 90},  // Second highest probability  
				{Pod: pod3, Score: 50},  // Should be included (no filtering)
			},
			maxPods:           2, // ratio = 2/3 = 0.667 ≥ 0.34, so no topN filtering
			iterations:        10,
			expectExactLength: 2, // Should return exactly 2 pods per iteration
		},
		{
			name: "Edge case: maxPods > filtered count, should use all pods",
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 100},
				{Pod: pod2, Score: 90},
				{Pod: pod3, Score: 50},
				{Pod: pod4, Score: 30},
				{Pod: pod5, Score: 10},
			},
			maxPods:           5, // maxPods >= totalPods, so use all 5 pods
			iterations:        10,
			expectExactLength: 5, // All 5 pods should be returned
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			picker := NewWeightedRandomPicker(test.maxPods)
			selectionCounts := make(map[string]int)

			// Initialize counters for simple pod names
			for _, pod := range test.input {
				podName := pod.GetPod().NamespacedName.Name
				selectionCounts[podName] = 0
			}

			// Run multiple iterations to gather statistics
			var totalLength int
			for i := 0; i < test.iterations; i++ {
				// Create fresh copy of input for each iteration
				inputCopy := make([]*types.ScoredPod, len(test.input))
				for j, pod := range test.input {
					inputCopy[j] = &types.ScoredPod{Pod: pod.Pod, Score: pod.Score}
				}

				result := picker.Pick(context.Background(), types.NewCycleState(), inputCopy)
				totalLength += len(result.TargetPods)

				// Count selections for probability distribution (when selecting 1 pod)
				if test.maxPods == 1 && len(result.TargetPods) > 0 {
					selectedPodName := result.TargetPods[0].GetPod().NamespacedName.Name
					selectionCounts[selectedPodName]++
				}
			}

			// Check exact length if specified
			if test.expectExactLength > 0 {
				expectedTotalLength := test.expectExactLength * test.iterations
				if totalLength != expectedTotalLength {
					t.Errorf("Expected total length %d (avg %.1f), got %d (avg %.1f)", 
						expectedTotalLength, float64(test.expectExactLength), 
						totalLength, float64(totalLength)/float64(test.iterations))
				} else {
					t.Logf("Exact length test passed: %d pods selected per iteration ✓", test.expectExactLength)
				}
			}

			// Verify probability distribution (only for single pod selection tests)
			if test.expectedProbabilities != nil && test.maxPods == 1 {
				for podName, expectedProb := range test.expectedProbabilities {
					actualCount := selectionCounts[podName]
					actualProb := float64(actualCount) / float64(test.iterations)
					
					tolerance := expectedProb * test.tolerancePercent / 100.0
					lowerBound := expectedProb - tolerance
					upperBound := expectedProb + tolerance

					if actualProb < lowerBound || actualProb > upperBound {
						t.Errorf("Pod %s: expected probability %.3f ±%.1f%%, got %.3f (count: %d/%d)",
							podName, expectedProb, test.tolerancePercent, actualProb, actualCount, test.iterations)
					} else {
						t.Logf("Pod %s: expected %.3f, got %.3f (count: %d/%d) ✓",
							podName, expectedProb, actualProb, actualCount, test.iterations)
					}
				}
			}
		})
	}
}

