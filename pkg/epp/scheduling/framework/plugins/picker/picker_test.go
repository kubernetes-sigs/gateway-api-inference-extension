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

	// A-Res algorithm uses U^(1/w) transformation which introduces statistical variance
	// beyond simple proportional sampling. Generous tolerance is required to prevent
	// flaky tests in CI environments, especially for multi-tier weights.
	tests := []struct {
		name                 string
		input                []*types.ScoredPod
		maxPods              int                 // maxNumOfEndpoints for this test
		iterations           int
		expectedProbabilities map[string]float64 // pod name -> expected probability
		tolerancePercent     float64             // acceptable deviation percentage
	}{
		{
			name: "High weight dominance test",
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 10}, // Lower weight
				{Pod: pod2, Score: 90}, // Higher weight (should dominate)
			},
			maxPods:    1,
			iterations: 2000,
			expectedProbabilities: map[string]float64{
				"pod1": 0.10,
				"pod2": 0.90,
			},
			tolerancePercent: 20.0,
		},
		{
			name: "Equal weights test - A-Res uniform distribution",
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 100}, // Equal weights (higher values for better numerical precision)
				{Pod: pod2, Score: 100}, // Equal weights should yield uniform distribution
				{Pod: pod3, Score: 100}, // Equal weights in A-Res
			},
			maxPods:    1,
			iterations: 1500,
			expectedProbabilities: map[string]float64{
				"pod1": 0.333, // Equal weights should yield uniform distribution
				"pod2": 0.333, // A-Res maintains equal probability for equal weights
				"pod3": 0.333, // Each pod has theoretically equal chance
			},
			tolerancePercent: 20.0,
		},
		{
			name: "Zero weight exclusion test - A-Res edge case",
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 30}, // Normal weight, should be selected
				{Pod: pod2, Score: 0},  // Zero weight, never selected in A-Res
			},
			maxPods:    1,
			iterations: 500,
			expectedProbabilities: map[string]float64{
				"pod1": 1.0, // Only pod with positive weight
				"pod2": 0.0, // Zero weight pods are filtered out
			},
			tolerancePercent: 5.0, // ±5% tolerance (should be exact for zero weights)
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
			maxPods:    1,
			iterations: 1000,
			expectedProbabilities: map[string]float64{
				"pod1": 0.345, // Highest weight gets highest probability
				"pod2": 0.310, // High weight gets high probability
				"pod3": 0.172, // Medium weight gets medium probability
				"pod4": 0.103, // Low weight gets low probability
				"pod5": 0.069, // Lowest weight gets lowest probability
			},
			tolerancePercent: 25.0,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			picker := NewWeightedRandomPicker(test.maxPods)
			selectionCounts := make(map[string]int)

			// Initialize selection counters for each pod
			for _, pod := range test.input {
				podName := pod.GetPod().NamespacedName.Name
				selectionCounts[podName] = 0
			}

			// Run multiple iterations to gather statistical data
			for i := 0; i < test.iterations; i++ {
				result := picker.Pick(context.Background(), types.NewCycleState(), test.input)

				// Count selections for probability analysis
				if len(result.TargetPods) > 0 {
					selectedPodName := result.TargetPods[0].GetPod().NamespacedName.Name
					selectionCounts[selectedPodName]++
				}
			}

			// Verify probability distribution
			if test.expectedProbabilities != nil {
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

