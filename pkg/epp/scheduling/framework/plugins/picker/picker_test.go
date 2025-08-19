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

	tests := []struct {
		name           string
		picker         framework.Picker
		input          []*types.ScoredPod
		expectedLength int
	}{
		{
			name:   "Single pod selection with weights",
			picker: NewWeightedRandomPicker(1), // Request only 1 pod using weighted random sampling
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 10}, // 10% probability
				{Pod: pod2, Score: 90}, // 90% probability (highest score, most likely to be selected)
				{Pod: pod3, Score: 0},  // 0% probability (zero weight)
			},
			expectedLength: 1, // Should return exactly 1 pod
		},
		{
			name:   "Multiple pod selection with equal weights",
			picker: NewWeightedRandomPicker(2),
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 50},
				{Pod: pod2, Score: 50},
				{Pod: pod3, Score: 50},
			},
			expectedLength: 2,
		},
		{
			name:   "All pods requested, less than available",
			picker: NewWeightedRandomPicker(5), // Request up to 5 pods, but only 3 candidates available
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 10},
				{Pod: pod2, Score: 20},
				{Pod: pod3, Score: 30},
			},
			expectedLength: 3, // Should return all 3 available pods
		},
		{
			name:   "Zero weight pods fallback to random selection",
			picker: NewWeightedRandomPicker(2),
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 0},
				{Pod: pod2, Score: 0},
				{Pod: pod3, Score: 0},
			},
			expectedLength: 2,
		},
		{
			name:           "Empty input",
			picker:         NewWeightedRandomPicker(1),
			input:          []*types.ScoredPod{},
			expectedLength: 0,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := test.picker.Pick(context.Background(), types.NewCycleState(), test.input)
			got := result.TargetPods

			if len(got) != test.expectedLength {
				t.Errorf("Expected %d pods, got %d", test.expectedLength, len(got))
			}

			// Verify that selected pods are from the input set
			inputPods := make(map[string]bool)
			for _, scoredPod := range test.input {
				inputPods[scoredPod.String()] = true
			}

			for _, targetPod := range got {
				if !inputPods[targetPod.String()] {
					t.Errorf("Selected pod %s not found in input set", targetPod.String())
				}
			}

			// Verify no duplicates
			selectedPods := make(map[string]bool)
			for _, targetPod := range got {
				podKey := targetPod.String()
				if selectedPods[podKey] {
					t.Errorf("Duplicate pod selected: %s", podKey)
				}
				selectedPods[podKey] = true
			}
		})
	}
}

func TestWeightedRandomPickerNormalization(t *testing.T) {
	pod1 := &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}}
	pod2 := &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}}
	pod3 := &types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}}

	tests := []struct {
		name                string
		normalizationType   NormalizationType
		maxRatio            float64
		input               []*types.ScoredPod
		verifyNormalization func(t *testing.T, pods []*types.ScoredPod)
	}{
		{
			name:              "Capping normalization",
			normalizationType: NormalizationCapping,
			maxRatio:          3.0,
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 10},
				{Pod: pod2, Score: 100},
				{Pod: pod3, Score: 20},
			},
			verifyNormalization: func(t *testing.T, pods []*types.ScoredPod) {
				// After capping: min=10, max allowed = 30 (10 * 3.0)
				// pod2 should be capped to 30
				maxScore := 0.0
				for _, pod := range pods {
					if pod.Score > maxScore {
						maxScore = pod.Score
					}
				}
				if maxScore > 30.0 {
					t.Errorf("Expected max score <= 30 after capping, got %f", maxScore)
				}
			},
		},
		{
			name:              "Logarithmic normalization",
			normalizationType: NormalizationLog,
			maxRatio:          3.0,
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 10},
				{Pod: pod2, Score: 100},
				{Pod: pod3, Score: 20},
			},
			verifyNormalization: func(t *testing.T, pods []*types.ScoredPod) {
				// Verify scores are reduced and in log scale
				// Log normalization applies log transformation to reduce score differences
				// Original scores: 10, 100, 20 -> Expected: log(10)≈2.30, log(100)≈4.61, log(20)≈3.00
				// This helps prevent high-scoring pods from dominating the selection
				if pods[0].Score >= 10 || pods[1].Score >= 100 || pods[2].Score >= 20 {
					t.Error("Expected all scores to be reduced after log normalization")
				}
			},
		},
		{
			name:              "Square root normalization",
			normalizationType: NormalizationSqrt,
			maxRatio:          3.0,
			input: []*types.ScoredPod{
				{Pod: pod1, Score: 10},
				{Pod: pod2, Score: 100},
				{Pod: pod3, Score: 20},
			},
			verifyNormalization: func(t *testing.T, pods []*types.ScoredPod) {
				// Verify scores are reduced and in sqrt scale
				// Square root normalization applies sqrt transformation to reduce score differences
				// Original scores: 10, 100, 20 -> Expected: sqrt(10)≈3.16, sqrt(100)=10, sqrt(20)≈4.47
				// This provides a moderate reduction compared to log normalization
				if pods[0].Score >= 10 || pods[1].Score >= 100 || pods[2].Score >= 20 {
					t.Error("Expected all scores to be reduced after sqrt normalization")
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			picker := NewWeightedRandomPicker(3, test.normalizationType, test.maxRatio)

			// Create a copy of input to avoid modifying the original
			inputCopy := make([]*types.ScoredPod, len(test.input))
			for i, pod := range test.input {
				inputCopy[i] = &types.ScoredPod{Pod: pod.Pod, Score: pod.Score}
			}

			result := picker.Pick(context.Background(), types.NewCycleState(), inputCopy)

			// Verify that we got some results
			if len(result.TargetPods) == 0 {
				t.Error("Expected some pods to be selected")
			}

			// Run normalization verification
			test.verifyNormalization(t, inputCopy)
		})
	}
}
