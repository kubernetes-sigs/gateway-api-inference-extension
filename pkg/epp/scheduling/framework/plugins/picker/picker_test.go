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
	"math"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

func TestPickMaxScorePicker(t *testing.T) {
	endpoint1 := &types.PodMetrics{EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}}
	endpoint2 := &types.PodMetrics{EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}}
	endpoint3 := &types.PodMetrics{EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}}

	tests := []struct {
		name               string
		picker             framework.Picker
		input              []*types.ScoredEndpoint
		output             []types.Endpoint
		tieBreakCandidates int // tie break is random, specify how many candidate with max score
	}{
		{
			name:   "Single max score",
			picker: NewMaxScorePicker(1),
			input: []*types.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 10},
				{Endpoint: endpoint2, Score: 25},
				{Endpoint: endpoint3, Score: 15},
			},
			output: []types.Endpoint{
				&types.ScoredEndpoint{Endpoint: endpoint2, Score: 25},
			},
		},
		{
			name:   "Multiple max scores, all are equally scored",
			picker: NewMaxScorePicker(2),
			input: []*types.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 50},
				{Endpoint: endpoint2, Score: 50},
				{Endpoint: endpoint3, Score: 30},
			},
			output: []types.Endpoint{
				&types.ScoredEndpoint{Endpoint: endpoint1, Score: 50},
				&types.ScoredEndpoint{Endpoint: endpoint2, Score: 50},
			},
			tieBreakCandidates: 2,
		},
		{
			name:   "Multiple results sorted by highest score, more pods than needed",
			picker: NewMaxScorePicker(2),
			input: []*types.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 20},
				{Endpoint: endpoint2, Score: 25},
				{Endpoint: endpoint3, Score: 30},
			},
			output: []types.Endpoint{
				&types.ScoredEndpoint{Endpoint: endpoint3, Score: 30},
				&types.ScoredEndpoint{Endpoint: endpoint2, Score: 25},
			},
		},
		{
			name:   "Multiple results sorted by highest score, less pods than needed",
			picker: NewMaxScorePicker(4), // picker is required to return 4 pods at most, but we have only 3.
			input: []*types.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 20},
				{Endpoint: endpoint2, Score: 25},
				{Endpoint: endpoint3, Score: 30},
			},
			output: []types.Endpoint{
				&types.ScoredEndpoint{Endpoint: endpoint3, Score: 30},
				&types.ScoredEndpoint{Endpoint: endpoint2, Score: 25},
				&types.ScoredEndpoint{Endpoint: endpoint1, Score: 20},
			},
		},
		{
			name:   "Multiple results sorted by highest score, num of pods exactly needed",
			picker: NewMaxScorePicker(3), // picker is required to return 3 pods at most, we have only 3.
			input: []*types.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 30},
				{Endpoint: endpoint2, Score: 25},
				{Endpoint: endpoint3, Score: 30},
			},
			output: []types.Endpoint{
				&types.ScoredEndpoint{Endpoint: endpoint1, Score: 30},
				&types.ScoredEndpoint{Endpoint: endpoint3, Score: 30},
				&types.ScoredEndpoint{Endpoint: endpoint2, Score: 25},
			},
			tieBreakCandidates: 2,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := test.picker.Pick(context.Background(), types.NewCycleState(), test.input)
			got := result.TargetEndpoints

			if test.tieBreakCandidates > 0 {
				testMaxScoredEndpoints := test.output[:test.tieBreakCandidates]
				gotMaxScoredEndpoints := got[:test.tieBreakCandidates]
				diff := cmp.Diff(testMaxScoredEndpoints, gotMaxScoredEndpoints, cmpopts.SortSlices(func(a, b types.Endpoint) bool {
					return a.String() < b.String() // predictable order within the endpoints with equal scores
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
	const (
		testIterations = 10000
		tolerance      = 0.05 // Verify within tolerance ±5%
	)

	endpoint1 := &types.PodMetrics{EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}}
	endpoint2 := &types.PodMetrics{EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}}
	endpoint3 := &types.PodMetrics{EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}}
	endpoint4 := &types.PodMetrics{EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod4"}}}
	endpoint5 := &types.PodMetrics{EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod5"}}}

	// A-Res algorithm uses U^(1/w) transformation which introduces statistical variance
	// beyond simple proportional sampling. Generous tolerance is required to prevent
	// flaky tests in CI environments, especially for multi-tier weights.
	tests := []struct {
		name    string
		input   []*types.ScoredEndpoint
		maxPods int // maxNumOfEndpoints for this test
	}{
		{
			name: "High weight dominance test",
			input: []*types.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 10}, // Lower weight
				{Endpoint: endpoint2, Score: 90}, // Higher weight (should dominate)
			},
			maxPods: 1,
		},
		{
			name: "Equal weights test - A-Res uniform distribution",
			input: []*types.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 100}, // Equal weights (higher values for better numerical precision)
				{Endpoint: endpoint2, Score: 100}, // Equal weights should yield uniform distribution
				{Endpoint: endpoint3, Score: 100}, // Equal weights in A-Res
			},
			maxPods: 1,
		},
		{
			name: "Zero weight exclusion test - A-Res edge case",
			input: []*types.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 30}, // Normal weight, should be selected
				{Endpoint: endpoint2, Score: 0},  // Zero weight, never selected in A-Res
			},
			maxPods: 1,
		},
		{
			name: "Multi-tier weighted test - A-Res complex distribution",
			input: []*types.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 100}, // Highest weight
				{Endpoint: endpoint2, Score: 90},  // High weight
				{Endpoint: endpoint3, Score: 50},  // Medium weight
				{Endpoint: endpoint4, Score: 30},  // Low weight
				{Endpoint: endpoint5, Score: 20},  // Lowest weight
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
			for _, endpoint := range test.input {
				podName := endpoint.GetMetadata().NamespacedName.Name
				if totalScore > 0 {
					expectedProbabilities[podName] = endpoint.Score / totalScore
				} else {
					expectedProbabilities[podName] = 0.0
				}
			}

			// Initialize selection counters for each pod
			selectionCounts := make(map[string]int)
			for _, endpoint := range test.input {
				endpointName := endpoint.GetMetadata().NamespacedName.Name
				selectionCounts[endpointName] = 0
			}

			// Run multiple iterations to gather statistical data
			for range testIterations {
				result := picker.Pick(context.Background(), types.NewCycleState(), test.input)

				// Count selections for probability analysis
				selectedEndpointName := result.TargetEndpoints[0].GetMetadata().NamespacedName.Name
				selectionCounts[selectedEndpointName]++
			}

			// Verify probability distribution
			for endpointName, expectedProb := range expectedProbabilities {
				actualCount := selectionCounts[endpointName]
				actualProb := float64(actualCount) / float64(testIterations)

				if math.Abs(actualProb-expectedProb) > tolerance {
					t.Errorf("Endpoint %s: expected probability %.3f ±%.1f%%, got %.3f (count: %d/%d)",
						endpointName, expectedProb, tolerance*100, actualProb, actualCount, testIterations)
				} else {
					t.Logf("Endpoint %s: expected %.3f, got %.3f (count: %d/%d) ✓",
						endpointName, expectedProb, actualProb, actualCount, testIterations)
				}
			}
		})
	}
}
