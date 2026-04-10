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

package random

import (
	"context"
	"math"
	"testing"

	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwksched "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

func TestPickRandomPicker(t *testing.T) {
	const (
		testIterations = 10000
		tolerance      = 0.05 // Verify within tolerance ±5%
	)

	endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, nil, nil)
	endpoint2 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, nil, nil)
	endpoint3 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}, nil, nil)

	tests := []struct {
		name    string
		picker  fwksched.Picker
		input   []*fwksched.ScoredEndpoint
		numPods int // expected number of returned endpoints
	}{
		{
			name:   "Single endpoint returned from multiple candidates",
			picker: NewRandomPicker(1),
			input: []*fwksched.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 10},
				{Endpoint: endpoint2, Score: 20},
				{Endpoint: endpoint3, Score: 30},
			},
			numPods: 1,
		},
		{
			name:   "Multiple endpoints returned, more candidates than needed",
			picker: NewRandomPicker(2),
			input: []*fwksched.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 10},
				{Endpoint: endpoint2, Score: 20},
				{Endpoint: endpoint3, Score: 30},
			},
			numPods: 2,
		},
		{
			name:   "Fewer candidates than maxNumOfEndpoints",
			picker: NewRandomPicker(5),
			input: []*fwksched.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 10},
				{Endpoint: endpoint2, Score: 20},
			},
			numPods: 2,
		},
		{
			name:   "Exact number of candidates as maxNumOfEndpoints",
			picker: NewRandomPicker(3),
			input: []*fwksched.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 10},
				{Endpoint: endpoint2, Score: 20},
				{Endpoint: endpoint3, Score: 30},
			},
			numPods: 3,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Verify correct number of endpoints returned
			result := test.picker.Pick(context.Background(), fwksched.NewCycleState(), test.input)
			got := result.TargetEndpoints
			if len(got) != test.numPods {
				t.Errorf("Expected %d endpoints, got %d", test.numPods, len(got))
			}

			// Verify uniform distribution: each endpoint's selection probability
			// should be min(numPods/numCandidates, 1.0), regardless of scores.
			selectionCounts := make(map[string]int)
			for _, ep := range test.input {
				selectionCounts[ep.GetMetadata().NamespacedName.Name] = 0
			}

			for range testIterations {
				res := test.picker.Pick(context.Background(), fwksched.NewCycleState(), test.input)
				for _, ep := range res.TargetEndpoints {
					selectionCounts[ep.GetMetadata().NamespacedName.Name]++
				}
			}

			expectedProb := math.Min(float64(test.numPods)/float64(len(test.input)), 1.0)
			for name, count := range selectionCounts {
				actualProb := float64(count) / float64(testIterations)
				if math.Abs(actualProb-expectedProb) > tolerance {
					t.Errorf("Endpoint %s: expected probability %.3f ±%.1f%%, got %.3f (count: %d/%d)",
						name, expectedProb, tolerance*100, actualProb, count, testIterations)
				}
			}
		})
	}
}
