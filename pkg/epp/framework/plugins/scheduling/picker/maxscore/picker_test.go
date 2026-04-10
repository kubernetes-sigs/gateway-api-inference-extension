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

package maxscore

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwksched "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

func TestPickMaxScorePicker(t *testing.T) {
	endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, nil, nil)
	endpoint2 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, nil, nil)
	endpoint3 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}, nil, nil)

	tests := []struct {
		name               string
		picker             fwksched.Picker
		input              []*fwksched.ScoredEndpoint
		output             []fwksched.Endpoint
		tieBreakCandidates int // tie break is random, specify how many candidate with max score
	}{
		{
			name:   "Single max score",
			picker: NewMaxScorePicker(1),
			input: []*fwksched.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 10},
				{Endpoint: endpoint2, Score: 25},
				{Endpoint: endpoint3, Score: 15},
			},
			output: []fwksched.Endpoint{
				&fwksched.ScoredEndpoint{Endpoint: endpoint2, Score: 25},
			},
		},
		{
			name:   "Multiple max scores, all are equally scored",
			picker: NewMaxScorePicker(2),
			input: []*fwksched.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 50},
				{Endpoint: endpoint2, Score: 50},
				{Endpoint: endpoint3, Score: 30},
			},
			output: []fwksched.Endpoint{
				&fwksched.ScoredEndpoint{Endpoint: endpoint1, Score: 50},
				&fwksched.ScoredEndpoint{Endpoint: endpoint2, Score: 50},
			},
			tieBreakCandidates: 2,
		},
		{
			name:   "Multiple results sorted by highest score, more pods than needed",
			picker: NewMaxScorePicker(2),
			input: []*fwksched.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 20},
				{Endpoint: endpoint2, Score: 25},
				{Endpoint: endpoint3, Score: 30},
			},
			output: []fwksched.Endpoint{
				&fwksched.ScoredEndpoint{Endpoint: endpoint3, Score: 30},
				&fwksched.ScoredEndpoint{Endpoint: endpoint2, Score: 25},
			},
		},
		{
			name:   "Multiple results sorted by highest score, less pods than needed",
			picker: NewMaxScorePicker(4), // picker is required to return 4 pods at most, but we have only 3.
			input: []*fwksched.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 20},
				{Endpoint: endpoint2, Score: 25},
				{Endpoint: endpoint3, Score: 30},
			},
			output: []fwksched.Endpoint{
				&fwksched.ScoredEndpoint{Endpoint: endpoint3, Score: 30},
				&fwksched.ScoredEndpoint{Endpoint: endpoint2, Score: 25},
				&fwksched.ScoredEndpoint{Endpoint: endpoint1, Score: 20},
			},
		},
		{
			name:   "Multiple results sorted by highest score, num of pods exactly needed",
			picker: NewMaxScorePicker(3), // picker is required to return 3 pods at most, we have only 3.
			input: []*fwksched.ScoredEndpoint{
				{Endpoint: endpoint1, Score: 30},
				{Endpoint: endpoint2, Score: 25},
				{Endpoint: endpoint3, Score: 30},
			},
			output: []fwksched.Endpoint{
				&fwksched.ScoredEndpoint{Endpoint: endpoint1, Score: 30},
				&fwksched.ScoredEndpoint{Endpoint: endpoint3, Score: 30},
				&fwksched.ScoredEndpoint{Endpoint: endpoint2, Score: 25},
			},
			tieBreakCandidates: 2,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := test.picker.Pick(context.Background(), fwksched.NewCycleState(), test.input)
			got := result.TargetEndpoints

			if test.tieBreakCandidates > 0 {
				testMaxScoredEndpoints := test.output[:test.tieBreakCandidates]
				gotMaxScoredEndpoints := got[:test.tieBreakCandidates]
				diff := cmp.Diff(testMaxScoredEndpoints, gotMaxScoredEndpoints, cmpopts.SortSlices(func(a, b fwksched.Endpoint) bool {
					return a.String() < b.String() // predictable order within the endpoints with equal scores
				}), cmp.Comparer(fwksched.ScoredEndpointComparer))
				if diff != "" {
					t.Errorf("Unexpected output (-want +got): %v", diff)
				}
				test.output = test.output[test.tieBreakCandidates:]
				got = got[test.tieBreakCandidates:]
			}

			if diff := cmp.Diff(test.output, got, cmp.Comparer(fwksched.ScoredEndpointComparer)); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}
