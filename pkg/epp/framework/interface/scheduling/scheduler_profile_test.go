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

package scheduling

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/uuid"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

func TestSchedulePlugins(t *testing.T) {
	tp1 := &testPlugin{
		TypeRes:   "test1",
		ScoreRes:  0.3,
		FilterRes: []k8stypes.NamespacedName{{Name: "pod1"}, {Name: "pod2"}, {Name: "pod3"}},
	}
	tp2 := &testPlugin{
		TypeRes:   "test2",
		ScoreRes:  0.8,
		FilterRes: []k8stypes.NamespacedName{{Name: "pod1"}, {Name: "pod2"}},
	}
	tp_filterAll := &testPlugin{
		TypeRes:   "filter all",
		FilterRes: []k8stypes.NamespacedName{},
	}
	pickerPlugin := &testPlugin{
		TypeRes: "picker",
		PickRes: k8stypes.NamespacedName{Name: "pod1"},
	}

	tests := []struct {
		name                string
		profile             *SchedulerProfile
		input               []Endpoint
		wantTargetEndpoint  k8stypes.NamespacedName
		targetEndpointScore float64
		// Number of expected endpoints to score (after filter)
		numEndpointsToScore int
		err                 bool
	}{
		{
			name: "all plugins executed successfully, all scorers with same weight",
			profile: NewSchedulerProfile().
				WithFilters(tp1, tp2).
				WithScorers(NewWeightedScorer(tp1, 1), NewWeightedScorer(tp2, 1)).
				WithPicker(pickerPlugin),
			input: []Endpoint{
				&PodMetrics{EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}},
				&PodMetrics{EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}},
				&PodMetrics{EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}},
			},
			wantTargetEndpoint:  k8stypes.NamespacedName{Name: "pod1"},
			targetEndpointScore: 1.1,
			numEndpointsToScore: 2,
			err:                 false,
		},
		{
			name: "all plugins executed successfully, different scorers weights",
			profile: NewSchedulerProfile().
				WithFilters(tp1, tp2).
				WithScorers(NewWeightedScorer(tp1, 60), NewWeightedScorer(tp2, 40)).
				WithPicker(pickerPlugin),
			input: []Endpoint{
				&PodMetrics{EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}},
				&PodMetrics{EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}},
				&PodMetrics{EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}},
			},
			wantTargetEndpoint:  k8stypes.NamespacedName{Name: "pod1"},
			targetEndpointScore: 50,
			numEndpointsToScore: 2,
			err:                 false,
		},
		{
			name: "filter all",
			profile: NewSchedulerProfile().
				WithFilters(tp1, tp_filterAll).
				WithScorers(NewWeightedScorer(tp1, 1), NewWeightedScorer(tp2, 1)).
				WithPicker(pickerPlugin),
			input: []Endpoint{
				&PodMetrics{EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}},
				&PodMetrics{EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}},
				&PodMetrics{EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}},
			},
			numEndpointsToScore: 0,
			err:                 true, // no available endpoints to server after filter all
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Reset all plugins before each new test case.
			for _, plugin := range test.profile.filters {
				plugin.(*testPlugin).reset()
			}
			for _, plugin := range test.profile.scorers {
				plugin.Scorer.(*testPlugin).reset()
			}
			test.profile.picker.(*testPlugin).reset()

			// Initialize the scheduling context
			request := &LLMRequest{
				TargetModel: "test-model",
				RequestId:   uuid.NewString(),
			}
			// Run profile cycle
			got, err := test.profile.Run(context.Background(), request, NewCycleState(), test.input)

			// Validate error state
			if test.err != (err != nil) {
				t.Fatalf("Unexpected error, got %v, want %v", err, test.err)
			}

			if err != nil {
				return
			}

			// Validate output
			wantRes := &ProfileRunResult{
				TargetEndpoints: []Endpoint{
					&PodMetrics{
						EndpointMetadata: &datalayer.EndpointMetadata{NamespacedName: test.wantTargetEndpoint},
					},
				},
			}

			if diff := cmp.Diff(wantRes, got); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
			// Validate plugin execution counts dynamically
			for _, plugin := range test.profile.filters {
				tp, _ := plugin.(*testPlugin)
				if tp.FilterCallCount != 1 {
					t.Errorf("Plugin '%s' Filter() called %d times, expected 1", plugin.TypedName(), tp.FilterCallCount)
				}
			}
			for _, plugin := range test.profile.scorers {
				tp, _ := plugin.Scorer.(*testPlugin)
				if tp.ScoreCallCount != 1 {
					t.Errorf("Plugin '%s' Score() called %d times, expected 1", plugin.TypedName(), tp.ScoreCallCount)
				}
				if test.numEndpointsToScore != tp.NumOfScoredEndpoints {
					t.Errorf("Plugin '%s' Score() called with %d pods, expected %d", plugin.TypedName(), tp.NumOfScoredEndpoints, test.numEndpointsToScore)
				}
			}
			tp, _ := test.profile.picker.(*testPlugin)
			if tp.NumOfPickerCandidates != test.numEndpointsToScore {
				t.Errorf("Picker plugin '%s' Pick() called with %d candidates, expected %d", tp.TypedName(), tp.NumOfPickerCandidates, tp.NumOfScoredEndpoints)
			}
			if tp.PickCallCount != 1 {
				t.Errorf("Picker plugin '%s' Pick() called %d times, expected 1", tp.TypedName(), tp.PickCallCount)
			}
			if tp.WinnerEndpointScore != test.targetEndpointScore {
				t.Errorf("winner pod score %v, expected %v", tp.WinnerEndpointScore, test.targetEndpointScore)
			}
		})
	}
}

// compile-time type assertion
var _ Filter = &testPlugin{}
var _ Scorer = &testPlugin{}
var _ Picker = &testPlugin{}

// testPlugin is an implementation useful in unit tests.
type testPlugin struct {
	typedName             fwkplugin.TypedName
	TypeRes               string
	ScoreCallCount        int
	NumOfScoredEndpoints  int
	ScoreRes              float64
	FilterCallCount       int
	FilterRes             []k8stypes.NamespacedName
	PickCallCount         int
	NumOfPickerCandidates int
	PickRes               k8stypes.NamespacedName
	WinnerEndpointScore   float64
}

func (tp *testPlugin) TypedName() fwkplugin.TypedName {
	return tp.typedName
}

func (tp *testPlugin) Category() ScorerCategory {
	return Distribution
}

func (tp *testPlugin) Filter(_ context.Context, _ *CycleState, _ *LLMRequest, endpoints []Endpoint) []Endpoint {
	tp.FilterCallCount++
	return findEndpoints(endpoints, tp.FilterRes...)

}

func (tp *testPlugin) Score(_ context.Context, _ *CycleState, _ *LLMRequest, endpoints []Endpoint) map[Endpoint]float64 {
	tp.ScoreCallCount++
	scoredEndpoints := make(map[Endpoint]float64, len(endpoints))
	for _, endpoint := range endpoints {
		scoredEndpoints[endpoint] += tp.ScoreRes
	}
	tp.NumOfScoredEndpoints = len(scoredEndpoints)
	return scoredEndpoints
}

func (tp *testPlugin) Pick(_ context.Context, _ *CycleState, scoredEndpoints []*ScoredEndpoint) *ProfileRunResult {
	tp.PickCallCount++
	tp.NumOfPickerCandidates = len(scoredEndpoints)

	winnerEndpoints := []Endpoint{}
	for _, scoredEndpoint := range scoredEndpoints {
		if scoredEndpoint.GetMetadata().NamespacedName.String() == tp.PickRes.String() {
			winnerEndpoints = append(winnerEndpoints, scoredEndpoint.Endpoint)
			tp.WinnerEndpointScore = scoredEndpoint.Score
		}
	}

	return &ProfileRunResult{TargetEndpoints: winnerEndpoints}
}

func (tp *testPlugin) reset() {
	tp.FilterCallCount = 0
	tp.ScoreCallCount = 0
	tp.NumOfScoredEndpoints = 0
	tp.PickCallCount = 0
	tp.NumOfPickerCandidates = 0
}

func findEndpoints(endpoints []Endpoint, names ...k8stypes.NamespacedName) []Endpoint {
	res := []Endpoint{}
	for _, endpoint := range endpoints {
		for _, name := range names {
			if endpoint.GetMetadata().NamespacedName.String() == name.String() {
				res = append(res, endpoint)
			}
		}
	}
	return res
}
