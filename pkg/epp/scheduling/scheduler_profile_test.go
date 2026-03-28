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
	"strings"
	"testing"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"

	"github.com/google/go-cmp/cmp"
	"github.com/google/uuid"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwksched "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
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
		input               []fwksched.Endpoint
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
			input: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, nil, nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, nil, nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}, nil, nil),
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
			input: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, nil, nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, nil, nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}, nil, nil),
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
			input: []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, nil, nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, nil, nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}, nil, nil),
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
			request := &requesthandling.InferenceRequest{
				LLM: &requesthandling.LLMRequest{
					TargetModel: "test-model",
					RequestId:   uuid.NewString(),
				},
			}
			// Run profile cycle
			got, err := test.profile.Run(context.Background(), request, fwksched.NewCycleState(), test.input)

			// Validate error state
			if test.err != (err != nil) {
				t.Fatalf("Unexpected error, got %v, want %v", err, test.err)
			}

			if err != nil {
				return
			}

			// Validate output
			wantRes := &fwksched.ProfileRunResult{
				TargetEndpoints: []fwksched.Endpoint{
					fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: test.wantTargetEndpoint}, nil, nil),
				},
			}

			if diff := cmp.Diff(wantRes, got, cmp.Comparer(fwksched.EndpointComparer)); diff != "" {
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
var _ fwksched.Filter = &testPlugin{}
var _ fwksched.Scorer = &testPlugin{}
var _ fwksched.Picker = &testPlugin{}

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

func (tp *testPlugin) Category() fwksched.ScorerCategory {
	return fwksched.Distribution
}

func (tp *testPlugin) Filter(_ context.Context, _ *fwksched.CycleState, _ *requesthandling.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	tp.FilterCallCount++
	return findEndpoints(endpoints, tp.FilterRes...)

}

func (tp *testPlugin) Score(_ context.Context, _ *fwksched.CycleState, _ *requesthandling.InferenceRequest, endpoints []fwksched.Endpoint) map[fwksched.Endpoint]float64 {
	tp.ScoreCallCount++
	scoredEndpoints := make(map[fwksched.Endpoint]float64, len(endpoints))
	for _, endpoint := range endpoints {
		scoredEndpoints[endpoint] += tp.ScoreRes
	}
	tp.NumOfScoredEndpoints = len(scoredEndpoints)
	return scoredEndpoints
}

func (tp *testPlugin) Pick(_ context.Context, _ *fwksched.CycleState, scoredEndpoints []*fwksched.ScoredEndpoint) *fwksched.ProfileRunResult {
	tp.PickCallCount++
	tp.NumOfPickerCandidates = len(scoredEndpoints)

	winnerEndpoints := []fwksched.Endpoint{}
	for _, scoredEndpoint := range scoredEndpoints {
		if scoredEndpoint.GetMetadata().NamespacedName.String() == tp.PickRes.String() {
			winnerEndpoints = append(winnerEndpoints, scoredEndpoint.Endpoint)
			tp.WinnerEndpointScore = scoredEndpoint.Score
		}
	}

	return &fwksched.ProfileRunResult{TargetEndpoints: winnerEndpoints}
}

func (tp *testPlugin) reset() {
	tp.FilterCallCount = 0
	tp.ScoreCallCount = 0
	tp.NumOfScoredEndpoints = 0
	tp.PickCallCount = 0
	tp.NumOfPickerCandidates = 0
}

func TestAddPlugins(t *testing.T) {
	tests := []struct {
		name        string
		plugins     []fwkplugin.Plugin
		wantFilters int
		wantScorers int
		wantPicker  bool
		wantErr     bool
		errContains string
	}{
		{
			name: "add WeightedScorer that also implements Filter and Picker",
			plugins: []fwkplugin.Plugin{
				NewWeightedScorer(&testPlugin{TypeRes: "multi"}, 1.0),
			},
			wantFilters: 1,
			wantScorers: 1,
			wantPicker:  true,
		},
		{
			name: "add plugin that only implements Filter",
			plugins: []fwkplugin.Plugin{
				&filterOnlyPlugin{typedName: fwkplugin.TypedName{Name: "filter-only"}},
			},
			wantFilters: 1,
			wantScorers: 0,
			wantPicker:  false,
		},
		{
			name: "error when adding Scorer without weight",
			plugins: []fwkplugin.Plugin{
				&testPlugin{TypeRes: "bare-scorer"},
			},
			wantErr:     true,
			errContains: "without a weight",
		},
		{
			name: "error when adding duplicate picker",
			plugins: []fwkplugin.Plugin{
				NewWeightedScorer(&testPlugin{TypeRes: "picker1"}, 1.0),
				NewWeightedScorer(&testPlugin{TypeRes: "picker2"}, 1.0),
			},
			wantErr:     true,
			errContains: "already have a registered picker",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			profile := NewSchedulerProfile()
			err := profile.AddPlugins(test.plugins...)

			if test.wantErr {
				if err == nil {
					t.Fatalf("expected error but got nil")
				}
				if test.errContains != "" && !strings.Contains(err.Error(), test.errContains) {
					t.Errorf("error %q does not contain %q", err.Error(), test.errContains)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(profile.filters) != test.wantFilters {
				t.Errorf("got %d filters, want %d", len(profile.filters), test.wantFilters)
			}
			if len(profile.scorers) != test.wantScorers {
				t.Errorf("got %d scorers, want %d", len(profile.scorers), test.wantScorers)
			}
			if test.wantPicker && profile.picker == nil {
				t.Errorf("expected picker to be set")
			}
			if !test.wantPicker && profile.picker != nil {
				t.Errorf("expected picker to be nil")
			}
		})
	}
}

func TestSchedulerProfileString(t *testing.T) {
	tp1 := &testPlugin{TypeRes: "test1"}
	tp2 := &testPlugin{TypeRes: "test2"}
	pickerPlugin := &testPlugin{TypeRes: "picker"}

	profile := NewSchedulerProfile().
		WithFilters(tp1, tp2).
		WithScorers(NewWeightedScorer(tp1, 1.5), NewWeightedScorer(tp2, 2.0)).
		WithPicker(pickerPlugin)

	result := profile.String()

	// Verify the string contains filter, scorer, and picker info
	if !strings.Contains(result, "Filters:") {
		t.Errorf("String() missing Filters section: %s", result)
	}
	if !strings.Contains(result, "Scorers:") {
		t.Errorf("String() missing Scorers section: %s", result)
	}
	if !strings.Contains(result, "Picker:") {
		t.Errorf("String() missing Picker section: %s", result)
	}
}

func TestEnforceScoreRange(t *testing.T) {
	tests := []struct {
		name  string
		score float64
		want  float64
	}{
		{name: "negative score clamped to 0", score: -0.5, want: 0},
		{name: "score above 1 clamped to 1", score: 1.5, want: 1},
		{name: "score at 0 stays 0", score: 0, want: 0},
		{name: "score at 1 stays 1", score: 1, want: 1},
		{name: "score in range stays unchanged", score: 0.5, want: 0.5},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := enforceScoreRange(test.score)
			if got != test.want {
				t.Errorf("enforceScoreRange(%v) = %v, want %v", test.score, got, test.want)
			}
		})
	}
}

func TestRunWithOutOfRangeScores(t *testing.T) {
	// Scorer that returns negative score
	negativeScorer := &testPlugin{
		TypeRes:   "negative",
		ScoreRes:  -0.5,
		FilterRes: []k8stypes.NamespacedName{{Name: "pod1"}},
	}
	// Scorer that returns score > 1
	overScorer := &testPlugin{
		TypeRes:   "over",
		ScoreRes:  1.5,
		FilterRes: []k8stypes.NamespacedName{{Name: "pod1"}},
	}
	pickerPlugin := &testPlugin{
		TypeRes: "picker",
		PickRes: k8stypes.NamespacedName{Name: "pod1"},
	}

	profile := NewSchedulerProfile().
		WithFilters().
		WithScorers(NewWeightedScorer(negativeScorer, 1), NewWeightedScorer(overScorer, 1)).
		WithPicker(pickerPlugin)

	input := []fwksched.Endpoint{
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, nil, nil),
	}

	request := &requesthandling.InferenceRequest{
		LLM: &requesthandling.LLMRequest{
			TargetModel: "test-model",
			RequestId:   uuid.NewString(),
		},
	}

	_, err := profile.Run(context.Background(), request, fwksched.NewCycleState(), input)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// negative score clamped to 0, over score clamped to 1: total = 0*1 + 1*1 = 1.0
	if pickerPlugin.WinnerEndpointScore != 1.0 {
		t.Errorf("expected winner score 1.0, got %v", pickerPlugin.WinnerEndpointScore)
	}
}

// filterOnlyPlugin implements only the Filter interface (not Scorer or Picker).
type filterOnlyPlugin struct {
	typedName fwkplugin.TypedName
}

func (p *filterOnlyPlugin) TypedName() fwkplugin.TypedName {
	return p.typedName
}

func (p *filterOnlyPlugin) Filter(_ context.Context, _ *fwksched.CycleState, _ *requesthandling.InferenceRequest, endpoints []fwksched.Endpoint) []fwksched.Endpoint {
	return endpoints
}

func findEndpoints(endpoints []fwksched.Endpoint, names ...k8stypes.NamespacedName) []fwksched.Endpoint {
	res := []fwksched.Endpoint{}
	for _, endpoint := range endpoints {
		for _, name := range names {
			if endpoint.GetMetadata().NamespacedName.String() == name.String() {
				res = append(res, endpoint)
			}
		}
	}
	return res
}
