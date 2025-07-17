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

package framework

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/uuid"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
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
		name           string
		profile        *SchedulerProfile
		input          []types.Pod
		wantTargetPod  k8stypes.NamespacedName
		targetPodScore float64
		// Number of expected pods to score (after filter)
		numPodsToScore int
		err            bool
	}{
		{
			name: "all plugins executed successfully, all scorers with same weight",
			profile: NewSchedulerProfile().
				WithFilters(tp1, tp2).
				WithScorers(NewWeightedScorer(tp1, 1), NewWeightedScorer(tp2, 1)).
				WithPicker(pickerPlugin).
				WithPostCyclePlugins(tp1, tp2),
			input: []types.Pod{
				&types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}},
				&types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}},
				&types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}},
			},
			wantTargetPod:  k8stypes.NamespacedName{Name: "pod1"},
			targetPodScore: 1.1,
			numPodsToScore: 2,
			err:            false,
		},
		{
			name: "all plugins executed successfully, different scorers weights",
			profile: NewSchedulerProfile().
				WithFilters(tp1, tp2).
				WithScorers(NewWeightedScorer(tp1, 60), NewWeightedScorer(tp2, 40)).
				WithPicker(pickerPlugin).
				WithPostCyclePlugins(tp1, tp2),
			input: []types.Pod{
				&types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}},
				&types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}},
				&types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}},
			},
			wantTargetPod:  k8stypes.NamespacedName{Name: "pod1"},
			targetPodScore: 50,
			numPodsToScore: 2,
			err:            false,
		},
		{
			name: "filter all",
			profile: NewSchedulerProfile().
				WithFilters(tp1, tp_filterAll).
				WithScorers(NewWeightedScorer(tp1, 1), NewWeightedScorer(tp2, 1)).
				WithPicker(pickerPlugin).
				WithPostCyclePlugins(tp1, tp2),
			input: []types.Pod{
				&types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}},
				&types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}},
				&types.PodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}},
			},
			numPodsToScore: 0,
			err:            true, // no available pods to server after filter all
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
			for _, plugin := range test.profile.postCyclePlugins {
				plugin.(*testPlugin).reset()
			}

			// Initialize the scheduling context
			request := &types.LLMRequest{
				TargetModel: "test-model",
				RequestId:   uuid.NewString(),
			}
			// Run profile cycle
			got, err := test.profile.Run(context.Background(), request, types.NewCycleState(), test.input)

			// Validate error state
			if test.err != (err != nil) {
				t.Fatalf("Unexpected error, got %v, want %v", err, test.err)
			}

			if err != nil {
				return
			}

			// Validate output
			wantRes := &types.ProfileRunResult{
				TargetPods: []types.Pod{
					&types.PodMetrics{
						Pod: &backend.Pod{NamespacedName: test.wantTargetPod},
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
				if test.numPodsToScore != tp.NumOfScoredPods {
					t.Errorf("Plugin '%s' Score() called with %d pods, expected %d", plugin.TypedName(), tp.NumOfScoredPods, test.numPodsToScore)
				}
			}
			tp, _ := test.profile.picker.(*testPlugin)
			if tp.NumOfPickerCandidates != test.numPodsToScore {
				t.Errorf("Picker plugin '%s' Pick() called with %d candidates, expected %d", tp.TypedName(), tp.NumOfPickerCandidates, tp.NumOfScoredPods)
			}
			if tp.PickCallCount != 1 {
				t.Errorf("Picker plugin '%s' Pick() called %d times, expected 1", tp.TypedName(), tp.PickCallCount)
			}
			if tp.WinnerPodScore != test.targetPodScore {
				t.Errorf("winner pod score %v, expected %v", tp.WinnerPodScore, test.targetPodScore)
			}
			for _, plugin := range test.profile.postCyclePlugins {
				tp, _ := plugin.(*testPlugin)
				if tp.PostCycleCallCount != 1 {
					t.Errorf("Plugin '%s' PostCycle() called %d times, expected 1", plugin.TypedName(), tp.PostCycleCallCount)
				}
			}
		})
	}
}

// compile-time type assertion
var _ Filter = &testPlugin{}
var _ Scorer = &testPlugin{}
var _ Picker = &testPlugin{}
var _ PostCycle = &testPlugin{}

// testPlugin is an implementation useful in unit tests.
type testPlugin struct {
	typedName             plugins.TypedName
	TypeRes               string
	ScoreCallCount        int
	NumOfScoredPods       int
	ScoreRes              float64
	FilterCallCount       int
	FilterRes             []k8stypes.NamespacedName
	PostCycleCallCount    int
	PickCallCount         int
	NumOfPickerCandidates int
	PickRes               k8stypes.NamespacedName
	WinnerPodScore        float64
}

func (tp *testPlugin) TypedName() plugins.TypedName {
	return tp.typedName
}

func (tp *testPlugin) Filter(_ context.Context, _ *types.CycleState, _ *types.LLMRequest, pods []types.Pod) []types.Pod {
	tp.FilterCallCount++
	return findPods(pods, tp.FilterRes...)

}

func (tp *testPlugin) Score(_ context.Context, _ *types.CycleState, _ *types.LLMRequest, pods []types.Pod) map[types.Pod]float64 {
	tp.ScoreCallCount++
	scoredPods := make(map[types.Pod]float64, len(pods))
	for _, pod := range pods {
		scoredPods[pod] += tp.ScoreRes
	}
	tp.NumOfScoredPods = len(scoredPods)
	return scoredPods
}

func (tp *testPlugin) Pick(_ context.Context, _ *types.CycleState, scoredPods []*types.ScoredPod) *types.ProfileRunResult {
	tp.PickCallCount++
	tp.NumOfPickerCandidates = len(scoredPods)

	winnerPods := []types.Pod{}
	for _, scoredPod := range scoredPods {
		if scoredPod.GetPod().NamespacedName.String() == tp.PickRes.String() {
			winnerPods = append(winnerPods, scoredPod.Pod)
			tp.WinnerPodScore = scoredPod.Score
		}
	}

	return &types.ProfileRunResult{TargetPods: winnerPods}
}

func (tp *testPlugin) PostCycle(_ context.Context, _ *types.CycleState, res *types.ProfileRunResult) {
	tp.PostCycleCallCount++
}

func (tp *testPlugin) reset() {
	tp.FilterCallCount = 0
	tp.ScoreCallCount = 0
	tp.NumOfScoredPods = 0
	tp.PostCycleCallCount = 0
	tp.PickCallCount = 0
	tp.NumOfPickerCandidates = 0
}

func findPods(pods []types.Pod, names ...k8stypes.NamespacedName) []types.Pod {
	res := []types.Pod{}
	for _, pod := range pods {
		for _, name := range names {
			if pod.GetPod().NamespacedName.String() == name.String() {
				res = append(res, pod)
			}
		}
	}
	return res
}

func TestSchedulerProfileLogConfiguration(t *testing.T) {
	// Create test plugins
	tp1 := &testPlugin{TypeRes: "test-filter"}
	tp2 := &testPlugin{TypeRes: "test-scorer"}
	tp3 := &testPlugin{TypeRes: "test-picker"}
	tp4 := &testPlugin{TypeRes: "test-postcycle"}

	// Create a scheduler profile with various plugins
	profile := NewSchedulerProfile().
		WithFilters(tp1).
		WithScorers(NewWeightedScorer(tp2, 10)).
		WithPicker(tp3).
		WithPostCyclePlugins(tp4)

	// Test LogConfiguration doesn't panic and can be called
	// We can't easily test the actual log output without complex setup,
	// but we can ensure the method executes without errors
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("LogConfiguration panicked: %v", r)
		}
	}()

	// Use a test logger context
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	logger := log.FromContext(ctx)

	profile.LogConfiguration(logger, "test-profile")
}
