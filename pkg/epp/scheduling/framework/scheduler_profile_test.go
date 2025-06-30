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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics" // Import config for thresholds
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

func TestSchedulePlugins(t *testing.T) {
	tp1 := newTestPlugin("test1", 0.3,
		[]k8stypes.NamespacedName{{Name: "pod1"}, {Name: "pod2"}, {Name: "pod3"}},
		k8stypes.NamespacedName{})
	tp2 := newTestPlugin("test2", 0.8,
		[]k8stypes.NamespacedName{{Name: "pod1"}, {Name: "pod2"}},
		k8stypes.NamespacedName{})
	tp_filterAll := newTestPlugin("filter all", 0.0,
		[]k8stypes.NamespacedName{}, k8stypes.NamespacedName{})
	pickerPlugin := newTestPlugin("picker", 0.0,
		[]k8stypes.NamespacedName{}, k8stypes.NamespacedName{Name: "pod1"})

	tests := []struct {
		name           string
		profile        *SchedulerProfile
		input          []backendmetrics.PodMetrics
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
			input: []backendmetrics.PodMetrics{
				&backendmetrics.FakePodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}},
				&backendmetrics.FakePodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}},
				&backendmetrics.FakePodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}},
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
			input: []backendmetrics.PodMetrics{
				&backendmetrics.FakePodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}},
				&backendmetrics.FakePodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}},
				&backendmetrics.FakePodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}},
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
			input: []backendmetrics.PodMetrics{
				&backendmetrics.FakePodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}},
				&backendmetrics.FakePodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}},
				&backendmetrics.FakePodMetrics{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}},
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
			got, err := test.profile.Run(context.Background(), request, types.NewCycleState(), types.ToSchedulerPodMetrics(test.input))

			// Validate error state
			if test.err != (err != nil) {
				t.Fatalf("Unexpected error, got %v, want %v", err, test.err)
			}

			if err != nil {
				return
			}

			// Validate output
			wantPod := &types.PodMetrics{
				Pod: &backend.Pod{NamespacedName: test.wantTargetPod, Labels: make(map[string]string)},
			}
			wantRes := &types.ProfileRunResult{
				TargetPod: wantPod,
			}

			if diff := cmp.Diff(wantRes, got); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
			// Validate plugin execution counts dynamically
			for _, plugin := range test.profile.filters {
				tp, _ := plugin.(*testPlugin)
				if tp.FilterCallCount != 1 {
					t.Errorf("Plugin %s Filter() called %d times, expected 1", plugin.GetTypedName(), tp.FilterCallCount)
				}
			}
			for _, plugin := range test.profile.scorers {
				tp, _ := plugin.Scorer.(*testPlugin)
				if tp.ScoreCallCount != 1 {
					t.Errorf("Plugin %s Score() called %d times, expected 1", plugin.GetTypedName(), tp.ScoreCallCount)
				}
				if test.numPodsToScore != tp.NumOfScoredPods {
					t.Errorf("Plugin %s Score() called with %d pods, expected %d", plugin.GetTypedName(), tp.NumOfScoredPods, test.numPodsToScore)
				}
			}
			tp, _ := test.profile.picker.(*testPlugin)
			if tp.NumOfPickerCandidates != test.numPodsToScore {
				t.Errorf("Picker plugin %s Pick() called with %d candidates, expected %d", tp.GetTypedName(), tp.NumOfPickerCandidates, tp.NumOfScoredPods)
			}
			if tp.PickCallCount != 1 {
				t.Errorf("Picker plugin %s Pick() called %d times, expected 1", tp.GetTypedName(), tp.PickCallCount)
			}
			if tp.WinnerPodScore != test.targetPodScore {
				t.Errorf("winner pod score %v, expected %v", tp.WinnerPodScore, test.targetPodScore)
			}
			for _, plugin := range test.profile.postCyclePlugins {
				tp, _ := plugin.(*testPlugin)
				if tp.PostScheduleCallCount != 1 {
					t.Errorf("Plugin %s PostSchedule() called %d times, expected 1", plugin.GetTypedName(), tp.PostScheduleCallCount)
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
	plugins.TypedName
	TypeRes               string
	ScoreCallCount        int
	NumOfScoredPods       int
	ScoreRes              float64
	FilterCallCount       int
	FilterRes             []k8stypes.NamespacedName
	PostScheduleCallCount int
	PickCallCount         int
	NumOfPickerCandidates int
	PickRes               k8stypes.NamespacedName
	WinnerPodScore        float64
}

func newTestPlugin(typeRes string, score float64, pruned []k8stypes.NamespacedName,
	target k8stypes.NamespacedName) *testPlugin {
	return &testPlugin{
		TypedName: plugins.TypedName{Type: typeRes, Name: "test-plugin"},
		ScoreRes:  score,
		FilterRes: pruned,
		PickRes:   target,
	}
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

	var winnerPod types.Pod
	for _, scoredPod := range scoredPods {
		if scoredPod.GetPod().NamespacedName.String() == tp.PickRes.String() {
			winnerPod = scoredPod.Pod
			tp.WinnerPodScore = scoredPod.Score
		}
	}

	return &types.ProfileRunResult{TargetPod: winnerPod}
}

func (tp *testPlugin) PostCycle(_ context.Context, _ *types.CycleState, res *types.ProfileRunResult) {
	tp.PostScheduleCallCount++
}

func (tp *testPlugin) reset() {
	tp.FilterCallCount = 0
	tp.ScoreCallCount = 0
	tp.NumOfScoredPods = 0
	tp.PostScheduleCallCount = 0
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
