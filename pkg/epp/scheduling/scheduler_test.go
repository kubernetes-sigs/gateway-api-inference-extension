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
	k8stypes "k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

// fakeDataStore is a mock implementation of the Datastore interface for
// testing.
type fakeDataStore struct {
	pods []*backendmetrics.FakePodMetrics
}

// PodGetAll returns all pod metrics from the fake datastore.
func (fds *fakeDataStore) PodGetAll() []backendmetrics.PodMetrics {
	pm := make([]backendmetrics.PodMetrics, 0, len(fds.pods))
	for _, pod := range fds.pods {
		pm = append(pm, pod)
	}
	return pm
}

// TestSchedule tests the behavior of the scheduler with its default
// configuration.
func TestSchedule(t *testing.T) {
	// Define common pod metrics for reusability in test cases.
	// Note: Cloned before use in test cases to avoid shared state issues if
	// metrics were mutable.
	pod1MetricsNaked := &backendmetrics.Metrics{
		WaitingQueueSize:    0,
		KVCacheUsagePercent: 0.2,
		MaxActiveModels:     2,
		ActiveModels:        map[string]int{"foo": 1, "bar": 1},
		WaitingModels:       make(map[string]int),
	}
	pod2MetricsNaked := &backendmetrics.Metrics{
		WaitingQueueSize:    3,
		KVCacheUsagePercent: 0.1,
		MaxActiveModels:     2,
		ActiveModels:        map[string]int{"foo": 1, "target-model": 1}, // Has "target-model"
		WaitingModels:       make(map[string]int),
	}
	pod3MetricsNaked := &backendmetrics.Metrics{
		WaitingQueueSize:    10,
		KVCacheUsagePercent: 0.2,
		MaxActiveModels:     2,
		ActiveModels:        map[string]int{"foo": 1},
		WaitingModels:       make(map[string]int),
	}
	pod4HighKVMetricsNaked := &backendmetrics.Metrics{
		WaitingQueueSize:    10,
		KVCacheUsagePercent: 0.9, // High KV
		MaxActiveModels:     2,
		ActiveModels:        map[string]int{"foo": 1, "bar": 1},
		WaitingModels:       make(map[string]int),
	}
	pod5HighKVMetricsNaked := &backendmetrics.Metrics{
		WaitingQueueSize:    3,    // Best queue among these high KV pods
		KVCacheUsagePercent: 0.85, // High KV
		MaxActiveModels:     2,
		ActiveModels:        map[string]int{"foo": 1, "target-model": 1}, // Has "target-model"
		WaitingModels:       make(map[string]int),
	}
	pod6HighKVMetricsNaked := &backendmetrics.Metrics{
		WaitingQueueSize:    10,
		KVCacheUsagePercent: 0.85, // High KV
		MaxActiveModels:     2,
		ActiveModels:        map[string]int{"foo": 1},
		WaitingModels:       make(map[string]int),
	}

	tests := []struct {
		name    string
		req     *types.LLMRequest
		input   []*backendmetrics.FakePodMetrics
		wantRes *types.Result
		wantErr bool
	}{
		{
			name: "no pods in datastore",
			req: &types.LLMRequest{
				Model:               "any-model",
				ResolvedTargetModel: "any-model",
				Critical:            true,
			},
			input:   []*backendmetrics.FakePodMetrics{},
			wantRes: nil,
			wantErr: true,
		},
		{
			name: "request for target-model",
			req: &types.LLMRequest{
				Model:               "target-model",
				ResolvedTargetModel: "target-model",
				Critical:            true,
			},
			// lowLatencyFilter applied to [pod1, pod2, pod3]:
			// 1. LowQueueFilter (threshold e.g. 128): all pass -> [pod1, pod2, pod3]
			// 2. LoRAAffinityFilter (for "target-model"): pod2 has affinity ->
			//    [pod2] (assuming strong preference)
			// 3. LeastQueueFilter on [pod2]: -> [pod2]
			// 4. LeastKVCacheFilter on [pod2]: -> [pod2]
			// Expected: pod2
			input: []*backendmetrics.FakePodMetrics{
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, Metrics: pod1MetricsNaked.Clone()},
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, Metrics: pod2MetricsNaked.Clone()},
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}, Metrics: pod3MetricsNaked.Clone()},
			},
			wantRes: &types.Result{
				TargetPod: &types.ScoredPod{Pod: &types.PodMetrics{
					Pod:     &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}, Labels: make(map[string]string)},
					Metrics: pod2MetricsNaked.Clone(),
				}},
			},
			wantErr: false,
		},
		{
			name: "request for other-model",
			req: &types.LLMRequest{
				Model:               "other-model",
				ResolvedTargetModel: "other-model", // No pod has this model active
				Critical:            false,
			},
			// Input pods:
			// pod1: Q=0, KV=0.2, ActiveModels=2/2 (no capacity for "other-model")
			// pod2: Q=3, KV=0.1, ActiveModels=2/2 (no capacity for "other-model")
			// pod3: Q=10, KV=0.2, ActiveModels=1/2 (has capacity for "other-model")
			// lowLatencyFilter applied to [pod1, pod2, pod3]:
			// 1. LowQueueFilter (threshold 128): all pass -> [pod1, pod2, pod3]
			// 2. LoRAAffinityFilter (for "other-model"): none have affinity.
			//    filteredAvailable: pod1 (no), pod2 (no), pod3 (yes). Returns [pod3].
			// 3. LeastQueueFilter on [pod3]: -> [pod3]
			// 4. LeastKVCacheFilter on [pod3]: -> [pod3]
			// Expected: pod3
			input: []*backendmetrics.FakePodMetrics{
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, Metrics: pod1MetricsNaked.Clone()},
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, Metrics: pod2MetricsNaked.Clone()},
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}, Metrics: pod3MetricsNaked.Clone()},
			},
			wantRes: &types.Result{
				TargetPod: &types.ScoredPod{Pod: &types.PodMetrics{
					Pod:     &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}, Labels: make(map[string]string)},
					Metrics: pod3MetricsNaked.Clone(),
				}},
			},
			wantErr: false,
		},
		{
			name: "request when all pods have high KV cache",
			req: &types.LLMRequest{
				Model:               "target-model",
				ResolvedTargetModel: "target-model",
				Critical:            false,
			},
			// lowLatencyFilter applied to [pod4, pod5, pod6]:
			// 1. LowQueueFilter (threshold 128): all pass -> [pod4, pod5, pod6]
			// 2. LoRAAffinityFilter (for "target-model"): pod5 has affinity ->
			//    [pod5] (assuming strong preference)
			// 3. LeastQueueFilter on [pod5]: -> [pod5]
			// 4. LeastKVCacheFilter on [pod5]: -> [pod5]
			// Expected: pod5
			input: []*backendmetrics.FakePodMetrics{
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod4"}}, Metrics: pod4HighKVMetricsNaked.Clone()},
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod5"}}, Metrics: pod5HighKVMetricsNaked.Clone()},
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod6"}}, Metrics: pod6HighKVMetricsNaked.Clone()},
			},
			wantRes: &types.Result{
				TargetPod: &types.ScoredPod{Pod: &types.PodMetrics{
					Pod:     &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod5"}, Labels: make(map[string]string)},
					Metrics: pod5HighKVMetricsNaked.Clone(),
				}},
			},
			wantErr: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			scheduler := NewScheduler(&fakeDataStore{pods: test.input})
			got, err := scheduler.Schedule(context.Background(), test.req)
			if test.wantErr != (err != nil) {
				t.Errorf("Unexpected error state: got err '%v', wantErr %t", err, test.wantErr)
			}
			if !test.wantErr {
				if got == nil {
					t.Fatalf("Expected non-nil result when no error was wanted, but got nil. wantRes: %v", test.wantRes)
				}
				if test.wantRes == nil {
					t.Fatalf("test.wantRes is nil, but no error was expected (wantErr is false). This might be a test setup issue. got: %v", got)
				}
				if diff := cmp.Diff(test.wantRes.TargetPod, got.TargetPod); diff != "" {
					t.Errorf("Unexpected TargetPod (-want +got):\n%s", diff)
				}
			}
		})
	}
}

func TestSchedulePlugins(t *testing.T) {
	tp1 := &TestPlugin{
		NameRes:   "test1",
		ScoreRes:  0.3,
		FilterRes: []k8stypes.NamespacedName{{Name: "pod1"}, {Name: "pod2"}, {Name: "pod3"}},
	}
	tp2 := &TestPlugin{
		NameRes:   "test2",
		ScoreRes:  0.8,
		FilterRes: []k8stypes.NamespacedName{{Name: "pod1"}, {Name: "pod2"}},
	}
	tpFilterAll := &TestPlugin{
		NameRes:   "filter all",
		FilterRes: []k8stypes.NamespacedName{},
	}
	pickerPlugin := &TestPlugin{
		NameRes: "picker",
		PickRes: k8stypes.NamespacedName{Name: "pod1"},
	}

	tests := []struct {
		name               string
		config             SchedulerConfig
		input              []*backendmetrics.FakePodMetrics
		wantTargetPodName  k8stypes.NamespacedName // Expected name of the picked pod
		expectedScoreSum   float64                 // Expected sum of scores for the picked pod if scorers run
		numPodsAfterFilter int                     // Number of pods expected after all filters run
		wantErr            bool
	}{
		{
			name: "all plugins executed successfully, all scorers with same weight",
			config: SchedulerConfig{
				preSchedulePlugins: []plugins.PreSchedule{tp1, tp2},
				filters:            []plugins.Filter{tp1, tp2}, // tp1 -> [p1,p2,p3], tp2 -> [p1,p2]
				scorers: map[plugins.Scorer]int{
					tp1: 1, // ScoreRes 0.3
					tp2: 1, // ScoreRes 0.8
				},
				picker:              pickerPlugin, // Will pick "pod1"
				postSchedulePlugins: []plugins.PostSchedule{tp1, tp2},
			},
			input: []*backendmetrics.FakePodMetrics{ // Initial pods available
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}},
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}},
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}},
			},
			wantTargetPodName:  k8stypes.NamespacedName{Name: "pod1"},
			expectedScoreSum:   0.3*1 + 0.8*1, // (tp1.ScoreRes * weight1) + (tp2.ScoreRes * weight2)
			numPodsAfterFilter: 2,             // After tp1 then tp2 filters
			wantErr:            false,
		},
		{
			name: "all plugins executed successfully, different scorers weights",
			config: SchedulerConfig{
				preSchedulePlugins: []plugins.PreSchedule{tp1, tp2},
				filters:            []plugins.Filter{tp1, tp2}, // tp1 -> [p1,p2,p3], tp2 -> [p1,p2]
				scorers: map[plugins.Scorer]int{
					tp1: 60, // ScoreRes 0.3
					tp2: 40, // ScoreRes 0.8
				},
				picker:              pickerPlugin, // Will pick "pod1"
				postSchedulePlugins: []plugins.PostSchedule{tp1, tp2},
			},
			input: []*backendmetrics.FakePodMetrics{
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}},
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}},
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}},
			},
			wantTargetPodName:  k8stypes.NamespacedName{Name: "pod1"},
			expectedScoreSum:   (0.3 * 60) + (0.8 * 40), // (0.3*60=18) + (0.8*40=32) = 50
			numPodsAfterFilter: 2,
			wantErr:            false,
		},
		{
			name: "filter all results in error",
			config: SchedulerConfig{
				preSchedulePlugins: []plugins.PreSchedule{tp1, tp2},
				filters:            []plugins.Filter{tp1, tpFilterAll}, // tp1 -> [p1,p2,p3], tpFilterAll -> []
				scorers: map[plugins.Scorer]int{ // Scorers won't run if filters return no pods
					tp1: 1,
					tp2: 1,
				},
				picker:              pickerPlugin,                     // Picker won't run effectively
				postSchedulePlugins: []plugins.PostSchedule{tp1, tp2}, // PostSchedule won't run
			},
			input: []*backendmetrics.FakePodMetrics{
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}},
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}},
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}},
			},
			numPodsAfterFilter: 0, // No pods left after tpFilterAll
			wantErr:            true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Reset all plugins before each new test case.
			for _, plugin := range test.config.preSchedulePlugins {
				plugin.(*TestPlugin).reset()
			}
			for _, plugin := range test.config.filters {
				plugin.(*TestPlugin).reset()
			}
			for plugin := range test.config.scorers {
				plugin.(*TestPlugin).reset()
			}
			test.config.picker.(*TestPlugin).reset()
			for _, plugin := range test.config.postSchedulePlugins {
				plugin.(*TestPlugin).reset()
			}

			scheduler := NewSchedulerWithConfig(&fakeDataStore{pods: test.input}, &test.config)

			req := &types.LLMRequest{Model: "test-model", ResolvedTargetModel: "test-model"}
			got, err := scheduler.Schedule(context.Background(), req)

			if test.wantErr != (err != nil) {
				t.Fatalf("Unexpected error state: got err '%v', wantErr %t", err, test.wantErr)
			}

			if err != nil {
				return
			}

			if got == nil || got.TargetPod == nil {
				t.Fatalf("Schedule() returned nil result or nil TargetPod when no error was expected")
			}
			if got.TargetPod.GetPod().NamespacedName != test.wantTargetPodName {
				t.Errorf("Unexpected TargetPod Name: got %s, want %s", got.TargetPod.GetPod().NamespacedName, test.wantTargetPodName)
			}

			// Validate plugin execution counts
			for _, plugin := range test.config.preSchedulePlugins {
				tp := plugin.(*TestPlugin)
				if tp.PreScheduleCallCount != 1 {
					t.Errorf("Plugin %s PreSchedule() called %d times, expected 1", plugin.Name(), tp.PreScheduleCallCount)
				}
			}

			for _, plugin := range test.config.filters {
				tp, _ := plugin.(*TestPlugin)
				if tp.FilterCallCount != 1 {
					t.Errorf("Plugin %s Filter() called %d times, expected 1", plugin.Name(), tp.FilterCallCount)
				}
			}

			for plugin := range test.config.scorers {
				tp, _ := plugin.(*TestPlugin)
				if tp.ScoreCallCount != 1 {
					t.Errorf("Plugin %s Score() called %d times, expected 1", plugin.Name(), tp.ScoreCallCount)
				}
				if test.numPodsAfterFilter != tp.NumOfScoredPods {
					t.Errorf("Plugin %s Score() called with %d pods, expected %d", plugin.Name(), tp.NumOfScoredPods, test.numPodsAfterFilter)
				}
			}

			tp, _ := test.config.picker.(*TestPlugin)
			if tp.NumOfPickerCandidates != test.numPodsAfterFilter {
				t.Errorf("Picker plugin %s Pick() called with %d candidates, expected %d", tp.Name(), tp.NumOfPickerCandidates, tp.NumOfScoredPods)
			}
			if tp.PickCallCount != 1 {
				t.Errorf("Picker plugin %s Pick() called %d times, expected 1", tp.Name(), tp.PickCallCount)
			}
			if tp.WinnerPodScore != test.expectedScoreSum {
				t.Errorf("winnder pod score %v, expected %v", tp.WinnerPodScore, test.expectedScoreSum)
			}

			for _, plugin := range test.config.postSchedulePlugins {
				tp, _ := plugin.(*TestPlugin)
				if tp.PostScheduleCallCount != 1 {
					t.Errorf("Plugin %s PostSchedule() called %d times, expected 1", plugin.Name(), tp.PostScheduleCallCount)
				}
			}
		})
	}
}

// TestPlugin is an implementation of all plugin interfaces, useful in unit
// tests.
type TestPlugin struct {
	NameRes               string
	ScoreCallCount        int
	NumOfScoredPods       int     // Number of pods passed to Score method
	ScoreRes              float64 // Score to assign to each pod
	FilterCallCount       int
	FilterRes             []k8stypes.NamespacedName // Pod names to return from Filter
	PreScheduleCallCount  int
	PostScheduleCallCount int
	PickCallCount         int
	NumOfPickerCandidates int                     // Number of candidates passed to Pick method
	PickRes               k8stypes.NamespacedName // Pod name to pick
	WinnerPodScore        float64                 // Actual final score of the pod picked by this TestPlugin (if acting as picker)
}

func (tp *TestPlugin) Name() string { return tp.NameRes }

func (tp *TestPlugin) PreSchedule(ctx *types.SchedulingContext) {
	tp.PreScheduleCallCount++
}

// Filter method for TestPlugin. It filters the input 'pods' list based on
// tp.FilterRes.
func (tp *TestPlugin) Filter(ctx *types.SchedulingContext, pods []types.Pod) []types.Pod {
	tp.FilterCallCount++
	if len(tp.FilterRes) == 0 {
		return []types.Pod{}
	}

	res := []types.Pod{}
	for _, pod := range pods {
		for _, nameToKeep := range tp.FilterRes {
			if pod.GetPod().NamespacedName == nameToKeep {
				res = append(res, pod)
				break // Found a match for this pod, move to the next pod in 'pods'
			}
		}
	}
	return res
}

func (tp *TestPlugin) Score(ctx *types.SchedulingContext, pods []types.Pod) map[types.Pod]float64 {
	tp.ScoreCallCount++
	tp.NumOfScoredPods = len(pods)
	scoredPods := make(map[types.Pod]float64, len(pods))
	for _, pod := range pods {
		scoredPods[pod] = tp.ScoreRes
	}
	return scoredPods
}

func (tp *TestPlugin) Pick(ctx *types.SchedulingContext, scoredPods []*types.ScoredPod) *types.Result {
	tp.PickCallCount++
	tp.NumOfPickerCandidates = len(scoredPods)
	if len(scoredPods) == 0 {
		return nil
	}

	for _, sp := range scoredPods {
		if sp.GetPod().NamespacedName == tp.PickRes {
			tp.WinnerPodScore = sp.Score
			return &types.Result{TargetPod: sp.Pod}
		}
	}
	// Fallback if PickRes not found (should ideally not happen in well-formed
	// tests for this plugin).
	// To make test plugin more robust for varied test cases, pick the first if
	// specific not found.
	if len(scoredPods) > 0 {
		tp.WinnerPodScore = scoredPods[0].Score
		return &types.Result{TargetPod: scoredPods[0].Pod}
	}
	return nil
}

func (tp *TestPlugin) PostSchedule(ctx *types.SchedulingContext, res *types.Result) {
	tp.PostScheduleCallCount++
}

func (tp *TestPlugin) reset() {
	tp.PreScheduleCallCount = 0
	tp.FilterCallCount = 0
	tp.ScoreCallCount = 0
	tp.NumOfScoredPods = 0
	tp.PostScheduleCallCount = 0
	tp.PickCallCount = 0
	tp.NumOfPickerCandidates = 0
	tp.WinnerPodScore = 0.0
}
