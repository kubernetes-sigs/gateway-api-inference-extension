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
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics" // Import config for thresholds
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

// Tests the default scheduler configuration and expected behavior.
func TestSchedule(t *testing.T) {
	tests := []struct {
		name    string
		req     *types.LLMRequest
		input   []*backendmetrics.FakePodMetrics
		wantRes *types.Result
		err     bool
	}{
		{
			name: "no pods in datastore",
			req: &types.LLMRequest{
				Model:               "any-model",
				ResolvedTargetModel: "any-model",
				Critical:            true,
			},
			input: []*backendmetrics.FakePodMetrics{},
			err:   true,
		},
		{
			name: "critical request",
			req: &types.LLMRequest{
				Model:               "critical",
				ResolvedTargetModel: "critical",
				Critical:            true,
			},
			// pod2 will be picked because it has relatively low queue size, with the requested
			// model being active, and has low KV cache.
			input: []*backendmetrics.FakePodMetrics{
				{
					Pod: &backendmetrics.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}},
					Metrics: &backendmetrics.Metrics{
						WaitingQueueSize:    0,
						KVCacheUsagePercent: 0.2,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo": 1,
							"bar": 1,
						},
					},
				},
				{
					Pod: &backendmetrics.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}},
					Metrics: &backendmetrics.Metrics{
						WaitingQueueSize:    3,
						KVCacheUsagePercent: 0.1,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo":      1,
							"critical": 1,
						},
					},
				},
				{
					Pod: &backendmetrics.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}},
					Metrics: &backendmetrics.Metrics{
						WaitingQueueSize:    10,
						KVCacheUsagePercent: 0.2,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo": 1,
						},
					},
				},
			},
			wantRes: &types.Result{
				TargetPod: &types.PodMetrics{
					Pod: &backendmetrics.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}},
					Metrics: &backendmetrics.Metrics{
						WaitingQueueSize:    3,
						KVCacheUsagePercent: 0.1,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo":      1,
							"critical": 1,
						},
						WaitingModels: map[string]int{},
					},
				},
			},
		},
		{
			name: "sheddable request, accepted",
			req: &types.LLMRequest{
				Model:               "sheddable",
				ResolvedTargetModel: "sheddable",
				Critical:            false,
			},
			// pod1 will be picked because it has capacity for the sheddable request.
			input: []*backendmetrics.FakePodMetrics{
				{
					Pod: &backendmetrics.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}},
					Metrics: &backendmetrics.Metrics{
						WaitingQueueSize:    0,
						KVCacheUsagePercent: 0.2,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo": 1,
							"bar": 1,
						},
					},
				},
				{
					Pod: &backendmetrics.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}},
					Metrics: &backendmetrics.Metrics{
						WaitingQueueSize:    3,
						KVCacheUsagePercent: 0.1,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo":      1,
							"critical": 1,
						},
					},
				},
				{
					Pod: &backendmetrics.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}},
					Metrics: &backendmetrics.Metrics{
						WaitingQueueSize:    10,
						KVCacheUsagePercent: 0.2,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo": 1,
						},
					},
				},
			},
			wantRes: &types.Result{
				TargetPod: &types.PodMetrics{
					Pod: &backendmetrics.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}},
					Metrics: &backendmetrics.Metrics{
						WaitingQueueSize:    0,
						KVCacheUsagePercent: 0.2,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo": 1,
							"bar": 1,
						},
						WaitingModels: map[string]int{},
					},
				},
			},
		},
		{
			name: "sheddable request, dropped",
			req: &types.LLMRequest{
				Model:               "sheddable",
				ResolvedTargetModel: "sheddable",
				Critical:            false,
			},
			// All pods have higher KV cache thant the threshold, so the sheddable request will be
			// dropped.
			input: []*backendmetrics.FakePodMetrics{
				{
					Pod: &backendmetrics.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}},
					Metrics: &backendmetrics.Metrics{
						WaitingQueueSize:    10,
						KVCacheUsagePercent: 0.9,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo": 1,
							"bar": 1,
						},
					},
				},
				{
					Pod: &backendmetrics.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}},
					Metrics: &backendmetrics.Metrics{
						WaitingQueueSize:    3,
						KVCacheUsagePercent: 0.85,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo":      1,
							"critical": 1,
						},
					},
				},
				{
					Pod: &backendmetrics.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}},
					Metrics: &backendmetrics.Metrics{
						WaitingQueueSize:    10,
						KVCacheUsagePercent: 0.85,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo": 1,
						},
					},
				},
			},
			wantRes: nil,
			err:     true,
		},
	}

	schedConfig := &SchedulerConfig{
		preSchedulePlugins:  []plugins.PreSchedule{},
		scorers:             []plugins.Scorer{},
		filters:             []plugins.Filter{defPlugin},
		postSchedulePlugins: []plugins.PostSchedule{},
		picker:              defPlugin,
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			scheduler := NewSchedulerWithConfig(&fakeDataStore{pods: test.input}, schedConfig)
			got, err := scheduler.Schedule(context.Background(), test.req)
			if test.err != (err != nil) {
				t.Errorf("Unexpected error, got %v, want %v", err, test.err)
			}

			opt := cmp.AllowUnexported(types.PodMetrics{})
			if diff := cmp.Diff(test.wantRes, got, opt); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
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
	tp_filterAll := &TestPlugin{
		NameRes:   "filter all",
		FilterRes: []k8stypes.NamespacedName{},
	}
	pickerPlugin := &TestPlugin{
		NameRes: "picker",
		PickRes: k8stypes.NamespacedName{Name: "pod1"},
	}

	tests := []struct {
		name           string
		config         SchedulerConfig
		input          []*backendmetrics.FakePodMetrics
		wantTargetPod  k8stypes.NamespacedName
		targetPodScore float64
		// Number of expected pods to score (after filter)
		numPodsToScore int
		err            bool
	}{
		{
			name: "all plugins executed successfully",
			config: SchedulerConfig{
				preSchedulePlugins:  []plugins.PreSchedule{tp1, tp2},
				filters:             []plugins.Filter{tp1, tp2},
				scorers:             []plugins.Scorer{tp1, tp2},
				postSchedulePlugins: []plugins.PostSchedule{tp1, tp2},
				picker:              pickerPlugin,
			},
			input: []*backendmetrics.FakePodMetrics{
				{Pod: &backendmetrics.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}},
				{Pod: &backendmetrics.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}},
				{Pod: &backendmetrics.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}},
			},
			wantTargetPod:  k8stypes.NamespacedName{Name: "pod1"},
			targetPodScore: 1.1,
			numPodsToScore: 2,
			err:            false,
		},
		{
			name: "filter all",
			config: SchedulerConfig{
				preSchedulePlugins:  []plugins.PreSchedule{tp1, tp2},
				filters:             []plugins.Filter{tp1, tp_filterAll},
				scorers:             []plugins.Scorer{tp1, tp2},
				postSchedulePlugins: []plugins.PostSchedule{tp1, tp2},
				picker:              pickerPlugin,
			},
			input: []*backendmetrics.FakePodMetrics{
				{Pod: &backendmetrics.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}},
				{Pod: &backendmetrics.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}},
				{Pod: &backendmetrics.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}},
			},
			numPodsToScore: 0,
			err:            true, // no available pods to server after filter all
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Reset all plugins before each new test case.
			for _, plugin := range test.config.preSchedulePlugins {
				plugin.(*TestPlugin).reset()
			}
			for _, plugin := range test.config.postSchedulePlugins {
				plugin.(*TestPlugin).reset()
			}
			for _, plugin := range test.config.filters {
				plugin.(*TestPlugin).reset()
			}
			for _, plugin := range test.config.scorers {
				plugin.(*TestPlugin).reset()
			}
			test.config.picker.(*TestPlugin).reset()

			// Initialize the scheduler
			scheduler := NewSchedulerWithConfig(&fakeDataStore{pods: test.input}, &test.config)

			req := &types.LLMRequest{Model: "test-model"}
			got, err := scheduler.Schedule(context.Background(), req)

			// Validate error state
			if test.err != (err != nil) {
				t.Fatalf("Unexpected error, got %v, want %v", err, test.err)
			}

			if err != nil {
				return
			}

			// Validate output
			opt := cmp.AllowUnexported(types.PodMetrics{})
			wantPod := &types.PodMetrics{
				Pod: &backendmetrics.Pod{NamespacedName: test.wantTargetPod},
			}
			wantPod.SetScore(test.targetPodScore)
			wantRes := &types.Result{TargetPod: wantPod}
			if diff := cmp.Diff(wantRes, got, opt); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}

			// Validate plugin execution counts dynamically
			for _, plugin := range test.config.preSchedulePlugins {
				tp, _ := plugin.(*TestPlugin)
				if tp.PreScheduleCallCount != 1 {
					t.Errorf("Plugin %s PreSchedule() called %d times, expected 1", tp.NameRes, tp.PreScheduleCallCount)
				}
			}

			for _, plugin := range test.config.filters {
				tp, _ := plugin.(*TestPlugin)
				if tp.FilterCallCount != 1 {
					t.Errorf("Plugin %s Filter() called %d times, expected 1", tp.NameRes, tp.FilterCallCount)
				}
			}

			for _, plugin := range test.config.scorers {
				tp, _ := plugin.(*TestPlugin)
				if tp.ScoreCallCount != test.numPodsToScore {
					t.Errorf("Plugin %s Score() called %d times, expected 1", tp.NameRes, tp.ScoreCallCount)
				}
			}

			for _, plugin := range test.config.postSchedulePlugins {
				tp, _ := plugin.(*TestPlugin)
				if tp.PostScheduleCallCount != 1 {
					t.Errorf("Plugin %s PostSchedule() called %d times, expected 1", tp.NameRes, tp.PostScheduleCallCount)
				}
			}

			tp, _ := test.config.picker.(*TestPlugin)
			if tp.PickCallCount != 1 {
				t.Errorf("Picker plugin %s Pick() called %d times, expected 1", tp.NameRes, tp.PickCallCount)
			}

		})
	}
}

type fakeDataStore struct {
	pods []*backendmetrics.FakePodMetrics
}

func (fds *fakeDataStore) PodGetAll() []backendmetrics.PodMetrics {
	pm := make([]backendmetrics.PodMetrics, 0, len(fds.pods))
	for _, pod := range fds.pods {
		pm = append(pm, pod)
	}
	return pm
}

// TestPlugin is an implementation useful in unit tests.
type TestPlugin struct {
	NameRes               string
	ScoreCallCount        int
	ScoreRes              float64
	FilterCallCount       int
	FilterRes             []k8stypes.NamespacedName
	PreScheduleCallCount  int
	PostScheduleCallCount int
	PickCallCount         int
	PickRes               k8stypes.NamespacedName
}

func (tp *TestPlugin) Name() string { return tp.NameRes }

func (tp *TestPlugin) PreSchedule(ctx *types.SchedulingContext) {
	tp.PreScheduleCallCount++
}

func (tp *TestPlugin) Filter(ctx *types.SchedulingContext, pods []types.Pod) []types.Pod {
	tp.FilterCallCount++
	return findPods(ctx, tp.FilterRes...)
}

func (tp *TestPlugin) Score(ctx *types.SchedulingContext, pod types.Pod) float64 {
	tp.ScoreCallCount++
	return tp.ScoreRes
}

func (tp *TestPlugin) PostSchedule(ctx *types.SchedulingContext, res *types.Result) {
	tp.PostScheduleCallCount++
}

func (tp *TestPlugin) Pick(ctx *types.SchedulingContext, pods []types.Pod) *types.Result {
	tp.PickCallCount++
	pod := findPods(ctx, tp.PickRes)[0]
	return &types.Result{TargetPod: pod}
}

func (tp *TestPlugin) reset() {
	tp.PreScheduleCallCount = 0
	tp.FilterCallCount = 0
	tp.ScoreCallCount = 0
	tp.PostScheduleCallCount = 0
	tp.PickCallCount = 0
}

func findPods(ctx *types.SchedulingContext, names ...k8stypes.NamespacedName) []types.Pod {
	res := []types.Pod{}
	for _, pod := range ctx.PodsSnapshot {
		for _, name := range names {
			if pod.GetPod().NamespacedName.String() == name.String() {
				res = append(res, pod)
			}
		}
	}
	return res
}
