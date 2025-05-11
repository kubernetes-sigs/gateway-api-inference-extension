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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
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
				TargetModel: "any-model",
				RequestId:   uuid.NewString(),
				Critical:    true,
			},
			input: []*backendmetrics.FakePodMetrics{},
			err:   true,
		},
		{
			name: "critical request",
			req: &types.LLMRequest{
				TargetModel: "critical",
				RequestId:   uuid.NewString(),
				Critical:    true,
			},
			// pod2 will be picked because it has relatively low queue size, with the requested
			// model being active, and has low KV cache.
			input: []*backendmetrics.FakePodMetrics{
				{
					Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}},
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
					Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}},
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
					Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}},
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
				TargetPod: &types.ScoredPod{
					Pod: &types.PodMetrics{
						Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}, Labels: make(map[string]string)},
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
		},
		{
			name: "sheddable request, accepted",
			req: &types.LLMRequest{
				TargetModel: "sheddable",
				RequestId:   uuid.NewString(),
				Critical:    false,
			},
			// pod1 will be picked because it has capacity for the sheddable request.
			input: []*backendmetrics.FakePodMetrics{
				{
					Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}},
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
					Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}},
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
					Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}},
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
				TargetPod: &types.ScoredPod{
					Pod: &types.PodMetrics{
						Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}, Labels: make(map[string]string)},
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
		},
		{
			name: "sheddable request, dropped",
			req: &types.LLMRequest{
				TargetModel: "sheddable",
				RequestId:   uuid.NewString(),
				Critical:    false,
			},
			// All pods have higher KV cache thant the threshold, so the sheddable request will be
			// dropped.
			input: []*backendmetrics.FakePodMetrics{
				{
					Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}},
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
					Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}},
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
					Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}},
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

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			scheduler := NewScheduler(&fakeDataStore{pods: test.input})
			got, err := scheduler.Schedule(context.Background(), test.req)
			if test.err != (err != nil) {
				t.Errorf("Unexpected error, got %v, want %v", err, test.err)
			}

			if diff := cmp.Diff(test.wantRes, got); diff != "" {
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
			name: "all plugins executed successfully, all scorers with same weight",
			config: SchedulerConfig{
				preSchedulePlugins: []plugins.PreSchedule{tp1, tp2},
				filters:            []plugins.Filter{tp1, tp2},
				scorers: map[plugins.Scorer]int{
					tp1: 1,
					tp2: 1,
				},
				picker:              pickerPlugin,
				postSchedulePlugins: []plugins.PostSchedule{tp1, tp2},
			},
			input: []*backendmetrics.FakePodMetrics{
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}},
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}},
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}},
			},
			wantTargetPod:  k8stypes.NamespacedName{Name: "pod1"},
			targetPodScore: 1.1,
			numPodsToScore: 2,
			err:            false,
		},
		{
			name: "all plugins executed successfully, different scorers weights",
			config: SchedulerConfig{
				preSchedulePlugins: []plugins.PreSchedule{tp1, tp2},
				filters:            []plugins.Filter{tp1, tp2},
				scorers: map[plugins.Scorer]int{
					tp1: 60,
					tp2: 40,
				},
				picker:              pickerPlugin,
				postSchedulePlugins: []plugins.PostSchedule{tp1, tp2},
			},
			input: []*backendmetrics.FakePodMetrics{
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}},
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}},
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}},
			},
			wantTargetPod:  k8stypes.NamespacedName{Name: "pod1"},
			targetPodScore: 50,
			numPodsToScore: 2,
			err:            false,
		},
		{
			name: "filter all",
			config: SchedulerConfig{
				preSchedulePlugins: []plugins.PreSchedule{tp1, tp2},
				filters:            []plugins.Filter{tp1, tp_filterAll},
				scorers: map[plugins.Scorer]int{
					tp1: 1,
					tp2: 1,
				},
				picker:              pickerPlugin,
				postSchedulePlugins: []plugins.PostSchedule{tp1, tp2},
			},
			input: []*backendmetrics.FakePodMetrics{
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}},
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}},
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}}},
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

			// Initialize the scheduler
			scheduler := NewSchedulerWithConfig(&fakeDataStore{pods: test.input}, &test.config)

			req := &types.LLMRequest{
				TargetModel: "test-model",
				RequestId:   uuid.NewString(),
			}
			got, err := scheduler.Schedule(context.Background(), req)

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
			wantRes := &types.Result{TargetPod: wantPod}
			if diff := cmp.Diff(wantRes, got); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}

			// Validate plugin execution counts dynamically
			for _, plugin := range test.config.preSchedulePlugins {
				tp, _ := plugin.(*TestPlugin)
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
				if test.numPodsToScore != tp.NumOfScoredPods {
					t.Errorf("Plugin %s Score() called with %d pods, expected %d", plugin.Name(), tp.NumOfScoredPods, test.numPodsToScore)
				}
			}

			tp, _ := test.config.picker.(*TestPlugin)
			if tp.NumOfPickerCandidates != test.numPodsToScore {
				t.Errorf("Picker plugin %s Pick() called with %d candidates, expected %d", tp.Name(), tp.NumOfPickerCandidates, tp.NumOfScoredPods)
			}
			if tp.PickCallCount != 1 {
				t.Errorf("Picker plugin %s Pick() called %d times, expected 1", tp.Name(), tp.PickCallCount)
			}
			if tp.WinnderPodScore != test.targetPodScore {
				t.Errorf("winnder pod score %v, expected %v", tp.WinnderPodScore, test.targetPodScore)
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

func TestPostResponse(t *testing.T) {
	pr1 := &testPostResponse{
		NameRes:                 "pr1",
		ExtraHeaders:            map[string]string{"x-session-id": "qwer-asdf-zxcv"},
		ReceivedResponseHeaders: make(map[string]string),
	}

	targetPod := k8stypes.NamespacedName{Name: "pod2"}

	tests := []struct {
		name               string
		config             SchedulerConfig
		input              []*backendmetrics.FakePodMetrics
		responseHeaders    map[string]string
		wantUpdatedHeaders map[string]string
	}{
		{
			name: "Simple postResponse test",
			config: SchedulerConfig{
				postResponsePlugins: []plugins.PostResponse{pr1},
			},
			input: []*backendmetrics.FakePodMetrics{
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}},
				{Pod: &backend.Pod{NamespacedName: targetPod}},
			},
			responseHeaders:    map[string]string{"Content-type": "application/json", "Content-Length": "1234"},
			wantUpdatedHeaders: map[string]string{"x-session-id": "qwer-asdf-zxcv", "Content-type": "application/json", "Content-Length": "1234"},
		},
	}

	for _, test := range tests {
		scheduler := NewSchedulerWithConfig(&fakeDataStore{pods: test.input}, &test.config)

		headers := map[string]string{}
		for k, v := range test.responseHeaders {
			headers[k] = v
		}
		resp := &types.LLMResponse{
			Headers: headers,
		}

		scheduler.OnResponse(context.Background(), resp, targetPod.String())

		if diff := cmp.Diff(test.responseHeaders, pr1.ReceivedResponseHeaders); diff != "" {
			t.Errorf("Unexpected output (-responseHeaders +ReceivedResponseHeaders): %v", diff)
		}

		if diff := cmp.Diff(test.wantUpdatedHeaders, resp.Headers); diff != "" {
			t.Errorf("Unexpected output (-wantUpdatedHeaders +resp.Headers): %v", diff)
		}
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
	NumOfScoredPods       int
	ScoreRes              float64
	FilterCallCount       int
	FilterRes             []k8stypes.NamespacedName
	PreScheduleCallCount  int
	PostScheduleCallCount int
	PickCallCount         int
	NumOfPickerCandidates int
	PickRes               k8stypes.NamespacedName
	WinnderPodScore       float64
}

func (tp *TestPlugin) Name() string { return tp.NameRes }

func (tp *TestPlugin) PreSchedule(ctx *types.SchedulingContext) {
	tp.PreScheduleCallCount++
}

func (tp *TestPlugin) Filter(ctx *types.SchedulingContext, pods []types.Pod) []types.Pod {
	tp.FilterCallCount++
	return findPods(ctx, tp.FilterRes...)

}

func (tp *TestPlugin) Score(ctx *types.SchedulingContext, pods []types.Pod) map[types.Pod]float64 {
	tp.ScoreCallCount++
	scoredPods := make(map[types.Pod]float64, len(pods))
	for _, pod := range pods {
		scoredPods[pod] += tp.ScoreRes
	}
	tp.NumOfScoredPods = len(scoredPods)
	return scoredPods
}

func (tp *TestPlugin) Pick(ctx *types.SchedulingContext, scoredPods []*types.ScoredPod) *types.Result {
	tp.PickCallCount++
	tp.NumOfPickerCandidates = len(scoredPods)
	pod := findPods(ctx, tp.PickRes)[0]
	tp.WinnderPodScore = getPodScore(scoredPods, pod)
	return &types.Result{TargetPod: pod}
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
}

type testPostResponse struct {
	NameRes                 string
	ReceivedResponseHeaders map[string]string
	ExtraHeaders            map[string]string
}

func (pr *testPostResponse) Name() string { return pr.NameRes }

func (pr *testPostResponse) PostResponse(ctx *types.SchedulingContext, pod types.Pod) {
	for key, value := range ctx.Resp.Headers {
		pr.ReceivedResponseHeaders[key] = value
	}
	for key, value := range pr.ExtraHeaders {
		ctx.Resp.Headers[key] = value
	}
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

func getPodScore(scoredPods []*types.ScoredPod, selectedPod types.Pod) float64 {
	finalScore := 0.0
	for _, scoredPod := range scoredPods {
		if scoredPod.GetPod().NamespacedName.String() == selectedPod.GetPod().NamespacedName.String() {
			finalScore = scoredPod.Score
			break
		}
	}
	return finalScore
}
