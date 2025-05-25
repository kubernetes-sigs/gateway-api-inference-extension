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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	profilepicker "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/profile-picker"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

// Tests the default scheduler configuration and expected behavior.
func TestSchedule(t *testing.T) {
	tests := []struct {
		name    string
		req     *types.LLMRequest
		input   []*backendmetrics.FakePodMetrics
		wantRes map[string]*types.Result
		err     bool
	}{
		{
			name: "no pods in datastore",
			req: &types.LLMRequest{
				TargetModel: "any-model",
				RequestId:   uuid.NewString(),
				Critical:    true,
			},
			input:   []*backendmetrics.FakePodMetrics{},
			wantRes: nil,
			err:     true,
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
					Metrics: &backendmetrics.MetricsState{
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
					Metrics: &backendmetrics.MetricsState{
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
					Metrics: &backendmetrics.MetricsState{
						WaitingQueueSize:    10,
						KVCacheUsagePercent: 0.2,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo": 1,
						},
					},
				},
			},
			wantRes: map[string]*types.Result{
				"default": {
					TargetPod: &types.ScoredPod{
						Pod: &types.PodMetrics{
							Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}, Labels: make(map[string]string)},
							MetricsState: &backendmetrics.MetricsState{
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
					Metrics: &backendmetrics.MetricsState{
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
					Metrics: &backendmetrics.MetricsState{
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
					Metrics: &backendmetrics.MetricsState{
						WaitingQueueSize:    10,
						KVCacheUsagePercent: 0.2,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo": 1,
						},
					},
				},
			},
			wantRes: map[string]*types.Result{
				"default": {
					TargetPod: &types.ScoredPod{
						Pod: &types.PodMetrics{
							Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}, Labels: make(map[string]string)},
							MetricsState: &backendmetrics.MetricsState{
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
					Metrics: &backendmetrics.MetricsState{
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
					Metrics: &backendmetrics.MetricsState{
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
					Metrics: &backendmetrics.MetricsState{
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

func TestPostResponse(t *testing.T) {
	pr1 := &testPostResponse{
		NameRes:                 "pr1",
		ExtraHeaders:            map[string]string{"x-session-id": "qwer-asdf-zxcv"},
		ReceivedResponseHeaders: make(map[string]string),
	}

	targetPod := k8stypes.NamespacedName{Name: "pod2"}

	tests := []struct {
		name               string
		config             *framework.SchedulerProfile
		input              []*backendmetrics.FakePodMetrics
		responseHeaders    map[string]string
		wantUpdatedHeaders map[string]string
	}{
		{
			name: "Simple postResponse test",
			config: framework.NewSchedulerProfile("default").
				WithPostResponsePlugins(pr1),
			input: []*backendmetrics.FakePodMetrics{
				{Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}},
				{Pod: &backend.Pod{NamespacedName: targetPod}},
			},
			responseHeaders:    map[string]string{"Content-type": "application/json", "Content-Length": "1234"},
			wantUpdatedHeaders: map[string]string{"x-session-id": "qwer-asdf-zxcv", "Content-type": "application/json", "Content-Length": "1234"},
		},
	}

	for _, test := range tests {
		schedulerConfig := NewSchedulerConfig(profilepicker.NewAllProfilesPicker(), []*framework.SchedulerProfile{test.config})
		scheduler := NewSchedulerWithConfig(&fakeDataStore{pods: test.input}, schedulerConfig)

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
