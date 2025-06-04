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
			},
			input:   []*backendmetrics.FakePodMetrics{},
			wantRes: nil,
			err:     true,
		},
		{
			name: "finds optimal pod",
			req: &types.LLMRequest{
				TargetModel: "critical",
				RequestId:   uuid.NewString(),
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
