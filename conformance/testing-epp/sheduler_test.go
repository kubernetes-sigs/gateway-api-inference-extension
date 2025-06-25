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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics" // Import config for thresholds
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

// Tests the scheduler for conformance tests.
func TestSchedule(t *testing.T) {
	tests := []struct {
		name    string
		input   []backendmetrics.PodMetrics
		req     *types.LLMRequest
		wantRes *types.SchedulingResult
		err     bool
	}{
		{
			name: "no candidate pods and req header is set",
			req: &types.LLMRequest{
				Headers:   map[string]string{"test-epp-endpoint-selection": "random-endpoint"},
				RequestId: uuid.NewString(),
			},
			wantRes: nil,
			err:     true,
		},
		{
			name: "req header not set",
			input: []backendmetrics.PodMetrics{
				&backendmetrics.FakePodMetrics{Pod: &backend.Pod{Address: "random-endpoint"}},
			},
			req: &types.LLMRequest{
				Headers:   map[string]string{}, // Deliberately set an empty header.
				RequestId: uuid.NewString(),
			},
			wantRes: nil,
			err:     true,
		},
		{
			name: "no pods address from the candidate pods matches req header address",
			input: []backendmetrics.PodMetrics{
				&backendmetrics.FakePodMetrics{Pod: &backend.Pod{Address: "nonmatched-endpoint"}},
			},
			req: &types.LLMRequest{
				Headers:   map[string]string{"test-epp-endpoint-selection": "matched-endpoint"},
				RequestId: uuid.NewString(),
			},
			wantRes: nil,
			err:     true,
		},
		{
			name: "one pod address from the candidate pods matches req header address",
			input: []backendmetrics.PodMetrics{
				&backendmetrics.FakePodMetrics{Pod: &backend.Pod{Address: "nonmatched-endpoint"}},
				&backendmetrics.FakePodMetrics{Pod: &backend.Pod{Address: "matched-endpoint"}},
			},
			req: &types.LLMRequest{
				Headers:   map[string]string{"test-epp-endpoint-selection": "matched-endpoint"},
				RequestId: uuid.NewString(),
			},
			wantRes: &types.SchedulingResult{
				ProfileResults: map[string]*types.ProfileRunResult{
					"req-header-based-profile": {
						TargetPod: &types.ScoredPod{
							Pod: &types.PodMetrics{
								Pod: &backend.Pod{
									Address: "matched-endpoint",
									Labels:  map[string]string{},
								},
							},
						},
					},
				},
				PrimaryProfileName: "req-header-based-profile",
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			scheduler := NewReqHeaderBasedScheduler()
			got, err := scheduler.Schedule(context.Background(), test.req, types.ToSchedulerPodMetrics(test.input))
			if test.err != (err != nil) {
				t.Errorf("Unexpected error, got %v, want %v", err, test.err)
			}

			if diff := cmp.Diff(test.wantRes, got); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}
