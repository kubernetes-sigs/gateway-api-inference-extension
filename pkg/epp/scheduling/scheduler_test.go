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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/multi/prefix"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/picker"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/profile"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/scorer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

// Tests the default scheduler configuration and expected behavior.
func TestSchedule(t *testing.T) {
	kvCacheUtilizationScorer := scorer.NewKVCacheUtilizationScorer()
	queueingScorer := scorer.NewQueueScorer()
	prefixCacheScorer := prefix.New(context.Background(), prefix.DefaultConfig)
	loraAffinityScorer := scorer.NewLoraAffinityScorer()

	defaultProfile := framework.NewSchedulerProfile().
		WithScorers(framework.NewWeightedScorer(kvCacheUtilizationScorer, 1),
			framework.NewWeightedScorer(queueingScorer, 1),
			framework.NewWeightedScorer(prefixCacheScorer, 1),
			framework.NewWeightedScorer(loraAffinityScorer, 1),
		).
		WithPicker(picker.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))

	profileHandler := profile.NewSingleProfileHandler()

	schedulerConfig := NewSchedulerConfig(profileHandler, map[string]*framework.SchedulerProfile{"default": defaultProfile})

	podMetrics1 := &types.PodMetrics{
		Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}},
		MetricsState: &backendmetrics.MetricsState{
			WaitingQueueSize:    0,
			KVCacheUsagePercent: 0.2,
			MaxActiveModels:     2,
			ActiveModels: map[string]int{
				"foo": 1,
				"bar": 1,
			},
		},
	}
	podMetrics2 := &types.PodMetrics{
		Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}},
		MetricsState: &backendmetrics.MetricsState{
			WaitingQueueSize:    0,
			KVCacheUsagePercent: 0.2,
			MaxActiveModels:     2,
			ActiveModels: map[string]int{
				"foo":      1,
				"critical": 1,
			},
		},
	}
	podMetrics3 := &types.PodMetrics{
		Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}},
		MetricsState: &backendmetrics.MetricsState{
			WaitingQueueSize:    10,
			KVCacheUsagePercent: 0.8,
			MaxActiveModels:     2,
			ActiveModels: map[string]int{
				"foo": 1,
			},
		},
	}

	// Map of raw scores for each pod, keyed by scorer type.

	rawScores := map[string]map[types.Pod]float64{
		"kv-cache-utilization-scorer": {
			podMetrics1: 0.8,
			podMetrics2: 0.8,
			podMetrics3: 0.19999999999999996,
		},
		"lora-affinity-scorer": {
			podMetrics1: 0,
			podMetrics2: 1.0,
			podMetrics3: 0.8,
		},
		"prefix-cache-scorer": {
			podMetrics1: 0,
			podMetrics2: 0,
			podMetrics3: 0,
		},
		"queue-scorer": {
			podMetrics1: 1.0,
			podMetrics2: 1.0,
			podMetrics3: 0,
		},
	}
	tests := []struct {
		name    string
		req     *types.LLMRequest
		input   []types.Pod
		wantRes *types.SchedulingResult
		err     bool
	}{
		{
			name: "no candidate pods",
			req: &types.LLMRequest{
				RequestId:   uuid.NewString(),
				TargetModel: "any-model",
			},
			input:   []types.Pod{},
			wantRes: nil,
			err:     true,
		},
		{
			name: "finds optimal pod",
			req: &types.LLMRequest{
				RequestId:   uuid.NewString(),
				TargetModel: "critical",
			},
			// pod2 will be picked because it has relatively low queue size, with the requested
			// model being active, and has low KV cache.
			input: []types.Pod{
				podMetrics1,
				podMetrics2,
				podMetrics3,
			},
			wantRes: &types.SchedulingResult{
				ProfileResults: map[string]*types.ProfileRunResult{
					"default": {
						TargetPods: []types.Pod{
							&types.ScoredPod{
								Pod:   podMetrics2,
								Score: 2.8,
							},
						},
						RawScores: rawScores,
					},
				},
				AllProfileRunResults: map[string]*types.ProfileRunResult{
					"default": {
						TargetPods: []types.Pod{
							&types.ScoredPod{
								Pod:   podMetrics2,
								Score: 2.8,
							},
						},
						RawScores: rawScores,
					},
				},
				PrimaryProfileName: "default",
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			scheduler := NewSchedulerWithConfig(schedulerConfig)
			got, err := scheduler.Schedule(context.Background(), test.req, test.input)
			if test.err != (err != nil) {
				t.Errorf("Unexpected error, got %v, want %v", err, test.err)
			}

			if diff := cmp.Diff(test.wantRes, got); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}
