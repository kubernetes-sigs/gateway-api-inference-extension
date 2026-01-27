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
	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"

	// Import config for thresholds
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	fwksched "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/multi/prefix"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/picker"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/profile"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/scorer"
)

// Tests the default scheduler configuration and expected behavior.
func TestSchedule(t *testing.T) {
	kvCacheUtilizationScorer := scorer.NewKVCacheUtilizationScorer()
	queueingScorer := scorer.NewQueueScorer()
	prefixCacheScorer, err := prefix.New(context.Background(), prefix.DefaultConfig)
	assert.NoError(t, err)
	loraAffinityScorer := scorer.NewLoraAffinityScorer()

	defaultProfile := NewSchedulerProfile().
		WithScorers(NewWeightedScorer(kvCacheUtilizationScorer, 1),
			NewWeightedScorer(queueingScorer, 1),
			NewWeightedScorer(prefixCacheScorer, 1),
			NewWeightedScorer(loraAffinityScorer, 1),
		).
		WithPicker(picker.NewMaxScorePicker(picker.DefaultMaxNumOfEndpoints))

	profileHandler := profile.NewSingleProfileHandler()

	schedulerConfig := NewSchedulerConfig(profileHandler, map[string]framework.SchedulerProfile{"default": defaultProfile})

	tests := []struct {
		name    string
		req     *framework.LLMRequest
		input   []framework.Endpoint
		wantRes *framework.SchedulingResult
		err     bool
	}{
		{
			name: "no candidate endpoints",
			req: &framework.LLMRequest{
				RequestId:   uuid.NewString(),
				TargetModel: "any-model",
			},
			input:   []framework.Endpoint{},
			wantRes: nil,
			err:     true,
		},
		{
			name: "finds optimal endpoint",
			req: &framework.LLMRequest{
				RequestId:   uuid.NewString(),
				TargetModel: "critical",
			},
			// pod2 will be picked because it has relatively low queue size, with the requested
			// model being active, and has low KV cache.
			input: []framework.Endpoint{
				fwksched.NewEndpoint(
					&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}},
					&datalayer.Metrics{
						WaitingQueueSize:    0,
						KVCacheUsagePercent: 0.2,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo": 1,
							"bar": 1,
						},
					}, nil),
				fwksched.NewEndpoint(
					&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}},
					&datalayer.Metrics{
						WaitingQueueSize:    0,
						KVCacheUsagePercent: 0.2,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo":      1,
							"critical": 1,
						},
					}, nil),
				fwksched.NewEndpoint(
					&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod3"}},
					&datalayer.Metrics{
						WaitingQueueSize:    10,
						KVCacheUsagePercent: 0.8,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo": 1,
						},
					}, nil),
			},
			wantRes: &framework.SchedulingResult{
				ProfileResults: map[string]*framework.ProfileRunResult{
					"default": {
						TargetEndpoints: []framework.Endpoint{
							&framework.ScoredEndpoint{
								Endpoint: fwksched.NewEndpoint(
									&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}},
									&datalayer.Metrics{
										WaitingQueueSize:    0,
										KVCacheUsagePercent: 0.2,
										MaxActiveModels:     2,
										ActiveModels: map[string]int{
											"foo":      1,
											"critical": 1,
										},
									}, nil),
								Score: 2.8,
							},
						},
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

			if diff := cmp.Diff(test.wantRes, got, cmp.Comparer(fwksched.ScoredEndpointComparer)); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}
