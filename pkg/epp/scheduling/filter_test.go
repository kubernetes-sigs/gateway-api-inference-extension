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
	"errors"
	"testing"

	"github.com/go-logr/logr"
	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

func TestFilter(t *testing.T) {
	logger := logutil.NewTestLogger()

	tests := []struct {
		name   string
		req    *LLMRequest
		input  []*datastore.PodMetrics
		output []*datastore.PodMetrics
		err    bool
		filter *filter
	}{
		{
			name: "simple filter without successor, failure",
			filter: &filter{filter: func(logger logr.Logger, req *LLMRequest, pods []*datastore.PodMetrics) ([]*datastore.PodMetrics, error) {
				return nil, errors.New("filter error")
			}},
			err: true,
		},
		{
			name:   "default filter, critical request",
			filter: defaultFilter,
			req: &LLMRequest{
				Model:               "critical",
				ResolvedTargetModel: "critical",
				Critical:            true,
			},
			// pod2 will be picked because it has relatively low queue size, with the requested
			// model being active, and has low KV cache.
			input: []*datastore.PodMetrics{
				{
					Pod: datastore.Pod{NamespacedName: types.NamespacedName{Name: "pod1"}},
					Metrics: datastore.Metrics{
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
					Pod: datastore.Pod{NamespacedName: types.NamespacedName{Name: "pod2"}},
					Metrics: datastore.Metrics{
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
					Pod: datastore.Pod{NamespacedName: types.NamespacedName{Name: "pod3"}},
					Metrics: datastore.Metrics{
						WaitingQueueSize:    10,
						KVCacheUsagePercent: 0.2,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo": 1,
						},
					},
				},
			},
			output: []*datastore.PodMetrics{
				{
					Pod: datastore.Pod{NamespacedName: types.NamespacedName{Name: "pod2"}},
					Metrics: datastore.Metrics{
						WaitingQueueSize:    3,
						KVCacheUsagePercent: 0.1,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo":      1,
							"critical": 1,
						},
					},
				},
			},
		},
		{
			name:   "default filter, sheddable request, accepted",
			filter: defaultFilter,
			req: &LLMRequest{
				Model:               "sheddable",
				ResolvedTargetModel: "sheddable",
				Critical:            false,
			},
			// pod1 will be picked because it has capacity for the sheddable request.
			input: []*datastore.PodMetrics{
				{
					Pod: datastore.Pod{NamespacedName: types.NamespacedName{Name: "pod1"}},
					Metrics: datastore.Metrics{
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
					Pod: datastore.Pod{NamespacedName: types.NamespacedName{Name: "pod2"}},
					Metrics: datastore.Metrics{
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
					Pod: datastore.Pod{NamespacedName: types.NamespacedName{Name: "pod3"}},
					Metrics: datastore.Metrics{
						WaitingQueueSize:    10,
						KVCacheUsagePercent: 0.2,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo": 1,
						},
					},
				},
			},
			output: []*datastore.PodMetrics{
				{
					Pod: datastore.Pod{NamespacedName: types.NamespacedName{Name: "pod1"}},
					Metrics: datastore.Metrics{
						WaitingQueueSize:    0,
						KVCacheUsagePercent: 0.2,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo": 1,
							"bar": 1,
						},
					},
				},
			},
		},
		{
			name:   "default filter, sheddable request, dropped",
			filter: defaultFilter,
			req: &LLMRequest{
				Model:               "sheddable",
				ResolvedTargetModel: "sheddable",
				Critical:            false,
			},
			// All pods have higher KV cache thant the threshold, so the sheddable request will be
			// dropped.
			input: []*datastore.PodMetrics{
				{
					Pod: datastore.Pod{NamespacedName: types.NamespacedName{Name: "pod1"}},
					Metrics: datastore.Metrics{
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
					Pod: datastore.Pod{NamespacedName: types.NamespacedName{Name: "pod2"}},
					Metrics: datastore.Metrics{
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
					Pod: datastore.Pod{NamespacedName: types.NamespacedName{Name: "pod3"}},
					Metrics: datastore.Metrics{
						WaitingQueueSize:    10,
						KVCacheUsagePercent: 0.85,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo": 1,
						},
					},
				},
			},
			output: []*datastore.PodMetrics{},
			err:    true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := test.filter.Filter(logger, test.req, test.input)
			if test.err != (err != nil) {
				t.Errorf("Unexpected error, got %v, want %v", err, test.err)
			}

			if diff := cmp.Diff(test.output, got); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}

func TestFilterFunc(t *testing.T) {
	logger := logutil.NewTestLogger()

	tests := []struct {
		name   string
		f      filterFunc
		req    *LLMRequest
		input  []*datastore.PodMetrics
		output []*datastore.PodMetrics
		err    bool
	}{
		{
			name:   "least queuing empty input",
			f:      leastQueuingFilterFunc,
			input:  []*datastore.PodMetrics{},
			output: []*datastore.PodMetrics{},
		},
		{
			name: "least queuing",
			f:    leastQueuingFilterFunc,
			input: []*datastore.PodMetrics{
				{
					Metrics: datastore.Metrics{
						WaitingQueueSize: 0,
					},
				},
				{
					Metrics: datastore.Metrics{
						WaitingQueueSize: 3,
					},
				},
				{
					Metrics: datastore.Metrics{
						WaitingQueueSize: 10,
					},
				},
			},
			output: []*datastore.PodMetrics{
				{
					Metrics: datastore.Metrics{
						WaitingQueueSize: 0,
					},
				},
				{
					Metrics: datastore.Metrics{
						WaitingQueueSize: 3,
					},
				},
			},
		},
		{
			name:   "least kv cache empty input",
			f:      leastKVCacheFilterFunc,
			input:  []*datastore.PodMetrics{},
			output: []*datastore.PodMetrics{},
		},
		{
			name: "least kv cache",
			f:    leastKVCacheFilterFunc,
			input: []*datastore.PodMetrics{
				{
					Metrics: datastore.Metrics{
						KVCacheUsagePercent: 0,
					},
				},
				{
					Metrics: datastore.Metrics{
						KVCacheUsagePercent: 0.3,
					},
				},
				{
					Metrics: datastore.Metrics{
						KVCacheUsagePercent: 1.0,
					},
				},
			},
			output: []*datastore.PodMetrics{
				{
					Metrics: datastore.Metrics{
						KVCacheUsagePercent: 0,
					},
				},
				{
					Metrics: datastore.Metrics{
						KVCacheUsagePercent: 0.3,
					},
				},
			},
		},
		{
			name: "noQueueAndLessThanKVCacheThresholdPredicate",
			f:    toFilterFunc(noQueueAndLessThanKVCacheThresholdPredicate(0, 0.8)),
			input: []*datastore.PodMetrics{
				{
					// This pod should be returned.
					Metrics: datastore.Metrics{
						WaitingQueueSize:    0,
						KVCacheUsagePercent: 0,
					},
				},
				{
					// Queue is non zero, despite low kv cache, should not return.
					Metrics: datastore.Metrics{
						WaitingQueueSize:    1,
						KVCacheUsagePercent: 0.3,
					},
				},
				{
					// High kv cache despite zero queue, should not return
					Metrics: datastore.Metrics{
						WaitingQueueSize:    0,
						KVCacheUsagePercent: 1.0,
					},
				},
			},
			output: []*datastore.PodMetrics{
				{
					Metrics: datastore.Metrics{
						WaitingQueueSize:    0,
						KVCacheUsagePercent: 0,
					},
				},
			},
		},
		{
			name: "low LoRA cost",
			f:    toFilterFunc(lowLoRACostPredicate),
			req: &LLMRequest{
				Model:               "model",
				ResolvedTargetModel: "model",
			},
			input: []*datastore.PodMetrics{
				// ActiveModels include input model, should be returned.
				{
					Metrics: datastore.Metrics{
						MaxActiveModels: 2,
						ActiveModels: map[string]int{
							"model": 1,
						},
					},
				},
				// Input model is not active, however the server has room to load another adapter.
				{
					Metrics: datastore.Metrics{
						MaxActiveModels: 2,
						ActiveModels: map[string]int{
							"another-model": 1,
						},
					},
				},
				// Input is not active, and the server has reached max active models.
				{
					Metrics: datastore.Metrics{
						MaxActiveModels: 2,
						ActiveModels: map[string]int{
							"foo": 1,
							"bar": 1,
						},
					},
				},
			},
			output: []*datastore.PodMetrics{
				{
					Metrics: datastore.Metrics{
						MaxActiveModels: 2,
						ActiveModels: map[string]int{
							"model": 1,
						},
					},
				},
				{
					Metrics: datastore.Metrics{
						MaxActiveModels: 2,
						ActiveModels: map[string]int{
							"another-model": 1,
						},
					},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := test.f(logger, test.req, test.input)
			if test.err != (err != nil) {
				t.Errorf("Unexpected error, got %v, want %v", err, test.err)
			}

			if diff := cmp.Diff(test.output, got); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}
