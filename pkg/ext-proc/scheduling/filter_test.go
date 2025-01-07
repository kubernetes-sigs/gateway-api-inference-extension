package scheduling

import (
	"errors"
	"testing"

	"github.com/google/go-cmp/cmp"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/backend"
)

func TestFilter(t *testing.T) {
	tests := []struct {
		name   string
		req    *LLMRequest
		input  []*backend.PodMetrics
		output []*backend.PodMetrics
		err    bool
		filter *filter
	}{
		{
			name: "simple filter without successor, failure",
			filter: &filter{filter: func(req *LLMRequest, pods []*backend.PodMetrics) ([]*backend.PodMetrics, error) {
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
			input: []*backend.PodMetrics{
				{
					Pod: backend.Pod{Name: "pod1"},
					Metrics: backend.Metrics{
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
					Pod: backend.Pod{Name: "pod2"},
					Metrics: backend.Metrics{
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
					Pod: backend.Pod{Name: "pod3"},
					Metrics: backend.Metrics{
						WaitingQueueSize:    10,
						KVCacheUsagePercent: 0.2,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo": 1,
						},
					},
				},
			},
			output: []*backend.PodMetrics{
				{
					Pod: backend.Pod{Name: "pod2"},
					Metrics: backend.Metrics{
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
			input: []*backend.PodMetrics{
				{
					Pod: backend.Pod{Name: "pod1"},
					Metrics: backend.Metrics{
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
					Pod: backend.Pod{Name: "pod2"},
					Metrics: backend.Metrics{
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
					Pod: backend.Pod{Name: "pod3"},
					Metrics: backend.Metrics{
						WaitingQueueSize:    10,
						KVCacheUsagePercent: 0.2,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo": 1,
						},
					},
				},
			},
			output: []*backend.PodMetrics{
				{
					Pod: backend.Pod{Name: "pod1"},
					Metrics: backend.Metrics{
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
			input: []*backend.PodMetrics{
				{
					Pod: backend.Pod{Name: "pod1"},
					Metrics: backend.Metrics{
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
					Pod: backend.Pod{Name: "pod2"},
					Metrics: backend.Metrics{
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
					Pod: backend.Pod{Name: "pod3"},
					Metrics: backend.Metrics{
						WaitingQueueSize:    10,
						KVCacheUsagePercent: 0.85,
						MaxActiveModels:     2,
						ActiveModels: map[string]int{
							"foo": 1,
						},
					},
				},
			},
			output: []*backend.PodMetrics{},
			err:    true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := test.filter.Filter(test.req, test.input)
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
	tests := []struct {
		name   string
		f      filterFunc
		req    *LLMRequest
		input  []*backend.PodMetrics
		output []*backend.PodMetrics
		err    bool
	}{
		{
			name:   "least queuing empty input",
			f:      leastQueuingFilterFunc,
			input:  []*backend.PodMetrics{},
			output: []*backend.PodMetrics{},
		},
		{
			name: "least queuing",
			f:    leastQueuingFilterFunc,
			input: []*backend.PodMetrics{
				{
					Metrics: backend.Metrics{
						WaitingQueueSize: 0,
					},
				},
				{
					Metrics: backend.Metrics{
						WaitingQueueSize: 3,
					},
				},
				{
					Metrics: backend.Metrics{
						WaitingQueueSize: 10,
					},
				},
			},
			output: []*backend.PodMetrics{
				{
					Metrics: backend.Metrics{
						WaitingQueueSize: 0,
					},
				},
				{
					Metrics: backend.Metrics{
						WaitingQueueSize: 3,
					},
				},
			},
		},
		{
			name:   "least kv cache empty input",
			f:      leastKVCacheFilterFunc,
			input:  []*backend.PodMetrics{},
			output: []*backend.PodMetrics{},
		},
		{
			name: "least kv cache",
			f:    leastKVCacheFilterFunc,
			input: []*backend.PodMetrics{
				{
					Metrics: backend.Metrics{
						KVCacheUsagePercent: 0,
					},
				},
				{
					Metrics: backend.Metrics{
						KVCacheUsagePercent: 0.3,
					},
				},
				{
					Metrics: backend.Metrics{
						KVCacheUsagePercent: 1.0,
					},
				},
			},
			output: []*backend.PodMetrics{
				{
					Metrics: backend.Metrics{
						KVCacheUsagePercent: 0,
					},
				},
				{
					Metrics: backend.Metrics{
						KVCacheUsagePercent: 0.3,
					},
				},
			},
		},
		{
			name: "noQueueAndLessThanKVCacheThresholdPredicate",
			f:    toFilterFunc(noQueueAndLessThanKVCacheThresholdPredicate(0, 0.8)),
			input: []*backend.PodMetrics{
				{
					// This pod should be returned.
					Metrics: backend.Metrics{
						WaitingQueueSize:    0,
						KVCacheUsagePercent: 0,
					},
				},
				{
					// Queue is non zero, despite low kv cache, should not return.
					Metrics: backend.Metrics{
						WaitingQueueSize:    1,
						KVCacheUsagePercent: 0.3,
					},
				},
				{
					// High kv cache despite zero queue, should not return
					Metrics: backend.Metrics{
						WaitingQueueSize:    0,
						KVCacheUsagePercent: 1.0,
					},
				},
			},
			output: []*backend.PodMetrics{
				{
					Metrics: backend.Metrics{
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
			input: []*backend.PodMetrics{
				// ActiveModels include input model, should be returned.
				{
					Metrics: backend.Metrics{
						MaxActiveModels: 2,
						ActiveModels: map[string]int{
							"model": 1,
						},
					},
				},
				// Input model is not active, however the server has room to load another adapter.
				{
					Metrics: backend.Metrics{
						MaxActiveModels: 2,
						ActiveModels: map[string]int{
							"another-model": 1,
						},
					},
				},
				// Input is not active, and the server has reached max active models.
				{
					Metrics: backend.Metrics{
						MaxActiveModels: 2,
						ActiveModels: map[string]int{
							"foo": 1,
							"bar": 1,
						},
					},
				},
			},
			output: []*backend.PodMetrics{
				{
					Metrics: backend.Metrics{
						MaxActiveModels: 2,
						ActiveModels: map[string]int{
							"model": 1,
						},
					},
				},
				{
					Metrics: backend.Metrics{
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
			got, err := test.f(test.req, test.input)
			if test.err != (err != nil) {
				t.Errorf("Unexpected error, got %v, want %v", err, test.err)
			}

			if diff := cmp.Diff(test.output, got); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}
