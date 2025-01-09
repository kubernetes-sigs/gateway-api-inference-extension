package scheduling

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/backend"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

// A copy from filter_test.go
func TestOrchestratedFilterChain(t *testing.T) {
	fakeFilterConfigMap := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			UID:             types.UID("111"),
			ResourceVersion: "222",
		},
		Data: map[string]string{
			"filter": `
{
    "name": "critical_request",
    "nextOnSuccess": {
      "name": "low_latency",
      "nextOnSuccess": {
        "name": "affinity_lora",
        "nextOnSuccess": {
          "name": "least_queuing",
          "nextOnSuccessOrFailure": {
            "name": "least_kv_cache"
          }
        },
        "nextOnFailure": {
          "name": "can_accept_new_lora",
          "nextOnSuccessOrFailure": {
            "name": "least_queuing",
            "nextOnSuccessOrFailure": {
              "name": "least_kv_cache"
            }
          }
        }
      },
      "nextOnFailure": {
        "name": "least_queuing",
        "nextOnSuccessOrFailure": {
          "name": "low_cost_lora",
          "nextOnSuccessOrFailure": {
            "name": "least_kv_cache"
          }
        }
      }
    },
    "nextOnFailure": {
      "name": "sheddable_request",
      "nextOnSuccess": {
        "name": "least_queuing",
        "nextOnSuccessOrFailure": {
          "name": "low_cost_lora",
          "nextOnSuccessOrFailure": {
            "name": "least_kv_cache"
          }
        }
      },
      "nextOnFailure": {
        "name": "drop_request"
      }
    }
  }
`,
		},
	}
	datastore := backend.NewK8sDataStore(backend.WithFilterConfigMap(fakeFilterConfigMap))
	o := NewFilterOrchestrator(datastore)
	tests := []struct {
		name   string
		req    *LLMRequest
		input  []*backend.PodMetrics
		output []*backend.PodMetrics
		err    bool
		filter *filterChainImpl
	}{
		{
			name:   "orchestrated filter, critical request",
			filter: o.Orchestrate().(*filterChainImpl),
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
			name:   "orchestrated filter, sheddable request, accepted",
			filter: o.Orchestrate().(*filterChainImpl),
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
			name:   "orchestrated filter, sheddable request, dropped",
			filter: o.Orchestrate().(*filterChainImpl),
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
