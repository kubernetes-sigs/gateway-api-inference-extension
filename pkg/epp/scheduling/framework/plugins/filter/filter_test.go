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

package filter

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/google/uuid"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/config"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/scorer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	"sigs.k8s.io/gateway-api-inference-extension/test/utils"
)

// compile-time type assertion
var _ framework.Filter = &filterAll{}

type filterAll struct {
	tn plugins.TypedName
}

func (f *filterAll) TypedName() plugins.TypedName {
	return f.tn
}

func newFilterAll() *filterAll {
	return &filterAll{
		tn: plugins.TypedName{Type: "filter-all", Name: "test-all"},
	}
}

func (f *filterAll) Filter(_ context.Context, _ *types.CycleState, _ *types.LLMRequest, pods []types.Pod) []types.Pod {
	return []types.Pod{}
}

func TestFilter(t *testing.T) {
	tests := []struct {
		name   string
		req    *types.LLMRequest
		filter framework.Filter
		input  []types.Pod
		output []types.Pod
	}{
		{
			name:   "simple filter filters all pods",
			filter: newFilterAll(),
			output: []types.Pod{},
		},
		{
			name:   "least queuing empty input",
			filter: NewLeastQueueFilter(),
			input:  []types.Pod{},
			output: []types.Pod{},
		},
		{
			name:   "least queuing",
			filter: NewLeastQueueFilter(),
			input: []types.Pod{
				&types.PodMetrics{
					MetricsState: &backendmetrics.MetricsState{
						WaitingQueueSize: 0,
					},
				},
				&types.PodMetrics{
					MetricsState: &backendmetrics.MetricsState{
						WaitingQueueSize: 3,
					},
				},
				&types.PodMetrics{
					MetricsState: &backendmetrics.MetricsState{
						WaitingQueueSize: 10,
					},
				},
			},
			output: []types.Pod{
				&types.PodMetrics{
					MetricsState: &backendmetrics.MetricsState{
						WaitingQueueSize: 0,
					},
				},
				&types.PodMetrics{
					MetricsState: &backendmetrics.MetricsState{
						WaitingQueueSize: 3,
					},
				},
			},
		},
		{
			name:   "least kv cache empty input",
			filter: NewLeastKVCacheFilter(),
			input:  []types.Pod{},
			output: []types.Pod{},
		},
		{
			name:   "least kv cache",
			filter: NewLeastKVCacheFilter(),
			input: []types.Pod{
				&types.PodMetrics{
					MetricsState: &backendmetrics.MetricsState{
						KVCacheUsagePercent: 0,
					},
				},
				&types.PodMetrics{
					MetricsState: &backendmetrics.MetricsState{
						KVCacheUsagePercent: 0.3,
					},
				},
				&types.PodMetrics{
					MetricsState: &backendmetrics.MetricsState{
						KVCacheUsagePercent: 1.0,
					},
				},
			},
			output: []types.Pod{
				&types.PodMetrics{
					MetricsState: &backendmetrics.MetricsState{
						KVCacheUsagePercent: 0,
					},
				},
				&types.PodMetrics{
					MetricsState: &backendmetrics.MetricsState{
						KVCacheUsagePercent: 0.3,
					},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := test.filter.Filter(context.Background(), types.NewCycleState(), test.req, test.input)

			if diff := cmp.Diff(test.output, got); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}

// TestLoRASoftAffinityDistribution tests that the loRASoftAffinityFilter function
// properly distributes requests according to the loraAffinityThreshold
func TestLoRASoftAffinityDistribution(t *testing.T) {
	const (
		testModelName     = "test-model"
		testAffinityModel = "test-affinity-model"
		numIterations     = 10000
		tolerancePercent  = 5.0 // Allow 5% tolerance from expected distribution
	)

	// Save original config value to restore later
	originalThreshold := config.Conf.LoraAffinityThreshold

	// Set a specific test value for this test
	testThreshold := 0.75 // 75%
	config.Conf.LoraAffinityThreshold = testThreshold

	// Ensure we restore the original threshold when test completes
	defer func() {
		config.Conf.LoraAffinityThreshold = originalThreshold
	}()

	// Create a test request and pods
	req := &types.LLMRequest{
		TargetModel: testAffinityModel,
		RequestId:   uuid.NewString(),
	}

	// Test setup: One affinity pod and one available pod
	pods := []types.Pod{
		&types.PodMetrics{
			Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "affinity-pod"}},
			MetricsState: &backendmetrics.MetricsState{
				MaxActiveModels: 2,
				ActiveModels: map[string]int{
					testAffinityModel: 1,
				},
			},
		},
		&types.PodMetrics{
			Pod: &backend.Pod{NamespacedName: k8stypes.NamespacedName{Name: "available-pod"}},
			MetricsState: &backendmetrics.MetricsState{
				MaxActiveModels: 2,
				ActiveModels:    map[string]int{},
			},
		},
	}
	// Run the filter function multiple times and count the results
	affinityCount := 0
	availableCount := 0

	// Use the test threshold value
	expectedAffinityPercent := config.Conf.LoraAffinityThreshold * 100
	expectedAvailabilityPercent := 100 - expectedAffinityPercent

	// initialize LoraAffinityFilter
	LoraAffinityFilter := NewLoraAffinityFilter(config.Conf.LoraAffinityThreshold)

	for range numIterations {
		result := LoraAffinityFilter.Filter(context.Background(), types.NewCycleState(), req, pods)

		// Check which type of pod was returned
		if len(result) != 1 {
			t.Fatalf("Expected exactly one pod in result, got %d", len(result))
		}

		// Identify if the returned pod is the affinity pod or available pod
		if _, exists := result[0].GetMetrics().ActiveModels[testAffinityModel]; exists {
			affinityCount++
		} else {
			availableCount++
		}
	}

	// Calculate the actual percentages
	actualAffinityPercent := float64(affinityCount) / float64(numIterations) * 100
	actualAvailablePercent := float64(availableCount) / float64(numIterations) * 100

	// Check if the distribution matches expected threshold within tolerance
	affinityLowerBound := expectedAffinityPercent - tolerancePercent
	affinityUpperBound := expectedAffinityPercent + tolerancePercent

	availableLowerBound := expectedAvailabilityPercent - tolerancePercent
	availableUpperBound := expectedAvailabilityPercent + tolerancePercent

	t.Logf("Distribution results over %d iterations:", numIterations)
	t.Logf("Expected affinity percent: %.2f%% (threshold: %.2f)", expectedAffinityPercent, config.Conf.LoraAffinityThreshold)
	t.Logf("Expected availability percent: %.2f%% (threshold: %.2f)", expectedAvailabilityPercent, config.Conf.LoraAffinityThreshold)
	t.Logf("Actual affinity percent: %.2f%% (%d out of %d)", actualAffinityPercent, affinityCount, numIterations)
	t.Logf("Actual available percent: %.2f%% (%d out of %d)", actualAvailablePercent, availableCount, numIterations)

	if actualAffinityPercent < affinityLowerBound || actualAffinityPercent > affinityUpperBound {
		t.Errorf("Affinity selection percent %.2f%% outside expected range %.2f%% to %.2f%%",
			actualAffinityPercent, affinityLowerBound, affinityUpperBound)
	}
	if actualAvailablePercent < availableLowerBound || actualAvailablePercent > availableUpperBound {
		t.Errorf("Availability selection percent %.2f%% outside expected range %.2f%% to %.2f%%",
			actualAvailablePercent, availableLowerBound, availableUpperBound)
	}
}

// TestDecisionTreeFilterFactory tests that the DecisionTreeFilterFactory function
// properly instantiates DecisionTreeFilter instances
func TestDecisionTreeFilterFactory(t *testing.T) {

	leastKvCacheFilter := NewLeastKVCacheFilter()
	leastQueueFilter := NewLeastQueueFilter()
	loraAffinityFilter := NewLoraAffinityFilter(config.Conf.LoraAffinityThreshold)
	lowQueueFilter := NewLowQueueFilter(config.Conf.QueueingThresholdLoRA)

	kvCacheScorer := scorer.NewKVCacheScorer()

	testHandle := utils.NewTestHandle(context.Background())

	testHandle.AddPlugin("leastKvCache", leastKvCacheFilter)
	testHandle.AddPlugin("leastQueue", leastQueueFilter)
	testHandle.AddPlugin("loraAffinity", loraAffinityFilter)
	testHandle.AddPlugin("lowQueue", lowQueueFilter)

	testHandle.AddPlugin("kvCacheScorer", kvCacheScorer)

	tests := []struct {
		name       string
		parameters string
		want       *DecisionTreeFilter
		wantErr    bool
	}{
		{
			name:       "success",
			parameters: decisionTreeParametersSuccess,
			want: &DecisionTreeFilter{
				Current: lowQueueFilter,
				NextOnSuccess: &DecisionTreeFilter{
					Current: loraAffinityFilter,
					NextOnSuccessOrFailure: &DecisionTreeFilter{
						Current: leastQueueFilter,
						NextOnSuccessOrFailure: &DecisionTreeFilter{
							Current: leastKvCacheFilter,
						},
					},
				},
				NextOnFailure: &DecisionTreeFilter{
					Current: leastQueueFilter,
					NextOnSuccessOrFailure: &DecisionTreeFilter{
						Current: loraAffinityFilter,
						NextOnSuccessOrFailure: &DecisionTreeFilter{
							Current: leastKvCacheFilter,
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name:       "bothError",
			parameters: decisionTreeParametersErrorBoth,
			want:       nil,
			wantErr:    true,
		},
		{
			name:       "noneError",
			parameters: decisionTreeParametersErrorNone,
			want:       nil,
			wantErr:    true,
		},
		{
			name:       "badPlugin",
			parameters: decisionTreeParametersErrorBadPlugin,
			want:       nil,
			wantErr:    true,
		},
		{
			name:       "notFilter",
			parameters: decisionTreeParametersErrorNotFilter,
			want:       nil,
			wantErr:    true,
		},
		{
			name:       "noCurrent",
			parameters: decisionTreeParametersErrorNoCurrent,
			want:       nil,
			wantErr:    true,
		},
		{
			name:       "badNextOnSuccess",
			parameters: decisionTreeParametersErrorBadNextOnSuccess,
			want:       nil,
			wantErr:    true,
		},
		{
			name:       "badNextOnFailure",
			parameters: decisionTreeParametersErrorBadNextOnFailure,
			want:       nil,
			wantErr:    true,
		},
		{
			name:       "badNextOnSuccessOrFailure",
			parameters: decisionTreeParametersErrorBadNextOnSuccessOrFailure,
			want:       nil,
			wantErr:    true,
		},
	}

	cmpOptions := cmpopts.IgnoreUnexported(LeastKVCacheFilter{}, LeastQueueFilter{},
		LoraAffinityFilter{}, LowQueueFilter{}, scorer.KVCacheScorer{}, plugins.TypedName{})

	for _, test := range tests {
		rawParameters := struct {
			Parameters json.RawMessage `json:"parameters"`
		}{}
		err := json.Unmarshal([]byte(test.parameters), &rawParameters)
		if err != nil {
			if test.wantErr {
				continue
			} else {
				t.Fatal("failed to parse JSON of test " + test.name)
			}
		}
		got, err := DecisionTreeFilterFactory("testing", rawParameters.Parameters, testHandle)
		if err != nil {
			if test.wantErr {
				continue
			}
			t.Fatalf("failed to instantiate DecisionTreeFilter. error: %s\n", err)
		}
		if test.wantErr {
			t.Fatalf("test %s did not return the expected error", test.name)
		}
		if diff := cmp.Diff(test.want, got, cmpOptions); diff != "" {
			t.Fatalf("In test %s DecisionTreeFactory returned unexpected response, diff(-want, +got): %v", test.name, diff)
		}
	}
}

const decisionTreeParametersSuccess = `
{
  "parameters": {
    "current": {
      "pluginRef": "lowQueue"
    },
    "nextOnSuccess": {
      "decisionTree": {
	    "current": {
          "pluginRef": "loraAffinity"
        },
        "nextOnSuccessOrFailure": {
          "decisionTree": {
	        "current": {
		      "pluginRef": "leastQueue"
            },
            "nextOnSuccessOrFailure": {
			  "decisionTree": {
	            "current": {
		          "pluginRef": "leastKvCache"
                }
              }
            }
          }
	    }
	  }
    },
    "nextOnFailure": {
      "decisionTree": {
	    "current": {
          "pluginRef": "leastQueue"
        },
        "nextOnSuccessOrFailure": {
		  "decisionTree": {
	        "current": {
		      "pluginRef": "loraAffinity"
            },
	        "nextOnSuccessOrFailure": {
	          "decisionTree": {
			    "current": {
		          "pluginRef": "leastKvCache"
                }
              }
            }
          }
        }
	  }
	}
  }
}
`

const decisionTreeParametersErrorBoth = `
{
  "parameters": {
    "current": {
      "pluginRef": "lowQueue",
	  "decisionTree": {
	    "current": {
          "pluginRef": "leastKvCache"
        }
      }
    }
  }
}
`

const decisionTreeParametersErrorNone = `
{
  "parameters": {
    "current": {
    }
  }
}
`

const decisionTreeParametersErrorBadPlugin = `
{
  "parameters": {
    "current": {
      "pluginRef": "plover"
    }
  }
}
`

const decisionTreeParametersErrorNotFilter = `
{
  "parameters": {
    "current": {
      "pluginRef": "kvCacheScorer"
    }
  }
}
`

const decisionTreeParametersErrorNoCurrent = `
{
  "parameters": {
    "NextOnSuccess": {
      "pluginRef": "lowQueue"
    }
  }
}
`

const decisionTreeParametersErrorBadNextOnSuccess = `
{
  "parameters": {
    "current": {
      "pluginRef": "lowQueue"
    },
    "NextOnSuccess": {
      "pluginRef": "kvCacheScorer"
    }
  }
}
`

const decisionTreeParametersErrorBadNextOnFailure = `
{
  "parameters": {
    "current": {
      "pluginRef": "lowQueue"
    },
    "NextOnFailure": {
      "pluginRef": "kvCacheScorer"
    }
  }
}
`

const decisionTreeParametersErrorBadNextOnSuccessOrFailure = `
{
  "parameters": {
    "current": {
      "pluginRef": "lowQueue"
    },
    "NextOnSuccessOrFailure": {
      "pluginRef": "kvCacheScorer"
    }
  }
}
`
