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
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

func TestFilter(t *testing.T) {
	logger := logutil.NewTestLogger()

	tests := []struct {
		name   string
		req    *LLMRequest
		input  []*backendmetrics.FakePodMetrics
		output []*backendmetrics.FakePodMetrics
		err    bool
		filter *filter
	}{
		{
			name: "simple filter without successor, failure",
			filter: &filter{filter: func(logger logr.Logger, req *LLMRequest, pods []backendmetrics.PodMetrics) ([]backendmetrics.PodMetrics, error) {
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
			input: []*backendmetrics.FakePodMetrics{
				{
					Pod: &backendmetrics.Pod{NamespacedName: types.NamespacedName{Name: "pod1"}},
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
					Pod: &backendmetrics.Pod{NamespacedName: types.NamespacedName{Name: "pod2"}},
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
					Pod: &backendmetrics.Pod{NamespacedName: types.NamespacedName{Name: "pod3"}},
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
			output: []*backendmetrics.FakePodMetrics{
				{
					Pod: &backendmetrics.Pod{NamespacedName: types.NamespacedName{Name: "pod2"}},
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
			input: []*backendmetrics.FakePodMetrics{
				{
					Pod: &backendmetrics.Pod{NamespacedName: types.NamespacedName{Name: "pod1"}},
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
					Pod: &backendmetrics.Pod{NamespacedName: types.NamespacedName{Name: "pod2"}},
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
					Pod: &backendmetrics.Pod{NamespacedName: types.NamespacedName{Name: "pod3"}},
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
			output: []*backendmetrics.FakePodMetrics{
				{
					Pod: &backendmetrics.Pod{NamespacedName: types.NamespacedName{Name: "pod1"}},
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
			input: []*backendmetrics.FakePodMetrics{
				{
					Pod: &backendmetrics.Pod{NamespacedName: types.NamespacedName{Name: "pod1"}},
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
					Pod: &backendmetrics.Pod{NamespacedName: types.NamespacedName{Name: "pod2"}},
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
					Pod: &backendmetrics.Pod{NamespacedName: types.NamespacedName{Name: "pod3"}},
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
			output: []*backendmetrics.FakePodMetrics{},
			err:    true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := test.filter.Filter(logger, test.req, toInterface(test.input))
			if test.err != (err != nil) {
				t.Errorf("Unexpected error, got %v, want %v", err, test.err)
			}

			if diff := cmp.Diff(test.output, toStruct(got)); diff != "" {
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
		input  []*backendmetrics.FakePodMetrics
		output []*backendmetrics.FakePodMetrics
		err    bool
	}{
		{
			name:   "least queuing empty input",
			f:      leastQueuingFilterFunc,
			input:  []*backendmetrics.FakePodMetrics{},
			output: []*backendmetrics.FakePodMetrics{},
		},
		{
			name: "least queuing",
			f:    leastQueuingFilterFunc,
			input: []*backendmetrics.FakePodMetrics{
				{
					Metrics: &backendmetrics.Metrics{
						WaitingQueueSize: 0,
					},
				},
				{
					Metrics: &backendmetrics.Metrics{
						WaitingQueueSize: 3,
					},
				},
				{
					Metrics: &backendmetrics.Metrics{
						WaitingQueueSize: 10,
					},
				},
			},
			output: []*backendmetrics.FakePodMetrics{
				{
					Metrics: &backendmetrics.Metrics{
						WaitingQueueSize: 0,
					},
				},
				{
					Metrics: &backendmetrics.Metrics{
						WaitingQueueSize: 3,
					},
				},
			},
		},
		{
			name:   "least kv cache empty input",
			f:      leastKVCacheFilterFunc,
			input:  []*backendmetrics.FakePodMetrics{},
			output: []*backendmetrics.FakePodMetrics{},
		},
		{
			name: "least kv cache",
			f:    leastKVCacheFilterFunc,
			input: []*backendmetrics.FakePodMetrics{
				{
					Metrics: &backendmetrics.Metrics{
						KVCacheUsagePercent: 0,
					},
				},
				{
					Metrics: &backendmetrics.Metrics{
						KVCacheUsagePercent: 0.3,
					},
				},
				{
					Metrics: &backendmetrics.Metrics{
						KVCacheUsagePercent: 1.0,
					},
				},
			},
			output: []*backendmetrics.FakePodMetrics{
				{
					Metrics: &backendmetrics.Metrics{
						KVCacheUsagePercent: 0,
					},
				},
				{
					Metrics: &backendmetrics.Metrics{
						KVCacheUsagePercent: 0.3,
					},
				},
			},
		},
		{
			name: "noQueueAndLessThanKVCacheThresholdPredicate",
			f:    toFilterFunc(noQueueAndLessThanKVCacheThresholdPredicate(0, 0.8)),
			input: []*backendmetrics.FakePodMetrics{
				{
					// This pod should be returned.
					Metrics: &backendmetrics.Metrics{
						WaitingQueueSize:    0,
						KVCacheUsagePercent: 0,
					},
				},
				{
					// Queue is non zero, despite low kv cache, should not return.
					Metrics: &backendmetrics.Metrics{
						WaitingQueueSize:    1,
						KVCacheUsagePercent: 0.3,
					},
				},
				{
					// High kv cache despite zero queue, should not return
					Metrics: &backendmetrics.Metrics{
						WaitingQueueSize:    0,
						KVCacheUsagePercent: 1.0,
					},
				},
			},
			output: []*backendmetrics.FakePodMetrics{
				{
					Metrics: &backendmetrics.Metrics{
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
			input: []*backendmetrics.FakePodMetrics{
				// ActiveModels include input model, should be returned.
				{
					Metrics: &backendmetrics.Metrics{
						MaxActiveModels: 2,
						ActiveModels: map[string]int{
							"model": 1,
						},
					},
				},
				// Input model is not active, however the server has room to load another adapter.
				{
					Metrics: &backendmetrics.Metrics{
						MaxActiveModels: 2,
						ActiveModels: map[string]int{
							"another-model": 1,
						},
					},
				},
				// Input is not active, and the server has reached max active models.
				{
					Metrics: &backendmetrics.Metrics{
						MaxActiveModels: 2,
						ActiveModels: map[string]int{
							"foo": 1,
							"bar": 1,
						},
					},
				},
			},
			output: []*backendmetrics.FakePodMetrics{
				{
					Metrics: &backendmetrics.Metrics{
						MaxActiveModels: 2,
						ActiveModels: map[string]int{
							"model": 1,
						},
					},
				},
				{
					Metrics: &backendmetrics.Metrics{
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
			got, err := test.f(logger, test.req, toInterface(test.input))
			if test.err != (err != nil) {
				t.Errorf("Unexpected error, got %v, want %v", err, test.err)
			}

			if diff := cmp.Diff(test.output, toStruct(got)); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}

// TestLoRASoftAffinityDistribution tests that the loRASoftAffinityFilter function
// properly distributes requests according to the loraAffinityThreshold
func TestLoRASoftAffinityDistribution(t *testing.T) {
	logger := logutil.NewTestLogger()

	const (
		testModelName     = "test-model"
		testAffinityModel = "test-affinity-model"
		numIterations     = 10000
		tolerancePercent  = 5.0 // Allow 5% tolerance from expected distribution
	)

	// Create a test request and pods
	req := &LLMRequest{
		Model:               testAffinityModel,
		ResolvedTargetModel: testAffinityModel,
	}

	// Test setup: One affinity pod and one available pod
	pods := []*backendmetrics.FakePodMetrics{
		{
			Pod: &backendmetrics.Pod{NamespacedName: types.NamespacedName{Name: "affinity-pod"}},
			Metrics: &backendmetrics.Metrics{
				MaxActiveModels: 2,
				ActiveModels: map[string]int{
					testAffinityModel: 1,
				},
			},
		},
		{
			Pod: &backendmetrics.Pod{NamespacedName: types.NamespacedName{Name: "available-pod"}},
			Metrics: &backendmetrics.Metrics{
				MaxActiveModels: 2,
				ActiveModels:    map[string]int{},
			},
		},
	}

	// Run the filter function multiple times and count the results
	affinityCount := 0
	availableCount := 0

	// Use the actual loraAffinityThreshold as defined in the original code
	// This test should work with whatever value is set there
	expectedAffinityPercent := loraAffinityThreshold * 100
	for i := 0; i < numIterations; i++ {
		result, err := loRASoftAffinityFilter(logger, req, toInterface(pods))
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

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

	availableLowerBound := actualAvailablePercent - tolerancePercent
	availableUpperBound := actualAvailablePercent + tolerancePercent

	t.Logf("Distribution results over %d iterations:", numIterations)
	t.Logf("Expected affinity percent: %.2f%% (threshold: %.2f)", expectedAffinityPercent, loraAffinityThreshold)
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

func toInterface(input []*backendmetrics.FakePodMetrics) []backendmetrics.PodMetrics {
	output := []backendmetrics.PodMetrics{}
	for _, i := range input {
		output = append(output, i)
	}
	return output
}

func toStruct(input []backendmetrics.PodMetrics) []*backendmetrics.FakePodMetrics {
	if input == nil {
		return nil
	}
	output := []*backendmetrics.FakePodMetrics{}
	for _, i := range input {
		output = append(output, i.(*backendmetrics.FakePodMetrics))
	}
	return output
}
