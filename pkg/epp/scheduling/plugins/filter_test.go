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

package plugins

import (
	"context"
	"errors"
	"testing"

	"github.com/google/go-cmp/cmp"
	k8stypes "k8s.io/apimachinery/pkg/types"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/config"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

func TestFilter(t *testing.T) {
	tests := []struct {
		name   string
		req    *types.LLMRequest
		input  []types.Pod
		output []types.Pod
		err    bool
		filter *DecisionTreeFilter
	}{
		{
			name: "simple filter without successor, failure",
			filter: &DecisionTreeFilter{
				Current: &Filter{
					name: "error",
					filter: func(ctx *types.Context, pods []types.Pod) ([]types.Pod, error) {
						return nil, errors.New("filter error")
					},
				},
			},
			err: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx := types.NewContext(context.Background(), test.req, test.input)
			got, err := test.filter.Filter(ctx, test.input)
			if test.err != (err != nil) {
				t.Errorf("Unexpected error, got %v, want %v", err, test.err)
			}

			opt := cmp.AllowUnexported(types.PodMetrics{})
			if diff := cmp.Diff(test.output, got, opt); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}

func TestFilterFunc(t *testing.T) {
	tests := []struct {
		name   string
		f      filterFunc
		req    *types.LLMRequest
		input  []types.Pod
		output []types.Pod
		err    bool
	}{
		{
			name:   "least queuing empty input",
			f:      leastQueuingFilterFunc,
			input:  []types.Pod{},
			output: []types.Pod{},
		},
		{
			name: "least queuing",
			f:    leastQueuingFilterFunc,
			input: []types.Pod{
				&types.PodMetrics{
					Metrics: &backendmetrics.Metrics{
						WaitingQueueSize: 0,
					},
				},
				&types.PodMetrics{
					Metrics: &backendmetrics.Metrics{
						WaitingQueueSize: 3,
					},
				},
				&types.PodMetrics{
					Metrics: &backendmetrics.Metrics{
						WaitingQueueSize: 10,
					},
				},
			},
			output: []types.Pod{
				&types.PodMetrics{
					Metrics: &backendmetrics.Metrics{
						WaitingQueueSize: 0,
					},
				},
				&types.PodMetrics{
					Metrics: &backendmetrics.Metrics{
						WaitingQueueSize: 3,
					},
				},
			},
		},
		{
			name:   "least kv cache empty input",
			f:      leastKVCacheFilterFunc,
			input:  []types.Pod{},
			output: []types.Pod{},
		},
		{
			name: "least kv cache",
			f:    leastKVCacheFilterFunc,
			input: []types.Pod{
				&types.PodMetrics{
					Metrics: &backendmetrics.Metrics{
						KVCacheUsagePercent: 0,
					},
				},
				&types.PodMetrics{
					Metrics: &backendmetrics.Metrics{
						KVCacheUsagePercent: 0.3,
					},
				},
				&types.PodMetrics{
					Metrics: &backendmetrics.Metrics{
						KVCacheUsagePercent: 1.0,
					},
				},
			},
			output: []types.Pod{
				&types.PodMetrics{
					Metrics: &backendmetrics.Metrics{
						KVCacheUsagePercent: 0,
					},
				},
				&types.PodMetrics{
					Metrics: &backendmetrics.Metrics{
						KVCacheUsagePercent: 0.3,
					},
				},
			},
		},
		{
			name: "lowQueueAndLessThanKVCacheThresholdPredicate",
			f:    toFilterFunc(queueThresholdPredicate(0).and(kvCacheThresholdPredicate(0.8))),
			input: []types.Pod{
				&types.PodMetrics{
					// This pod should be returned.
					Metrics: &backendmetrics.Metrics{
						WaitingQueueSize:    0,
						KVCacheUsagePercent: 0,
					},
				},
				&types.PodMetrics{
					// Queue is non zero, despite low kv cache, should not return.
					Metrics: &backendmetrics.Metrics{
						WaitingQueueSize:    1,
						KVCacheUsagePercent: 0.3,
					},
				},
				&types.PodMetrics{
					// High kv cache despite zero queue, should not return
					Metrics: &backendmetrics.Metrics{
						WaitingQueueSize:    0,
						KVCacheUsagePercent: 1.0,
					},
				},
			},
			output: []types.Pod{
				&types.PodMetrics{
					Metrics: &backendmetrics.Metrics{
						WaitingQueueSize:    0,
						KVCacheUsagePercent: 0,
					},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx := types.NewContext(context.Background(), test.req, test.input)
			got, err := test.f(ctx, test.input)
			if test.err != (err != nil) {
				t.Errorf("Unexpected error, got %v, want %v", err, test.err)
			}

			opt := cmp.AllowUnexported(types.PodMetrics{})
			if diff := cmp.Diff(test.output, got, opt); diff != "" {
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
		Model:               testAffinityModel,
		ResolvedTargetModel: testAffinityModel,
	}

	// Test setup: One affinity pod and one available pod
	pods := []types.Pod{
		&types.PodMetrics{
			Pod: &backendmetrics.Pod{NamespacedName: k8stypes.NamespacedName{Name: "affinity-pod"}},
			Metrics: &backendmetrics.Metrics{
				MaxActiveModels: 2,
				ActiveModels: map[string]int{
					testAffinityModel: 1,
				},
			},
		},
		&types.PodMetrics{
			Pod: &backendmetrics.Pod{NamespacedName: k8stypes.NamespacedName{Name: "available-pod"}},
			Metrics: &backendmetrics.Metrics{
				MaxActiveModels: 2,
				ActiveModels:    map[string]int{},
			},
		},
	}
	ctx := types.NewContext(context.Background(), req, pods)

	// Run the filter function multiple times and count the results
	affinityCount := 0
	availableCount := 0

	// Use the test threshold value
	expectedAffinityPercent := config.Conf.LoraAffinityThreshold * 100
	expectedAvailabilityPercent := 100 - expectedAffinityPercent

	for i := 0; i < numIterations; i++ {
		result, err := loRASoftAffinityFilterFunc(ctx, pods)
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
