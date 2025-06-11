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
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/uuid"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/config"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

// compile-time type assertion
var _ framework.Filter = &filterAll{}

type filterAll struct{}

func (f *filterAll) Type() string {
	return "filter-all"
}

func (f *filterAll) Name() string {
	return "test-all"
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
			filter: &filterAll{},
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

func TestSubsettingFilter(t *testing.T) {
	var makeFilterMetadata = func(data []interface{}) map[string]any {
		return map[string]any{
			"envoy.lb.subset_hint": map[string]any{
				"x-gateway-destination-endpoint-subset": data,
			},
		}
	}

	tests := []struct {
		name     string
		metadata map[string]any
		filter   framework.Filter
		input    []types.Pod
		output   []types.Pod
	}{
		{
			name:     "SubsetFilter, filter not present — return all pods",
			filter:   &SubsetFilter{},
			metadata: map[string]any{},
			input: []types.Pod{
				&types.PodMetrics{
					Pod: &backend.Pod{Address: "10.0.0.1"},
				},
				&types.PodMetrics{
					Pod: &backend.Pod{Address: "10.0.0.2"},
				},
			},
			output: []types.Pod{
				&types.PodMetrics{
					Pod: &backend.Pod{Address: "10.0.0.1"},
				},
				&types.PodMetrics{
					Pod: &backend.Pod{Address: "10.0.0.2"},
				},
			},
		},
		{
			name:     "SubsetFilter, namespace present filter not present — return all pods",
			filter:   &SubsetFilter{},
			metadata: map[string]any{"envoy.lb.subset_hint": map[string]any{}},
			input: []types.Pod{
				&types.PodMetrics{
					Pod: &backend.Pod{Address: "10.0.0.1"},
				},
				&types.PodMetrics{
					Pod: &backend.Pod{Address: "10.0.0.2"},
				},
			},
			output: []types.Pod{
				&types.PodMetrics{
					Pod: &backend.Pod{Address: "10.0.0.1"},
				},
				&types.PodMetrics{
					Pod: &backend.Pod{Address: "10.0.0.2"},
				},
			},
		},
		{
			name:     "SubsetFilter, filter present with empty list — return no pods",
			filter:   &SubsetFilter{},
			metadata: makeFilterMetadata([]interface{}{}),
			input: []types.Pod{
				&types.PodMetrics{
					Pod: &backend.Pod{Address: "10.0.0.1"},
				},
				&types.PodMetrics{
					Pod: &backend.Pod{Address: "10.0.0.2"},
				},
			},
			output: []types.Pod{},
		},
		{
			name:     "SubsetFilter, subset with one matching pod",
			metadata: makeFilterMetadata([]interface{}{"10.0.0.1"}),
			filter:   &SubsetFilter{},
			input: []types.Pod{
				&types.PodMetrics{
					Pod: &backend.Pod{Address: "10.0.0.1"},
				},
				&types.PodMetrics{
					Pod: &backend.Pod{Address: "10.0.0.2"},
				},
			},
			output: []types.Pod{
				&types.PodMetrics{
					Pod: &backend.Pod{Address: "10.0.0.1"},
				},
			},
		},
		{
			name:     "SubsetFilter, subset with multiple matching pods",
			metadata: makeFilterMetadata([]interface{}{"10.0.0.1", "10.0.0.2", "10.0.0.3"}),
			filter:   &SubsetFilter{},
			input: []types.Pod{
				&types.PodMetrics{
					Pod: &backend.Pod{Address: "10.0.0.1"},
				},
				&types.PodMetrics{
					Pod: &backend.Pod{Address: "10.0.0.2"},
				},
			},
			output: []types.Pod{
				&types.PodMetrics{
					Pod: &backend.Pod{Address: "10.0.0.1"},
				},
				&types.PodMetrics{
					Pod: &backend.Pod{Address: "10.0.0.2"},
				},
			},
		},
		{
			name:     "SubsetFilter, subset with no matching pods",
			metadata: makeFilterMetadata([]interface{}{"10.0.0.3"}),
			filter:   &SubsetFilter{},
			input: []types.Pod{
				&types.PodMetrics{
					Pod: &backend.Pod{Address: "10.0.0.1"},
				},
				&types.PodMetrics{
					Pod: &backend.Pod{Address: "10.0.0.2"},
				},
			},
			output: []types.Pod{},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			req := types.NewLLMRequest(uuid.NewString(), "", "", nil, test.metadata)
			got := test.filter.Filter(context.Background(), types.NewCycleState(), req, test.input)

			if diff := cmp.Diff(test.output, got); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}
