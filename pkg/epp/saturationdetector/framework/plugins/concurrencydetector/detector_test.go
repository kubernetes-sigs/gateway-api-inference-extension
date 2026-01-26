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

package concurrencydetector

import (
	"context"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

// TestNewPlugin_Configuration validates that the plugin correctly applies defaults and respects explicit configuration
// values.
func TestNewPlugin_Configuration(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name                   string
		config                 Config
		effectiveMax           int64
		effectiveHeadroomBurst int64
	}{
		{
			name:                   "defaults_applied_on_zero_values",
			config:                 Config{}, // Zero values
			effectiveMax:           100,      // DefaultMaxConcurrency
			effectiveHeadroomBurst: 100,      // Default Headroom is 0.0, so limit == max
		},
		{
			name: "explicit_values_respected",
			config: Config{
				MaxConcurrency: 50,
				Headroom:       0.2, // 20% burst
			},
			effectiveMax:           50,
			effectiveHeadroomBurst: 60, // 50 * 1.2 = 60
		},
		{
			name: "negative_max_resets_to_default",
			config: Config{
				MaxConcurrency: -10,
			},
			effectiveMax:           100,
			effectiveHeadroomBurst: 100,
		},
		{
			name: "negative_headroom_resets_to_default",
			config: Config{
				MaxConcurrency: 10,
				Headroom:       -0.5,
			},
			effectiveMax:           10,
			effectiveHeadroomBurst: 10, // Headroom resets to 0.0
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			ctx := context.Background()
			detector := NewDetector(tc.config)
			endpointName := "test-endpoint"

			// 1. Verify MaxConcurrency via IsSaturated

			// A. Drive load to just below the limit (Max - 1)
			driveLoad(ctx, detector, endpointName, int(tc.effectiveMax-1))
			require.False(t, detector.IsSaturated(ctx, []backendmetrics.PodMetrics{newFakePodMetric(endpointName)}),
				"expected NOT saturated at MaxConcurrency-1")

			// B. Increment to exactly the limit (Max)
			driveLoad(ctx, detector, endpointName, 1)
			require.True(t, detector.IsSaturated(ctx, []backendmetrics.PodMetrics{newFakePodMetric(endpointName)}),
				"expected saturated at MaxConcurrency")

			// 2. Verify Headroom via Filter

			// Reset state first.
			detector.DeleteEndpoint(fullEndpointName(endpointName))

			// A. Drive load to just below the burst limit (Limit - 1)
			driveLoad(ctx, detector, endpointName, int(tc.effectiveHeadroomBurst-1))
			kept := detector.Filter(ctx, nil, nil, []schedulingtypes.Endpoint{newStubSchedulingEndpoint(endpointName)})
			require.Len(t, kept, 1, "expected endpoint to be KEPT at BurstLimit-1")

			// B. Reach the burst limit (Limit)
			driveLoad(ctx, detector, endpointName, 1)
			kept = detector.Filter(ctx, nil, nil, []schedulingtypes.Endpoint{newStubSchedulingEndpoint(endpointName)})
			require.Len(t, kept, 0, "expected endpoint to be FILTERED at BurstLimit")
		})
	}
}

// TestDetector_IsSaturated verifies the global circuit breaker logic.
// It ensures saturation is reported ONLY when ALL candidate endpoints are full.
func TestDetector_IsSaturated(t *testing.T) {
	t.Parallel()

	const maxConcurrency = 5
	config := Config{MaxConcurrency: maxConcurrency}

	tests := []struct {
		name               string
		endpointLoadSetup  map[string]int // Map of EndpointName -> Request Count
		candidateEndpoints []string       // Endpoints passed to IsSaturated
		wantSaturation     bool
	}{
		{
			name:               "empty_candidate_list_fail_closed",
			endpointLoadSetup:  nil,
			candidateEndpoints: []string{},
			wantSaturation:     true,
		},
		{
			name:               "single_endpoint_with_capacity",
			endpointLoadSetup:  map[string]int{"endpoint-a": 4},
			candidateEndpoints: []string{"endpoint-a"},
			wantSaturation:     false,
		},
		{
			name:               "single_endpoint_full",
			endpointLoadSetup:  map[string]int{"endpoint-a": 5},
			candidateEndpoints: []string{"endpoint-a"},
			wantSaturation:     true,
		},
		{
			name:               "multi_endpoint_one_available",
			endpointLoadSetup:  map[string]int{"endpoint-a": 5, "endpoint-b": 4},
			candidateEndpoints: []string{"endpoint-a", "endpoint-b"},
			wantSaturation:     false,
		},
		{
			name:               "multi_endpoint_all_full",
			endpointLoadSetup:  map[string]int{"endpoint-a": 5, "endpoint-b": 6},
			candidateEndpoints: []string{"endpoint-a", "endpoint-b"},
			wantSaturation:     true,
		},
		{
			name:               "unknown_endpoint_assumed_empty",
			endpointLoadSetup:  nil,
			candidateEndpoints: []string{"endpoint-unknown"},
			wantSaturation:     false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			ctx := context.Background()
			detector := NewDetector(config)

			// Setup load.
			for endpointName, load := range tc.endpointLoadSetup {
				driveLoad(ctx, detector, endpointName, load)
			}

			// Build candidates.
			candidates := make([]backendmetrics.PodMetrics, 0, len(tc.candidateEndpoints))
			for _, name := range tc.candidateEndpoints {
				candidates = append(candidates, newFakePodMetric(name))
			}

			got := detector.IsSaturated(ctx, candidates)
			require.Equal(t, tc.wantSaturation, got, "IsSaturated result mismatch")
		})
	}
}

// TestDetector_Lifecycle verifies the full state transition cycle:
// New -> PreRequest (Inc) -> ResponseComplete (Dec) -> DeleteEndpoint (Reset).
func TestDetector_Lifecycle(t *testing.T) {
	t.Parallel()

	// MaxConcurrency 1 makes state changes immediate.
	detector := NewDetector(Config{MaxConcurrency: 1})
	ctx := context.Background()
	endpointName := "lifecycle-endpoint"
	candidates := []backendmetrics.PodMetrics{newFakePodMetric(endpointName)}

	// 1. Initially Empty
	require.False(t, detector.IsSaturated(ctx, candidates), "expected initially empty")

	// 2. Increment (Saturated)
	detector.PreRequest(ctx, nil, makeSchedulingResult(endpointName))
	require.True(t, detector.IsSaturated(ctx, candidates), "expected saturated after 1 request")

	// 3. Decrement (Available)
	targetEndpoint := newStubSchedulingEndpoint(endpointName)
	detector.ResponseComplete(ctx, nil, nil, targetEndpoint.metadata)
	require.False(t, detector.IsSaturated(ctx, candidates), "expected available after completion")

	// 4. Increment again -> Delete -> Verify Reset
	detector.PreRequest(ctx, nil, makeSchedulingResult(endpointName))
	require.True(t, detector.IsSaturated(ctx, candidates), "re-saturation failed")

	detector.DeleteEndpoint(fullEndpointName(endpointName))

	// After deletion, the endpoint is "unknown" to the tracker, effectively count=0.
	require.False(t, detector.IsSaturated(ctx, candidates), "expected clean state after DeleteEndpoint")
}

// TestDetector_ConcurrencyStress performs a targeted race condition check.
// It verifies that atomic counters remain accurate under heavy contention.
func TestDetector_ConcurrencyStress(t *testing.T) {
	t.Parallel()

	// Config doesn't matter much here as we check internal state, but keep it consistent.
	detector := NewDetector(Config{MaxConcurrency: 10000})
	ctx := context.Background()
	endpointName := "stress-endpoint"
	fullID := fullEndpointName(endpointName)

	// 1. Pre-warm the tracker.
	// We must ensure the atomic counter exists in the map before starting the race.
	// Otherwise, early 'dec' calls might be ignored (safety feature) if they beat the first 'inc' calls, causing a
	// positive drift.
	warmUpRes := makeSchedulingResult(endpointName)
	warmUpEndpoint := newStubSchedulingEndpoint(endpointName)
	detector.PreRequest(ctx, nil, warmUpRes)                          // Creates entry, count=1
	detector.ResponseComplete(ctx, nil, nil, warmUpEndpoint.metadata) // Decrements, count=0

	const (
		numGoroutines = 50
		opsPerRoutine = 1000
	)

	var wg sync.WaitGroup
	wg.Add(numGoroutines * 2)

	// Launch increments.
	for range numGoroutines {
		go func() {
			defer wg.Done()
			res := makeSchedulingResult(endpointName)
			for range opsPerRoutine {
				detector.PreRequest(ctx, nil, res)
			}
		}()
	}

	// Launch decrements.
	for range numGoroutines {
		go func() {
			defer wg.Done()
			targetEndpoint := newStubSchedulingEndpoint(endpointName)
			for range opsPerRoutine {
				detector.ResponseComplete(ctx, nil, nil, targetEndpoint.metadata)
			}
		}()
	}

	wg.Wait()

	// Strict white-box check: Counter MUST be exactly 0.
	finalCount := detector.tracker.get(fullID)
	require.Equal(t, int64(0), finalCount, "atomic counter drift detected; expected 0")
}

// --- Test Helpers & Mocks ---

func driveLoad(ctx context.Context, detector *Detector, endpointName string, count int) {
	res := makeSchedulingResult(endpointName)
	for i := 0; i < count; i++ {
		detector.PreRequest(ctx, nil, res)
	}
}

func fullEndpointName(name string) string {
	return types.NamespacedName{Name: name, Namespace: "default"}.String()
}

// makeSchedulingResult creates a minimal result for PreRequest
func makeSchedulingResult(endpointName string) *schedulingtypes.SchedulingResult {
	return &schedulingtypes.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*schedulingtypes.ProfileRunResult{
			"default": {
				TargetEndpoints: []schedulingtypes.Endpoint{newStubSchedulingEndpoint(endpointName)},
			},
		},
	}
}

func newFakePodMetric(name string) *backendmetrics.FakePodMetrics {
	return &backendmetrics.FakePodMetrics{
		Metadata: &fwkdl.EndpointMetadata{NamespacedName: types.NamespacedName{Name: name, Namespace: "default"}},
	}
}

// stubSchedulingEndpoint mocks schedulingtypes.Endpoint for Filter.
// It embeds the interface to satisfy the compiler but only implements GetMetadata.
type stubSchedulingEndpoint struct {
	schedulingtypes.Endpoint
	metadata *fwkdl.EndpointMetadata
}

func newStubSchedulingEndpoint(name string) *stubSchedulingEndpoint {
	return &stubSchedulingEndpoint{
		metadata: &fwkdl.EndpointMetadata{NamespacedName: types.NamespacedName{Name: name, Namespace: "default"}},
	}
}

func (f *stubSchedulingEndpoint) GetMetadata() *fwkdl.EndpointMetadata { return f.metadata }
