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
	"fmt"
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

			// 1. Verify MaxConcurrency via Saturation

			// A. Drive load to just below the limit (Max - 1)
			// Expected Saturation = (Max - 1) / Max
			driveLoad(ctx, detector, endpointName, int(tc.effectiveMax-1))
			expectedSat := float64(tc.effectiveMax-1) / float64(tc.effectiveMax)
			actualSat := detector.Saturation(ctx, []backendmetrics.PodMetrics{newFakePodMetric(endpointName)})
			require.InDelta(t, expectedSat, actualSat, 1e-6, "expected saturation gradient to reflect partial load")

			// B. Increment to exactly the limit (Max)
			// Expected Saturation = 1.0
			driveLoad(ctx, detector, endpointName, 1)
			actualSat = detector.Saturation(ctx, []backendmetrics.PodMetrics{newFakePodMetric(endpointName)})
			require.InDelta(t, 1.0, actualSat, 1e-6, `expected 100% saturation at MaxConcurrency`)

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

// TestDetector_Saturation verifies the global saturation gradient logic.
// It ensures saturation is reported as an aggregate ratio of TotalInflight / TotalCapacity.
func TestDetector_Saturation(t *testing.T) {
	t.Parallel()

	const maxConcurrency = 10
	config := Config{MaxConcurrency: maxConcurrency}

	tests := []struct {
		name               string
		endpointLoadSetup  map[string]int // Map of EndpointName -> Request Count
		candidateEndpoints []string
		wantSaturation     float64
	}{
		{
			name:               "empty_candidate_list_fail_closed",
			endpointLoadSetup:  nil,
			candidateEndpoints: []string{},
			wantSaturation:     1.0,
		},
		{
			name:               "single_endpoint_empty",
			endpointLoadSetup:  map[string]int{"endpoint-a": 0},
			candidateEndpoints: []string{"endpoint-a"},
			wantSaturation:     0.0,
		},
		{
			name:               "single_endpoint_half_full",
			endpointLoadSetup:  map[string]int{"endpoint-a": 5},
			candidateEndpoints: []string{"endpoint-a"},
			wantSaturation:     0.5,
		},
		{
			name:               "single_endpoint_full",
			endpointLoadSetup:  map[string]int{"endpoint-a": 10},
			candidateEndpoints: []string{"endpoint-a"},
			wantSaturation:     1.0,
		},
		{
			name:               "multi_endpoint_mixed_load",
			endpointLoadSetup:  map[string]int{"endpoint-a": 10, "endpoint-b": 0},
			candidateEndpoints: []string{"endpoint-a", "endpoint-b"},
			wantSaturation:     0.5,
		},
		{
			name:               "multi_endpoint_overloaded",
			endpointLoadSetup:  map[string]int{"endpoint-a": 15, "endpoint-b": 5},
			candidateEndpoints: []string{"endpoint-a", "endpoint-b"},
			wantSaturation:     1.0,
		},
		{
			name:               "multi_endpoint_very_overloaded",
			endpointLoadSetup:  map[string]int{"endpoint-a": 15, "endpoint-b": 15},
			candidateEndpoints: []string{"endpoint-a", "endpoint-b"},
			wantSaturation:     1.5,
		},
		{
			name:               "unknown_endpoint_assumed_empty",
			endpointLoadSetup:  nil,
			candidateEndpoints: []string{"endpoint-unknown"},
			wantSaturation:     0.0,
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

			got := detector.Saturation(ctx, candidates)
			require.InDelta(t, tc.wantSaturation, got, 1e-6, "Saturation result mismatch")
		})
	}
}

// TestDetector_Lifecycle verifies the full state transition cycle:
// New -> PreRequest (Inc) -> ResponseComplete (Dec) -> DeleteEndpoint (Reset).
func TestDetector_Lifecycle(t *testing.T) {
	t.Parallel()

	// MaxConcurrency 1 makes math simple.
	detector := NewDetector(Config{MaxConcurrency: 1})
	ctx := context.Background()
	endpointName := "lifecycle-endpoint"
	candidates := []backendmetrics.PodMetrics{newFakePodMetric(endpointName)}

	// 1. Initially Empty
	require.InDelta(t, 0.0, detector.Saturation(ctx, candidates), 1e-6, "expected initially 0.0")

	// 2. Increment (Saturated)
	detector.PreRequest(ctx, nil, makeSchedulingResult(endpointName))
	require.InDelta(t, 1.0, detector.Saturation(ctx, candidates), 1e-6, "expected 1.0 after 1 request")

	// 3. Decrement (Available)
	targetEndpoint := newStubSchedulingEndpoint(endpointName)
	detector.ResponseComplete(ctx, nil, nil, targetEndpoint.metadata)
	require.InDelta(t, 0.0, detector.Saturation(ctx, candidates), 1e-6, "expected 0.0 after completion")

	// 4. Increment again -> Delete -> Verify Reset
	detector.PreRequest(ctx, nil, makeSchedulingResult(endpointName))
	require.InDelta(t, 1.0, detector.Saturation(ctx, candidates), 1e-6, "re-saturation failed")

	detector.DeleteEndpoint(fullEndpointName(endpointName))

	// After deletion, the endpoint is "unknown" to the tracker, effectively count=0.
	require.InDelta(t, 0.0, detector.Saturation(ctx, candidates), 1e-6, "expected clean state after DeleteEndpoint")
}

// TestDetector_TokenSaturation verifies saturation calculation in token mode.
func TestDetector_TokenSaturation(t *testing.T) {
	t.Parallel()

	// MaxTokenConcurrency set to 100 for simple testing
	const maxTokenConcurrency = 100
	config := Config{
		ConcurrencyMode:     ConcurrencyModeTokens,
		MaxTokenConcurrency: maxTokenConcurrency,
	}

	tests := []struct {
		name               string
		requests           []*schedulingtypes.LLMRequest
		candidateEndpoints []string
		wantSaturation     float64
	}{
		{
			name:               "empty_requests",
			requests:           nil,
			candidateEndpoints: []string{"endpoint-a"},
			wantSaturation:     0.0,
		},
		{
			name: "single_endpoint_partial_tokens",
			requests: []*schedulingtypes.LLMRequest{
				makeTokenRequest("r1", "1234"), // 3 tokens with default estimator
			},
			candidateEndpoints: []string{"endpoint-a"},
			wantSaturation:     0.03, // 3/100
		},
		{
			name: "single_endpoint_half_full",
			requests: func() []*schedulingtypes.LLMRequest {
				// "1234567890123456" (16 chars) = 10 tokens. 5 requests = 50 tokens.
				prompt := "1234567890123456"
				var reqs []*schedulingtypes.LLMRequest
				for i := range 5 {
					reqs = append(reqs, makeTokenRequest(fmt.Sprintf("r%d", i+1), prompt))
				}
				return reqs
			}(),
			candidateEndpoints: []string{"endpoint-a"},
			wantSaturation:     0.5,
		},
		{
			name: "single_endpoint_full",
			requests: func() []*schedulingtypes.LLMRequest {
				// 10 tokens per request * 10 requests = 100 tokens.
				prompt := "1234567890123456"
				var reqs []*schedulingtypes.LLMRequest
				for i := range 10 {
					reqs = append(reqs, makeTokenRequest(fmt.Sprintf("r%d", i+1), prompt))
				}
				return reqs
			}(),
			candidateEndpoints: []string{"endpoint-a"},
			wantSaturation:     1.0,
		},
		{
			name: "multiple_endpoints_mixed_token_load",
			requests: func() []*schedulingtypes.LLMRequest {
				// endpoint-a: 50 tokens, endpoint-b: 0 (driveTokenLoad targets endpoint-a only)
				prompt := "1234567890123456"
				var reqs []*schedulingtypes.LLMRequest
				for i := range 5 {
					reqs = append(reqs, makeTokenRequest(fmt.Sprintf("r%d", i+1), prompt))
				}
				return reqs
			}(),
			candidateEndpoints: []string{"endpoint-a", "endpoint-b"},
			wantSaturation:     0.25, // 50 tokens / (100+100) capacity
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			ctx := context.Background()
			detector := NewDetector(config)

			driveTokenLoad(ctx, detector, "endpoint-a", tc.requests)

			candidates := make([]backendmetrics.PodMetrics, 0, len(tc.candidateEndpoints))
			for _, name := range tc.candidateEndpoints {
				candidates = append(candidates, newFakePodMetric(name))
			}

			got := detector.Saturation(ctx, candidates)
			require.InDelta(t, tc.wantSaturation, got, 1e-6, "Token saturation mismatch")
		})
	}
}

// TestDetector_TokenFilter verifies Filter behavior in token mode.
func TestDetector_TokenFilter(t *testing.T) {
	t.Parallel()

	config := Config{
		ConcurrencyMode:     ConcurrencyModeTokens,
		MaxTokenConcurrency: 100,
		Headroom:            0.2, // Burst limit = 100 * 1.2 = 120 tokens
	}
	
	ctx := context.Background()
	detector := NewDetector(config)
	endpointName := "token-filter-endpoint"
	endpoints := []schedulingtypes.Endpoint{newStubSchedulingEndpoint(endpointName)}

	// Drive 110 tokens (just below 120 burst limit) -> endpoint should pass filter
	// "1234567890123456" = 10 tokens. 11 requests = 110 tokens.
	prompt := "1234567890123456"
	reqs := make([]*schedulingtypes.LLMRequest, 0, 11)
	for i := range 11 {
		reqs = append(reqs, makeTokenRequest(fmt.Sprintf("r%d", i+1), prompt))
	}
	driveTokenLoad(ctx, detector, endpointName, reqs)

	kept := detector.Filter(ctx, nil, nil, endpoints)
	require.Len(t, kept, 1, "endpoint should pass filter below burst limit")

	// Add one more request to reach 120 tokens -> filtered out
	driveTokenLoad(ctx, detector, endpointName, []*schedulingtypes.LLMRequest{
		makeTokenRequest("r12", prompt),
	})
	kept = detector.Filter(ctx, nil, nil, endpoints)
	require.Len(t, kept, 0, "endpoint should be filtered at burst limit")
}

// TestDetector_TokenLifecycle verifies PreRequest/ResponseComplete with token ledger.
func TestDetector_TokenLifecycle(t *testing.T) {
	t.Parallel()

	config := Config{
		ConcurrencyMode:     ConcurrencyModeTokens,
		MaxTokenConcurrency: 100,
	}
	ctx := context.Background()
	detector := NewDetector(config)
	endpointName := "token-lifecycle-endpoint"
	candidates := []backendmetrics.PodMetrics{newFakePodMetric(endpointName)}
	targetEndpoint := newStubSchedulingEndpoint(endpointName)

	// PreRequest adds tokens ("1234567890123456" = 10 tokens)
	req1 := makeTokenRequest("req1", "1234567890123456")
	detector.PreRequest(ctx, req1, makeSchedulingResult(endpointName))
	require.InDelta(t, 0.1, detector.Saturation(ctx, candidates), 1e-6, "saturation after 1 request (10 tokens)")

	// ResponseComplete removes tokens
	detector.ResponseComplete(ctx, req1, nil, targetEndpoint.metadata)
	require.InDelta(t, 0.0, detector.Saturation(ctx, candidates), 1e-6, "saturation after completion")

	// Multiple requests, complete some
	req2 := makeTokenRequest("req2", "1234")
	req3 := makeTokenRequest("req3", "1234")
	detector.PreRequest(ctx, req2, makeSchedulingResult(endpointName))
	detector.PreRequest(ctx, req3, makeSchedulingResult(endpointName))
	require.InDelta(t, 6.0/100.0, detector.Saturation(ctx, candidates), 1e-6, "6 tokens in flight")

	detector.ResponseComplete(ctx, req2, nil, targetEndpoint.metadata)
	require.InDelta(t, 3.0/100.0, detector.Saturation(ctx, candidates), 1e-6, "3 tokens after one completion")

	detector.ResponseComplete(ctx, req3, nil, targetEndpoint.metadata)
	require.InDelta(t, 0.0, detector.Saturation(ctx, candidates), 1e-6, "empty after all completions")
}

// TestDetector_TokenDeleteEndpoint verifies DeleteEndpoint, clears token ledger for the endpoint.
func TestDetector_TokenDeleteEndpoint(t *testing.T) {
	t.Parallel()

	config := Config{
		ConcurrencyMode:     ConcurrencyModeTokens,
		MaxTokenConcurrency: 100,
	}
	ctx := context.Background()
	detector := NewDetector(config)
	endpointName := "token-delete-endpoint"
	candidates := []backendmetrics.PodMetrics{newFakePodMetric(endpointName)}

	req := makeTokenRequest("req1", "1234567890123456")
	req.RequestId = "req1"
	detector.PreRequest(ctx, req, makeSchedulingResult(endpointName))
	require.InDelta(t, 0.1, detector.Saturation(ctx, candidates), 1e-6)

	detector.DeleteEndpoint(fullEndpointName(endpointName))
	require.InDelta(t, 0.0, detector.Saturation(ctx, candidates), 1e-6, "tokens cleared after DeleteEndpoint")
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

// makeTokenRequest creates an LLMRequest with a prompt.
func makeTokenRequest(requestID, prompt string) *schedulingtypes.LLMRequest {
	return &schedulingtypes.LLMRequest{
		RequestId: requestID,
		Body: &schedulingtypes.LLMRequestBody{
			Completions: &schedulingtypes.CompletionsRequest{Prompt: prompt},
		},
	}
}

// driveTokenLoad drives token based load by issuing PreRequest with the given requests.
func driveTokenLoad(ctx context.Context, detector *Detector, endpointName string, requests []*schedulingtypes.LLMRequest) {
	res := makeSchedulingResult(endpointName)
	for i, req := range requests {
		if req != nil && req.RequestId == "" {
			req.RequestId = fmt.Sprintf("req%d", i+1)
		}
		detector.PreRequest(ctx, req, res)
	}
}
