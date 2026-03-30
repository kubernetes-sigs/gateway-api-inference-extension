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

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"

	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/test/utils"
)

// TestConcurrencyDetectorFactory validates the initialization of the concurrency detector plugin.
// It ensures that valid configuration blocks successfully instantiate the plugin while malformed or
// semantically invalid configurations are rejected with descriptive errors.
func TestConcurrencyDetectorFactory(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		configJSON []byte
		wantError  bool
	}{
		{
			name:       "valid configuration",
			configJSON: []byte(`{"mode": "requests", "maxConcurrency": 50, "headroom": 0.2}`),
			wantError:  false,
		},
		{
			name:       "invalid schema",
			configJSON: []byte(`{"maxConcurrency": "invalid_type"}`),
			wantError:  true,
		},
		{
			name:       "empty config applies defaults",
			configJSON: []byte(`{}`),
			wantError:  false,
		},
		{
			name:       "invalid max concurrency",
			configJSON: []byte(`{"maxConcurrency": 0}`),
			wantError:  true,
		},
		{
			name:       "invalid max token concurrency",
			configJSON: []byte(`{"maxTokenConcurrency": 0}`),
			wantError:  true,
		},
		{
			name:       "invalid headroom",
			configJSON: []byte(`{"headroom": -0.5}`),
			wantError:  true,
		},
		{
			name:       "invalid mode",
			configJSON: []byte(`{"concurrencyMode": "magic"}`),
			wantError:  true,
		},
		{
			name:       "high headroom warning",
			configJSON: []byte(`{"headroom": 2.0}`),
			wantError:  false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			plugin, err := ConcurrencyDetectorFactory("test-concurrency-detector",
				tc.configJSON, utils.NewTestHandle(t.Context()))
			if tc.wantError {
				require.Error(t, err, "Expected initialization to fail on invalid configuration")
				require.Nil(t, plugin, "Plugin must be nil when initialization fails")
			} else {
				require.NoError(t, err, "Expected initialization to succeed with valid configuration")
				require.NotNil(t, plugin, "Plugin must not be nil on success")
			}
		})
	}
}

// TestDetector_Configuration evaluates the internal mechanics of the configuration parameters.
// It verifies that gradients are correctly mapped over limits and that fallback logic respects
// maximum capacities under synthetic traffic load.
func TestDetector_Configuration(t *testing.T) {
	t.Parallel()

	tc := struct {
		config                 config
		effectiveMax           int64
		effectiveHeadroomBurst int64
	}{
		config: config{
			maxConcurrency: 50,
			headroom:       0.2, // 20% burst
		},
		effectiveMax:           50,
		effectiveHeadroomBurst: 60, // 50 * 1.2 = 60
	}

	t.Run("verify max concurrency saturation gradient", func(t *testing.T) {
		t.Parallel()
		ctx := t.Context()
		detector := newDetector("test-detector", tc.config, logr.Discard())
		endpointName := "test-endpoint"

		// Drive load to just below the limit (Max - 1).
		// Expected Saturation = (Max - 1) / Max
		driveLoad(ctx, detector, endpointName, int(tc.effectiveMax-1))
		expectedSat := float64(tc.effectiveMax-1) / float64(tc.effectiveMax)
		actualSat := detector.Saturation(ctx, []datalayer.Endpoint{newFakeEndpoint(endpointName)})
		require.InDelta(t, expectedSat, actualSat, 1e-6, "Saturation must linearly reflect partial load")

		// Increment to exactly the limit (Max).
		// Expected Saturation = 1.0
		driveLoad(ctx, detector, endpointName, 1)
		actualSat = detector.Saturation(ctx, []datalayer.Endpoint{newFakeEndpoint(endpointName)})
		require.InDelta(t, 1.0, actualSat, 1e-6, "Saturation must cap at 1.0 at maxConcurrency limit")
	})

	t.Run("verify headroom filter limits", func(t *testing.T) {
		t.Parallel()
		ctx := t.Context()
		detector := newDetector("test-detector", tc.config, logr.Discard())
		endpointName := "test-endpoint"

		// Drive load to just below the burst limit (Limit - 1).
		driveLoad(ctx, detector, endpointName, int(tc.effectiveHeadroomBurst-1))
		kept := detector.Filter(ctx, nil, nil, []schedulingtypes.Endpoint{newStubSchedulingEndpoint(endpointName)})
		require.Len(t, kept, 1, "Endpoint should be retained when operating below burst capacity")

		// Reach the burst limit (Limit).
		driveLoad(ctx, detector, endpointName, 1)

		t.Run("fallback to clean endpoint", func(t *testing.T) {
			cleanEndpoint := "clean-endpoint"
			kept = detector.Filter(ctx, nil, nil, []schedulingtypes.Endpoint{
				newStubSchedulingEndpoint(endpointName),
				newStubSchedulingEndpoint(cleanEndpoint),
			})
			require.Len(t, kept, 1, "Filter should drop the overloaded endpoint")
			require.Equal(t, cleanEndpoint, kept[0].GetMetadata().GetNamespacedName().Name,
				"Filter should retain the clean fallback endpoint")
		})
	})
}

// TestDetector_TypedName verifies that the runtime type identification of the plugin is populated
// correctly during instantiation and accurately reflects the hardcoded concurrency-detector type.
func TestDetector_TypedName(t *testing.T) {
	t.Parallel()
	plugin, err := ConcurrencyDetectorFactory("test-plugin", []byte(`{}`), utils.NewTestHandle(t.Context()))
	require.NoError(t, err, "Plugin initialization should succeed")
	require.Equal(t, "test-plugin", plugin.TypedName().Name,
		"TypedName must match the name provided during initialization")
	require.Equal(t, "concurrency-detector", plugin.TypedName().Type,
		"TypedName.Type must be exactly 'concurrency-detector'")
}

// TestDetector_Saturation evaluates the quantitative scaling of the saturation output.
// It verifies that 0, partial, overloaded, and over-saturated loads return exact mathematical
// representations, protecting upper layers from incorrect metrics.
func TestDetector_Saturation(t *testing.T) {
	t.Parallel()

	const maxConcurrency = 10
	config := config{maxConcurrency: maxConcurrency}

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
			detector := newDetector("test-detector", config, logr.Discard())

			// Setup load.
			for endpointName, load := range tc.endpointLoadSetup {
				driveLoad(ctx, detector, endpointName, load)
			}

			// Build candidates.
			candidates := make([]datalayer.Endpoint, 0, len(tc.candidateEndpoints))
			for _, name := range tc.candidateEndpoints {
				candidates = append(candidates, newFakeEndpoint(name))
			}

			got := detector.Saturation(ctx, candidates)
			require.InDelta(t, tc.wantSaturation, got, 1e-6, "Saturation result mismatch")
		})
	}
}

// TestDetector_Lifecycle verifies the full state transition cycle:
// New -> PreRequest (Inc) -> ResponseBody EndOfStream (Dec) -> DeleteEndpoint (Reset).
func TestDetector_Lifecycle(t *testing.T) {
	t.Parallel()

	// MaxConcurrency 1 makes math simple.
	detector := newDetector("test", config{maxConcurrency: 1}, logr.Discard())
	ctx := context.Background()
	endpointName := "lifecycle-endpoint"
	candidates := []datalayer.Endpoint{newFakeEndpoint(endpointName)}

	// 1. Initially Empty
	require.InDelta(t, 0.0, detector.Saturation(ctx, candidates), 1e-6, "expected initially 0.0")

	// 2. Increment (Saturated)
	detector.PreRequest(ctx, nil, makeSchedulingResult(endpointName))
	require.InDelta(t, 1.0, detector.Saturation(ctx, candidates), 1e-6, "expected 1.0 after 1 request")

	// 3. Decrement (Available)
	targetEndpoint := newStubSchedulingEndpoint(endpointName)
	detector.ResponseBody(ctx, nil, &requestcontrol.Response{EndOfStream: true}, targetEndpoint.metadata)
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
	config := config{
		mode:                modeTokens,
		maxTokenConcurrency: maxTokenConcurrency,
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
				reqs := make([]*schedulingtypes.LLMRequest, 0, 5)
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
				reqs := make([]*schedulingtypes.LLMRequest, 0, 10)
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
				reqs := make([]*schedulingtypes.LLMRequest, 0, 5)
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
			detector := newDetector("test-detector", config, logr.Discard())

			driveTokenLoad(ctx, detector, "endpoint-a", tc.requests)

			candidates := make([]datalayer.Endpoint, 0, len(tc.candidateEndpoints))
			for _, name := range tc.candidateEndpoints {
				candidates = append(candidates, newFakeEndpoint(name))
			}

			got := detector.Saturation(ctx, candidates)
			require.InDelta(t, tc.wantSaturation, got, 1e-6, "Token saturation mismatch")
		})
	}
}

// TestDetector_TokenFilter verifies Filter behavior in token mode.
func TestDetector_TokenFilter(t *testing.T) {
	t.Parallel()

	config := config{
		mode:                modeTokens,
		maxTokenConcurrency: 100,
		headroom:            0.2, // Burst limit = 100 * 1.2 = 120 tokens
	}

	ctx := context.Background()
	detector := newDetector("test-detector", config, logr.Discard())
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

// TestDetector_TokenLifecycle verifies PreRequest/ResponseBody (EndOfStream) token accounting.
func TestDetector_TokenLifecycle(t *testing.T) {
	t.Parallel()

	config := config{
		mode:                modeTokens,
		maxTokenConcurrency: 100,
	}
	ctx := context.Background()
	detector := newDetector("test-detector", config, logr.Discard())
	endpointName := "token-lifecycle-endpoint"
	candidates := []datalayer.Endpoint{newFakeEndpoint(endpointName)}
	targetEndpoint := newStubSchedulingEndpoint(endpointName)

	// PreRequest adds tokens ("1234567890123456" = 10 tokens)
	req1 := makeTokenRequest("req1", "1234567890123456")
	detector.PreRequest(ctx, req1, makeSchedulingResult(endpointName))
	require.InDelta(t, 0.1, detector.Saturation(ctx, candidates), 1e-6, "saturation after 1 request (10 tokens)")

	eos := &requestcontrol.Response{EndOfStream: true}
	detector.ResponseBody(ctx, req1, eos, targetEndpoint.metadata)
	require.InDelta(t, 0.0, detector.Saturation(ctx, candidates), 1e-6, "saturation after completion")

	// Multiple requests, complete some
	req2 := makeTokenRequest("req2", "1234")
	req3 := makeTokenRequest("req3", "1234")
	detector.PreRequest(ctx, req2, makeSchedulingResult(endpointName))
	detector.PreRequest(ctx, req3, makeSchedulingResult(endpointName))
	require.InDelta(t, 6.0/100.0, detector.Saturation(ctx, candidates), 1e-6, "6 tokens in flight")

	detector.ResponseBody(ctx, req2, eos, targetEndpoint.metadata)
	require.InDelta(t, 3.0/100.0, detector.Saturation(ctx, candidates), 1e-6, "3 tokens after one completion")

	detector.ResponseBody(ctx, req3, eos, targetEndpoint.metadata)
	require.InDelta(t, 0.0, detector.Saturation(ctx, candidates), 1e-6, "empty after all completions")
}

// TestDetector_TokenDeleteEndpoint verifies DeleteEndpoint, clears token ledger for the endpoint.
func TestDetector_TokenDeleteEndpoint(t *testing.T) {
	t.Parallel()

	config := config{
		mode:                modeTokens,
		maxTokenConcurrency: 100,
	}
	ctx := context.Background()
	detector := newDetector("test-detector", config, logr.Discard())
	endpointName := "token-delete-endpoint"
	candidates := []datalayer.Endpoint{newFakeEndpoint(endpointName)}

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
	detector := newDetector("test-detector", config{maxConcurrency: 10000}, logr.Discard())
	ctx := context.Background()
	endpointName := "stress-endpoint"
	fullID := fullEndpointName(endpointName)

	// 1. Pre-warm the tracker.
	// We must ensure the atomic counter exists in the map before starting the race.
	// Otherwise, early 'dec' calls might be ignored (safety feature) if they beat the first 'inc' calls, causing a
	// positive drift.
	warmUpRes := makeSchedulingResult(endpointName)
	warmUpEndpoint := newStubSchedulingEndpoint(endpointName)
	detector.PreRequest(ctx, nil, warmUpRes)                                                              // Creates entry, count=1
	detector.ResponseBody(ctx, nil, &requestcontrol.Response{EndOfStream: true}, warmUpEndpoint.metadata) // Decrements, count=0

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
				detector.ResponseBody(ctx, nil, &requestcontrol.Response{EndOfStream: true}, targetEndpoint.metadata)
			}
		}()
	}

	wg.Wait()

	// Strict white-box check: Counter MUST be exactly 0.
	finalCount := detector.requestTracker.get(fullID)
	require.Equal(t, int64(0), finalCount, "atomic counter drift detected; expected 0")
}

// --- Test Helpers & Mocks ---

func driveLoad(ctx context.Context, detector *detector, endpointName string, count int) {
	res := makeSchedulingResult(endpointName)
	for range count {
		detector.PreRequest(ctx, nil, res)
	}
}

func fullEndpointName(name string) string {
	key := plugin.NewEndPointKey(name, "default", 8000)
	return (&key).String()
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

func newFakeEndpoint(name string) *backendmetrics.FakePodMetrics {
	return &backendmetrics.FakePodMetrics{
		Metadata: &datalayer.EndpointMetadata{Key: plugin.NewEndPointKey(name, "default", 8000)},
	}
}

// stubSchedulingEndpoint mocks schedulingtypes.Endpoint for Filter.
// It embeds the interface to satisfy the compiler but only implements GetMetadata.
type stubSchedulingEndpoint struct {
	schedulingtypes.Endpoint
	metadata *datalayer.EndpointMetadata
}

func newStubSchedulingEndpoint(name string) *stubSchedulingEndpoint {
	return &stubSchedulingEndpoint{
		metadata: &datalayer.EndpointMetadata{Key: plugin.NewEndPointKey(name, "default", 8000)},
	}
}

func (f *stubSchedulingEndpoint) GetMetadata() *datalayer.EndpointMetadata { return f.metadata }

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
func driveTokenLoad(ctx context.Context, detector *detector, endpointName string, requests []*schedulingtypes.LLMRequest) {
	res := makeSchedulingResult(endpointName)
	for i, req := range requests {
		if req != nil && req.RequestId == "" {
			req.RequestId = fmt.Sprintf("req%d", i+1)
		}
		detector.PreRequest(ctx, req, res)
	}
}
