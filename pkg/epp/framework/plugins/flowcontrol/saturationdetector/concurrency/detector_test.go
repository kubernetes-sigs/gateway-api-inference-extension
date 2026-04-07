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

package concurrency

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrconcurrency "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/requestcontrol/requestdataproducer/inflightload"
)

// localRegistry is a thread-safe storage for simulated endpoint load.
type localRegistry struct {
	mu     sync.RWMutex
	counts map[string]*attrconcurrency.InFlightLoad
}

func newLocalRegistry() *localRegistry {
	return &localRegistry{
		counts: make(map[string]*attrconcurrency.InFlightLoad),
	}
}

func (r *localRegistry) get(id string) *attrconcurrency.InFlightLoad {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if load, ok := r.counts[id]; ok {
		return load.Clone().(*attrconcurrency.InFlightLoad)
	}
	return &attrconcurrency.InFlightLoad{}
}

func (r *localRegistry) update(id string, fn func(*attrconcurrency.InFlightLoad)) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.counts[id]; !ok {
		r.counts[id] = &attrconcurrency.InFlightLoad{}
	}
	fn(r.counts[id])
}

func (r *localRegistry) delete(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.counts, id)
}

// TestConcurrencyDetectorFactory validates the initialization of the concurrency detector plugin.
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
				tc.configJSON, fwkplugin.NewEppHandle(t.Context(), func() []types.NamespacedName { return nil }))
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
func TestDetector_Configuration(t *testing.T) {
	t.Parallel()

	tc := struct {
		config                 config
		effectiveMax           int64
		effectiveHeadroomBurst int64
	}{
		config: config{
			mode:           modeRequests,
			maxConcurrency: 50,
			headroom:       0.2, // 20% burst
		},
		effectiveMax:           50,
		effectiveHeadroomBurst: 60, // 50 * 1.2 = 60
	}

	t.Run("verify max concurrency saturation gradient", func(t *testing.T) {
		t.Parallel()
		reg := newLocalRegistry()
		ctx := t.Context()
		detector := newDetector("test-detector", tc.config, logr.Discard())
		endpointName := "test-endpoint"

		driveLoad(ctx, reg, detector, endpointName, int(tc.effectiveMax-1))
		expectedSat := float64(tc.effectiveMax-1) / float64(tc.effectiveMax)
		actualSat := detector.Saturation(ctx, []datalayer.Endpoint{newFakeEndpoint(reg, endpointName)})
		require.InDelta(t, expectedSat, actualSat, 1e-6, "Saturation must linearly reflect partial load")

		driveLoad(ctx, reg, detector, endpointName, 1)
		actualSat = detector.Saturation(ctx, []datalayer.Endpoint{newFakeEndpoint(reg, endpointName)})
		require.InDelta(t, 1.0, actualSat, 1e-6, "Saturation must cap at 1.0 at maxConcurrency limit")
	})

	t.Run("verify headroom filter limits", func(t *testing.T) {
		t.Parallel()
		reg := newLocalRegistry()
		ctx := t.Context()
		detector := newDetector("test-detector", tc.config, logr.Discard())
		endpointName := "test-endpoint"

		driveLoad(ctx, reg, detector, endpointName, int(tc.effectiveHeadroomBurst-1))
		kept := detector.Filter(ctx, nil, nil, []schedulingtypes.Endpoint{newStubSchedulingEndpoint(reg, endpointName)})
		require.Len(t, kept, 1, "Endpoint should be retained when operating below burst capacity")

		driveLoad(ctx, reg, detector, endpointName, 1)

		t.Run("fallback to clean endpoint", func(t *testing.T) {
			cleanEndpoint := "clean-endpoint"
			kept = detector.Filter(ctx, nil, nil, []schedulingtypes.Endpoint{
				newStubSchedulingEndpoint(reg, endpointName),
				newStubSchedulingEndpoint(reg, cleanEndpoint),
			})
			require.Len(t, kept, 1, "Filter should drop the overloaded endpoint")
			require.Equal(t, cleanEndpoint, kept[0].GetMetadata().NamespacedName.Name)
		})
	})
}

// TestDetector_TypedName verifies that the runtime type identification of the plugin is populated
func TestDetector_TypedName(t *testing.T) {
	t.Parallel()
	plugin, err := ConcurrencyDetectorFactory("test-plugin", []byte(`{}`), fwkplugin.NewEppHandle(
		t.Context(), func() []types.NamespacedName { return nil }))
	require.NoError(t, err, "Plugin initialization should succeed")
	require.Equal(t, "test-plugin", plugin.TypedName().Name)
	require.Equal(t, "concurrency-detector", plugin.TypedName().Type)
}

// TestDetector_Saturation evaluates the quantitative scaling of the saturation output.
func TestDetector_Saturation(t *testing.T) {
	t.Parallel()

	const maxConcurrency = 10
	config := config{mode: modeRequests, maxConcurrency: maxConcurrency}

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
			reg := newLocalRegistry()
			ctx := context.Background()
			detector := newDetector("test-detector", config, logr.Discard())

			for endpointName, load := range tc.endpointLoadSetup {
				driveLoad(ctx, reg, detector, endpointName, load)
			}

			candidates := make([]datalayer.Endpoint, 0, len(tc.candidateEndpoints))
			for _, name := range tc.candidateEndpoints {
				candidates = append(candidates, newFakeEndpoint(reg, name))
			}

			got := detector.Saturation(ctx, candidates)
			require.InDelta(t, tc.wantSaturation, got, 1e-6, "Saturation result mismatch")
		})
	}
}

// TestDetector_Lifecycle verifies the full state transition cycle.
func TestDetector_Lifecycle(t *testing.T) {
	t.Parallel()

	reg := newLocalRegistry()
	detector := newDetector("test", config{maxConcurrency: 1}, logr.Discard())
	ctx := context.Background()
	endpointName := "lifecycle-endpoint"
	candidates := []datalayer.Endpoint{newFakeEndpoint(reg, endpointName)}

	// 1. Initially Empty
	require.InDelta(t, 0.0, detector.Saturation(ctx, candidates), 1e-6, "expected initially 0.0")

	// 2. Increment (Saturated)
	simulatePreRequest(ctx, reg, nil, makeSchedulingResult(reg, endpointName))
	require.InDelta(t, 1.0, detector.Saturation(ctx, candidates), 1e-6, "expected 1.0 after 1 request")

	// 3. Decrement (Available)
	targetEndpoint := newStubSchedulingEndpoint(reg, endpointName)
	simulateResponseBody(ctx, reg, nil, &requestcontrol.Response{EndOfStream: true}, targetEndpoint.metadata)
	require.InDelta(t, 0.0, detector.Saturation(ctx, candidates), 1e-6, "expected 0.0 after completion")

	// 4. Increment again -> Delete -> Verify Reset
	simulatePreRequest(ctx, reg, nil, makeSchedulingResult(reg, endpointName))
	require.InDelta(t, 1.0, detector.Saturation(ctx, candidates), 1e-6, "re-saturation failed")

	simulateDeleteEndpoint(reg, fullEndpointName(endpointName))
	require.InDelta(t, 0.0, detector.Saturation(ctx, candidates), 1e-6, "expected clean state after DeleteEndpoint")
}

// TestDetector_TokenSaturation verifies saturation calculation in token mode.
func TestDetector_TokenSaturation(t *testing.T) {
	t.Parallel()

	const maxTokenConcurrency = 100
	config := config{
		mode:                modeTokens,
		maxTokenConcurrency: maxTokenConcurrency,
	}

	tests := []struct {
		name               string
		requests           []*schedulingtypes.InferenceRequest
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
			requests: []*schedulingtypes.InferenceRequest{
				makeTokenRequest("r1", "1234"), // 3 tokens with default estimator
			},
			candidateEndpoints: []string{"endpoint-a"},
			wantSaturation:     0.03, // 3/100
		},
		{
			name: "single_endpoint_half_full",
			requests: func() []*schedulingtypes.InferenceRequest {
				// "1234567890123456" (16 chars) = 10 tokens. 5 requests = 50 tokens.
				prompt := "1234567890123456"
				reqs := make([]*schedulingtypes.InferenceRequest, 0, 5)
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
			requests: func() []*schedulingtypes.InferenceRequest {
				// 10 tokens per request * 10 requests = 100 tokens.
				prompt := "1234567890123456"
				reqs := make([]*schedulingtypes.InferenceRequest, 0, 10)
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
			requests: func() []*schedulingtypes.InferenceRequest {
				// endpoint-a: 50 tokens, endpoint-b: 0 (driveTokenLoad targets endpoint-a only)
				prompt := "1234567890123456"
				reqs := make([]*schedulingtypes.InferenceRequest, 0, 5)
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
			reg := newLocalRegistry()
			ctx := context.Background()
			detector := newDetector("test-detector", config, logr.Discard())

			driveTokenLoad(ctx, reg, detector, "endpoint-a", tc.requests)

			candidates := make([]datalayer.Endpoint, 0, len(tc.candidateEndpoints))
			for _, name := range tc.candidateEndpoints {
				candidates = append(candidates, newFakeEndpoint(reg, name))
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
	reg := newLocalRegistry()
	detector := newDetector("test-detector", config, logr.Discard())
	endpointName := "token-filter-endpoint"
	endpoints := []schedulingtypes.Endpoint{newStubSchedulingEndpoint(reg, endpointName)}

	// Drive 110 tokens (just below 120 burst limit) -> endpoint should pass filter
	// "1234567890123456" = 10 tokens. 11 requests = 110 tokens.
	prompt := "1234567890123456"
	reqs := make([]*schedulingtypes.InferenceRequest, 0, 11)
	for i := range 11 {
		reqs = append(reqs, makeTokenRequest(fmt.Sprintf("r%d", i+1), prompt))
	}
	driveTokenLoad(ctx, reg, detector, endpointName, reqs)

	kept := detector.Filter(ctx, nil, nil, endpoints)
	require.Len(t, kept, 1, "endpoint should pass filter below burst limit")

	// Add one more request to reach 120 tokens -> filtered out
	driveTokenLoad(ctx, reg, detector, endpointName, []*schedulingtypes.InferenceRequest{
		makeTokenRequest("r12", prompt),
	})
	kept = detector.Filter(ctx, nil, nil, endpoints)
	require.Len(t, kept, 0, "endpoint should be filtered at burst limit")
}

// TestDetector_TokenLifecycle verifies token accounting.
func TestDetector_TokenLifecycle(t *testing.T) {
	t.Parallel()

	reg := newLocalRegistry()
	config := config{mode: modeTokens, maxTokenConcurrency: 100}
	ctx := context.Background()
	detector := newDetector("test-detector", config, logr.Discard())
	endpointName := "token-lifecycle-endpoint"
	candidates := []datalayer.Endpoint{newFakeEndpoint(reg, endpointName)}
	targetEndpoint := newStubSchedulingEndpoint(reg, endpointName)

	req1 := makeTokenRequest("req1", "1234567890123456") // 10 tokens
	simulatePreRequest(ctx, reg, req1, makeSchedulingResult(reg, endpointName))
	require.InDelta(t, 0.1, detector.Saturation(ctx, candidates), 1e-6)

	eos := &requestcontrol.Response{EndOfStream: true}
	simulateResponseBody(ctx, reg, req1, eos, targetEndpoint.metadata)
	require.InDelta(t, 0.0, detector.Saturation(ctx, candidates), 1e-6)
}

// TestDetector_TokenDeleteEndpoint verifies clearing tokens.
func TestDetector_TokenDeleteEndpoint(t *testing.T) {
	t.Parallel()

	reg := newLocalRegistry()
	config := config{mode: modeTokens, maxTokenConcurrency: 100}
	ctx := context.Background()
	detector := newDetector("test-detector", config, logr.Discard())
	endpointName := "token-delete-endpoint"
	candidates := []datalayer.Endpoint{newFakeEndpoint(reg, endpointName)}

	req := makeTokenRequest("req1", "1234567890123456")
	simulatePreRequest(ctx, reg, req, makeSchedulingResult(reg, endpointName))
	require.InDelta(t, 0.1, detector.Saturation(ctx, candidates), 1e-6)

	simulateDeleteEndpoint(reg, fullEndpointName(endpointName))
	require.InDelta(t, 0.0, detector.Saturation(ctx, candidates), 1e-6, "expected clean state after DeleteEndpoint")
}

// TestDetector_ConcurrencyStress performs race condition check.
func TestDetector_ConcurrencyStress(t *testing.T) {
	t.Parallel()

	reg := newLocalRegistry()
	ctx := context.Background()
	endpointName := "stress-endpoint"
	fullID := fullEndpointName(endpointName)

	warmUpRes := makeSchedulingResult(reg, endpointName)
	warmUpEndpoint := newStubSchedulingEndpoint(reg, endpointName)
	simulatePreRequest(ctx, reg, nil, warmUpRes)
	simulateResponseBody(ctx, reg, nil, &requestcontrol.Response{EndOfStream: true}, warmUpEndpoint.metadata)

	const numGoroutines = 50
	const opsPerRoutine = 1000

	var wg sync.WaitGroup
	wg.Add(numGoroutines * 2)

	for range numGoroutines {
		go func() {
			defer wg.Done()
			res := makeSchedulingResult(reg, endpointName)
			for range opsPerRoutine {
				simulatePreRequest(ctx, reg, nil, res)
			}
		}()
	}

	for range numGoroutines {
		go func() {
			defer wg.Done()
			targetEndpoint := newStubSchedulingEndpoint(reg, endpointName)
			for range opsPerRoutine {
				simulateResponseBody(ctx, reg, nil, &requestcontrol.Response{EndOfStream: true}, targetEndpoint.metadata)
			}
		}()
	}

	wg.Wait()
	require.Equal(t, int64(0), reg.get(fullID).Requests)
}

// --- Test Helpers & Mocks ---

func simulatePreRequest(_ context.Context, reg *localRegistry, req *schedulingtypes.InferenceRequest, result *schedulingtypes.SchedulingResult) {
	endpointName := result.ProfileResults[result.PrimaryProfileName].TargetEndpoints[0].GetMetadata().NamespacedName.Name
	id := fullEndpointName(endpointName)
	reg.update(id, func(load *attrconcurrency.InFlightLoad) {
		load.Requests++
		if req != nil {
			load.Tokens += inflightload.NewSimpleTokenEstimator().Estimate(req)
		}
	})
}

func simulateResponseBody(_ context.Context, reg *localRegistry, req *schedulingtypes.InferenceRequest, resp *requestcontrol.Response, metadata *datalayer.EndpointMetadata) {
	if metadata == nil || resp == nil || !resp.EndOfStream {
		return
	}
	id := metadata.NamespacedName.String()
	reg.update(id, func(load *attrconcurrency.InFlightLoad) {
		load.Requests--
		if req != nil {
			load.Tokens -= inflightload.NewSimpleTokenEstimator().Estimate(req)
		}
	})
}

func simulateDeleteEndpoint(reg *localRegistry, id string) {
	reg.delete(id)
}

func driveLoad(_ context.Context, reg *localRegistry, _ *detector, endpointName string, count int) {
	id := fullEndpointName(endpointName)
	reg.update(id, func(load *attrconcurrency.InFlightLoad) {
		load.Requests += int64(count)
	})
}

func driveTokenLoad(_ context.Context, reg *localRegistry, _ *detector, endpointName string, requests []*schedulingtypes.InferenceRequest) {
	id := fullEndpointName(endpointName)
	var total int64
	estimator := inflightload.NewSimpleTokenEstimator()
	for _, r := range requests {
		total += estimator.Estimate(r)
	}
	reg.update(id, func(load *attrconcurrency.InFlightLoad) {
		load.Tokens += total
		load.Requests += int64(len(requests))
	})
}

func fullEndpointName(name string) string {
	return types.NamespacedName{Name: name, Namespace: "default"}.String()
}

func makeSchedulingResult(reg *localRegistry, endpointName string) *schedulingtypes.SchedulingResult {
	return &schedulingtypes.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*schedulingtypes.ProfileRunResult{
			"default": {
				TargetEndpoints: []schedulingtypes.Endpoint{newStubSchedulingEndpoint(reg, endpointName)},
			},
		},
	}
}

// liveEndpoint implements datalayer.Endpoint dynamically.
type liveEndpoint struct {
	metadata *datalayer.EndpointMetadata
	reg      *localRegistry
	id       string
}

func (e *liveEndpoint) GetMetadata() *datalayer.EndpointMetadata     { return e.metadata }
func (e *liveEndpoint) UpdateMetadata(m *datalayer.EndpointMetadata) { e.metadata = m }
func (e *liveEndpoint) GetAttributes() datalayer.AttributeMap        { return e }
func (e *liveEndpoint) GetMetrics() *datalayer.Metrics               { return nil }
func (e *liveEndpoint) UpdateMetrics(*datalayer.Metrics)             {}
func (e *liveEndpoint) String() string                               { return e.id }

// liveEndpoint also implements AttributeMap.
func (e *liveEndpoint) Get(key string) (datalayer.Cloneable, bool) {
	if key == attrconcurrency.InFlightLoadKey {
		return e.reg.get(e.id), true
	}
	return nil, false
}
func (e *liveEndpoint) Put(string, datalayer.Cloneable) {}
func (e *liveEndpoint) Keys() []string                  { return []string{attrconcurrency.InFlightLoadKey} }
func (e *liveEndpoint) Clone() datalayer.AttributeMap   { return e }

func newFakeEndpoint(reg *localRegistry, name string) datalayer.Endpoint {
	id := fullEndpointName(name)
	return &liveEndpoint{
		metadata: &datalayer.EndpointMetadata{NamespacedName: types.NamespacedName{Name: name, Namespace: "default"}},
		reg:      reg,
		id:       id,
	}
}

// liveSchedulingEndpoint is a "live" mock for scheduling.Endpoint.
type liveSchedulingEndpoint struct {
	schedulingtypes.Endpoint
	metadata *datalayer.EndpointMetadata
	reg      *localRegistry
	id       string
}

func newStubSchedulingEndpoint(reg *localRegistry, name string) *liveSchedulingEndpoint {
	return &liveSchedulingEndpoint{
		metadata: &datalayer.EndpointMetadata{NamespacedName: types.NamespacedName{Name: name, Namespace: "default"}},
		reg:      reg,
		id:       fullEndpointName(name),
	}
}

func (f *liveSchedulingEndpoint) GetMetadata() *datalayer.EndpointMetadata { return f.metadata }
func (f *liveSchedulingEndpoint) Get(key string) (datalayer.Cloneable, bool) {
	if key == attrconcurrency.InFlightLoadKey {
		return f.reg.get(f.id), true
	}
	return nil, false
}
func (f *liveSchedulingEndpoint) Put(string, datalayer.Cloneable) {}
func (f *liveSchedulingEndpoint) Keys() []string                  { return []string{attrconcurrency.InFlightLoadKey} }
func (f *liveSchedulingEndpoint) String() string                  { return f.id }
func (f *liveSchedulingEndpoint) Clone() datalayer.AttributeMap   { return f }

func makeTokenRequest(requestID, prompt string) *schedulingtypes.InferenceRequest {
	return &schedulingtypes.InferenceRequest{
		RequestId: requestID,
		Body: &schedulingtypes.InferenceRequestBody{
			Completions: &schedulingtypes.CompletionsRequest{Prompt: schedulingtypes.Prompt{Raw: prompt}},
		},
	}
}
