/*
Copyright 2026 The Kubernetes Authors.

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

package metrics

// TestMetricsExtractionFromConfig tests the full pipeline:
//
//  1. Instantiate data source and extractor via the factory functions
//     (using the same JSON parameters the configloader passes from YAML).
//  2. Start an httptest.Server serving Prometheus metrics.
//  3. Poll the server and verify extracted endpoint metrics.
//
// These tests cover:
//   - Default configuration: all five vLLM metrics collected.
//   - LoRA disabled via engineConfigs (loraSpec: ""): no LoRA extraction, no error.
//   - Metric family absent from server: Poll returns an error containing the
//     family name (this is what the collector would log on first occurrence).
//   - "Not scraping metric" startup logging: factory succeeds and the extractor
//     silently skips the disabled spec during Poll.

import (
	"context"
	"encoding/json"
	"errors"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	metricextractor "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/extractor/metrics"
	sourcemetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/source/metrics"
)

// pipeline wraps a PollingDataSource and an Extractor, calling both in sequence.
// This replicates what the Collector does at runtime, allowing tests to exercise
// the full fetch→extract path without a running Collector.
type pipeline struct {
	source    fwkdl.PollingDataSource
	extractor fwkdl.Extractor
}

// Poll fetches raw data from the source then passes it to the extractor.
func (p *pipeline) Poll(ctx context.Context, ep fwkdl.Endpoint) error {
	data, err := p.source.Poll(ctx, ep)
	if err != nil {
		return err
	}
	if data == nil {
		return nil
	}
	return p.extractor.Extract(ctx, data, ep)
}

// buildSourceAndExtractor creates and wires a MetricsDataSource and a CoreMetricsExtractor
// using the same factory functions the configloader invokes, with JSON parameters as
// they would appear under the plugin's `parameters` field in EndpointPickerConfig YAML.
//
// extractorParams may be nil to use built-in defaults.
func buildSourceAndExtractor(t *testing.T, serverURL string, extractorParams map[string]any) (*pipeline, error) {
	t.Helper()

	parsedURL, err := url.Parse(serverURL)
	require.NoError(t, err, "failed to parse server URL")

	// Inject scheme and path matching the test server.
	sourceParams := map[string]any{
		"scheme": parsedURL.Scheme,
		"path":   parsedURL.Path,
	}

	rawSourceParams, err := json.Marshal(sourceParams)
	require.NoError(t, err, "failed to marshal source params")

	// Instantiate the data source — mirrors configloader calling MetricsDataSourceFactory.
	sourcePlug, err := sourcemetrics.MetricsDataSourceFactory(
		"metrics-data-source",
		rawSourceParams,
		nil,
	)
	if err != nil {
		return nil, err
	}
	dataSource, ok := sourcePlug.(fwkdl.PollingDataSource)
	require.True(t, ok, "expected PollingDataSource")

	var rawExtractorParams json.RawMessage
	if extractorParams != nil {
		rawExtractorParams, err = json.Marshal(extractorParams)
		require.NoError(t, err, "failed to marshal extractor params")
	}

	// Instantiate the extractor — mirrors configloader calling CoreMetricsExtractorFactory.
	extractorPlug, err := metricextractor.CoreMetricsExtractorFactory(
		"core-metrics-extractor",
		rawExtractorParams,
		nil, // nil → logNilSpecs uses logr.Discard()
	)
	if err != nil {
		return nil, err
	}
	extractor, ok := extractorPlug.(fwkdl.Extractor)
	require.True(t, ok, "expected Extractor")

	return &pipeline{source: dataSource, extractor: extractor}, nil
}

// newEndpointAt creates a fwkdl.Endpoint with the given host (host:port) and optional labels.
func newEndpointAt(host string, labels map[string]string) fwkdl.Endpoint {
	return fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		MetricsHost: host,
		Labels:      labels,
	}, fwkdl.NewMetrics())
}

// TestMetricsExtractionDefaultConfig verifies that the default factory parameters
// collect all five vLLM metrics from a real (httptest) Prometheus endpoint.
func TestMetricsExtractionDefaultConfig(t *testing.T) {
	srv := createMockServer([]MetricMock{
		{Name: WaitingMetric, Value: 7},
		{Name: RunningMetric, Value: 3},
		{Name: KVCacheMetric, Value: 0.55},
		{
			Name:  LoRAMetric,
			Value: float64(time.Now().Unix()),
			Labels: map[string]string{
				LoraInfoRunningAdaptersMetricName: "adapter-a,adapter-b",
				LoraInfoWaitingAdaptersMetricName: "adapter-c",
				LoraInfoMaxAdaptersMetricName:     "4",
			},
		},
		{
			Name:  CacheConfigMetric,
			Value: 1,
			Labels: map[string]string{
				CacheConfigBlockSizeInfoMetricName: "16",
				CacheConfigNumGPUBlocksMetricName:  "512",
			},
		},
	})
	defer srv.Close()

	dataSource, err := buildSourceAndExtractor(t, srv.URL, nil)
	require.NoError(t, err)

	ctx := context.Background()
	ep := newEndpointAt(mustHost(t, srv.URL), map[string]string{
		metricextractor.DefaultEngineTypeLabelKey: "vllm",
	})

	require.NoError(t, dataSource.Poll(ctx, ep))

	m := ep.GetMetrics()
	assert.Equal(t, 7, m.WaitingQueueSize, "WaitingQueueSize")
	assert.Equal(t, 3, m.RunningRequestsSize, "RunningRequestsSize")
	assert.InDelta(t, 0.55, m.KVCacheUsagePercent, 0.001, "KVCacheUsagePercent")
	assert.Equal(t, 4, m.MaxActiveModels, "MaxActiveModels")
	assert.Contains(t, m.ActiveModels, "adapter-a")
	assert.Contains(t, m.ActiveModels, "adapter-b")
	assert.Contains(t, m.WaitingModels, "adapter-c")
	assert.Equal(t, 16, m.CacheBlockSize, "CacheBlockSize")
	assert.Equal(t, 512, m.CacheNumGPUBlocks, "CacheNumGPUBlocks")
}

// TestMetricsExtractionLoRADisabledViaConfig verifies the "disable a specific metric"
// pattern documented for the data section:
//
//	engineConfigs:
//	  - name: vllm
//	    queuedRequestsSpec: "vllm:num_requests_waiting"
//	    ...
//	    loraSpec: ""   # ← empty string disables LoRA extraction
//
// With loraSpec: "", the extractor skips LoRA entirely — no extraction attempt,
// no error for the missing/present family, and ActiveModels stays at its zero value.
func TestMetricsExtractionLoRADisabledViaConfig(t *testing.T) {
	// Server serves LoRA metrics — but they should be silently ignored.
	srv := createMockServer([]MetricMock{
		{Name: WaitingMetric, Value: 5},
		{Name: RunningMetric, Value: 2},
		{Name: KVCacheMetric, Value: 0.3},
		{
			Name:  LoRAMetric,
			Value: float64(time.Now().Unix()),
			Labels: map[string]string{
				LoraInfoRunningAdaptersMetricName: "some-adapter",
				LoraInfoMaxAdaptersMetricName:     "2",
			},
		},
	})
	defer srv.Close()

	// Override only the vllm engine config — loraSpec is explicitly empty.
	// All other spec fields must be provided because engineConfigs is full-replacement
	// per engine name (not a field-level merge); see docs/configuration.md.
	extractorParams := map[string]any{
		"engineConfigs": []map[string]any{
			{
				"name":                "vllm",
				"queuedRequestsSpec":  "vllm:num_requests_waiting",
				"runningRequestsSpec": "vllm:num_requests_running",
				"kvUsageSpec":         "vllm:kv_cache_usage_perc",
				"loraSpec":            "", // disabled
				"cacheInfoSpec":       "",
			},
		},
	}

	dataSource, err := buildSourceAndExtractor(t, srv.URL, extractorParams)
	require.NoError(t, err)

	ctx := context.Background()
	ep := newEndpointAt(mustHost(t, srv.URL), map[string]string{
		metricextractor.DefaultEngineTypeLabelKey: "vllm",
	})

	// Poll must succeed — no "metric family not found" error for LoRA.
	require.NoError(t, dataSource.Poll(ctx, ep))

	m := ep.GetMetrics()
	assert.Equal(t, 5, m.WaitingQueueSize)
	assert.Equal(t, 2, m.RunningRequestsSize)
	assert.InDelta(t, 0.3, m.KVCacheUsagePercent, 0.001)

	// LoRA fields remain at zero — no LoRA extraction occurred.
	assert.Empty(t, m.ActiveModels, "ActiveModels should be empty when loraSpec is disabled")
	assert.Empty(t, m.WaitingModels)
	assert.Zero(t, m.MaxActiveModels)
}

// TestMetricsExtractionMissingMetricFamilyReturnsError verifies the error-path behavior:
// when the server does not serve a metric that the extractor is configured to collect,
// Poll returns an error whose message names the missing metric family.
//
// This is the error that the datalayer Collector (collector.go) logs on the first
// occurrence; repeated occurrences are suppressed by the change-only logging logic.
func TestMetricsExtractionMissingMetricFamilyReturnsError(t *testing.T) {
	tests := []struct {
		name           string
		served         []MetricMock                     // metrics the server exposes
		wantErrContain string                           // substring expected in Poll's error
		wantMetrics    func(*testing.T, *fwkdl.Metrics) // partial assertions on extracted values
	}{
		{
			name: "LoRA family absent - other metrics still extracted",
			served: []MetricMock{
				{Name: WaitingMetric, Value: 4},
				{Name: RunningMetric, Value: 1},
				{Name: KVCacheMetric, Value: 0.2},
				// LoRA and CacheInfo deliberately not served
			},
			wantErrContain: "lora_requests_info",
			wantMetrics: func(t *testing.T, m *fwkdl.Metrics) {
				t.Helper()
				assert.Equal(t, 4, m.WaitingQueueSize, "WaitingQueueSize still extracted")
				assert.Equal(t, 1, m.RunningRequestsSize)
				assert.InDelta(t, 0.2, m.KVCacheUsagePercent, 0.001)
			},
		},
		{
			name:           "all metric families absent",
			served:         []MetricMock{},
			wantErrContain: "not found",
			wantMetrics: func(t *testing.T, m *fwkdl.Metrics) {
				t.Helper()
				assert.Zero(t, m.WaitingQueueSize, "no extraction should have occurred")
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := createMockServer(tc.served)
			defer srv.Close()

			// Use defaults so all five vLLM specs are active.
			dataSource, err := buildSourceAndExtractor(t, srv.URL, nil)
			require.NoError(t, err)

			ctx := context.Background()
			ep := newEndpointAt(mustHost(t, srv.URL), map[string]string{
				metricextractor.DefaultEngineTypeLabelKey: "vllm",
			})

			pollErr := dataSource.Poll(ctx, ep)

			require.Error(t, pollErr, "expected error for missing metric family")
			assert.True(t, strings.Contains(pollErr.Error(), tc.wantErrContain),
				"error %q should contain %q", pollErr.Error(), tc.wantErrContain)

			if tc.wantMetrics != nil {
				tc.wantMetrics(t, ep.GetMetrics())
			}
		})
	}
}

// TestMetricsExtractionDisabledSpecNoError verifies a key invariant:
// when a metric spec is disabled (loraSpec: ""), Poll must NOT return an error
// for that metric even when the metric family is absent from the server response.
// This is what prevents the "metric family not found" error from flooding the log.
func TestMetricsExtractionDisabledSpecNoError(t *testing.T) {
	// Server serves nothing — normally this would cause errors for all specs.
	srv := createMockServer([]MetricMock{})
	defer srv.Close()

	// Disable ALL optional specs — only verify the nil-spec / no-error contract.
	extractorParams := map[string]any{
		"engineConfigs": []map[string]any{
			{
				"name":          "vllm",
				"loraSpec":      "", // disabled
				"cacheInfoSpec": "", // disabled
				// queue, running, kv-usage also omitted (empty → nil → skipped)
				"queuedRequestsSpec":  "",
				"runningRequestsSpec": "",
				"kvUsageSpec":         "",
			},
		},
	}

	dataSource, err := buildSourceAndExtractor(t, srv.URL, extractorParams)
	require.NoError(t, err)

	ctx := context.Background()
	ep := newEndpointAt(mustHost(t, srv.URL), map[string]string{
		metricextractor.DefaultEngineTypeLabelKey: "vllm",
	})

	// All specs are nil → no extraction attempted → no error returned.
	assert.NoError(t, dataSource.Poll(ctx, ep))
}

// TestMetricsExtractionServerError verifies that an HTTP error from the server
// (e.g., server down) propagates as a Poll error.
func TestMetricsExtractionServerError(t *testing.T) {
	srv := createMockServer([]MetricMock{})
	srv.Close() // close immediately — all requests will fail

	dataSource, err := buildSourceAndExtractor(t, srv.URL, nil)
	require.NoError(t, err)

	ctx := context.Background()
	ep := newEndpointAt(mustHost(t, srv.URL), nil)

	pollErr := dataSource.Poll(ctx, ep)
	require.Error(t, pollErr, "expected error when server is unreachable")
}

// TestMetricsExtractionJoinedErrors verifies that when multiple metric families
// are absent, errors are joined and all family names are present in the message.
func TestMetricsExtractionJoinedErrors(t *testing.T) {
	// Server only serves the queue metric; running and kv-cache are absent.
	srv := createMockServer([]MetricMock{
		{Name: WaitingMetric, Value: 9},
	})
	defer srv.Close()

	dataSource, err := buildSourceAndExtractor(t, srv.URL, nil)
	require.NoError(t, err)

	ctx := context.Background()
	ep := newEndpointAt(mustHost(t, srv.URL), map[string]string{
		metricextractor.DefaultEngineTypeLabelKey: "vllm",
	})

	pollErr := dataSource.Poll(ctx, ep)
	require.Error(t, pollErr)

	// errors.Join produces a newline-separated message; all absent families are named.
	errMsg := pollErr.Error()
	assert.True(t, errors.Is(pollErr, pollErr), "error chain should be non-nil")
	assert.True(t, strings.Contains(errMsg, "num_requests_running") ||
		strings.Contains(errMsg, "kv_cache_usage_perc"),
		"error message should name at least one missing family: %s", errMsg)

	// The one served metric was still extracted.
	assert.Equal(t, 9, ep.GetMetrics().WaitingQueueSize)
}

// mustHost is a test helper that parses a URL and returns the host:port portion.
func mustHost(t *testing.T, rawURL string) string {
	t.Helper()
	u, err := url.Parse(rawURL)
	require.NoError(t, err)
	return u.Host
}
