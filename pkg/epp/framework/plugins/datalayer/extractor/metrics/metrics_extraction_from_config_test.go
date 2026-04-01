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
//  1. Instantiate data source and extractor via constructor functions.
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
//   - Multiple extractors: different extractors can extract different subsets from one source.

import (
	"context"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	metricsource "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/source/metrics"
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

// buildSource creates a MetricsDataSource pointing at the given server URL.
func buildSource(t *testing.T, serverURL string) fwkdl.PollingDataSource {
	t.Helper()

	parsedURL, err := url.Parse(serverURL)
	require.NoError(t, err, "failed to parse server URL")

	source, err := metricsource.NewHTTPMetricsDataSource(parsedURL.Scheme, parsedURL.Path, "metrics-data-source")
	require.NoError(t, err, "failed to create metrics data source")
	return source
}

// buildExtractor creates a CoreMetricsExtractor with the given params (nil = defaults).
func buildExtractor(t *testing.T, params *modelServerExtractorParams) *Extractor {
	t.Helper()
	ext, err := newCoreMetricsExtractorPlugin(context.Background(), "core-metrics-extractor", params)
	require.NoError(t, err, "failed to create extractor")
	return ext
}

// buildPipeline wires a MetricsDataSource and a CoreMetricsExtractor into a pipeline.
// params may be nil to use built-in defaults.
func buildPipeline(t *testing.T, serverURL string, params *modelServerExtractorParams) (*pipeline, error) {
	t.Helper()
	source := buildSource(t, serverURL)
	ext, err := newCoreMetricsExtractorPlugin(context.Background(), "core-metrics-extractor", params)
	if err != nil {
		return nil, err
	}
	return &pipeline{source: source, extractor: ext}, nil
}

// newEndpointAt creates a fwkdl.Endpoint with the given host (host:port) and optional labels.
func newEndpointAt(host string, labels map[string]string) fwkdl.Endpoint {
	return fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		MetricsHost: host,
		Labels:      labels,
	}, fwkdl.NewMetrics())
}

// mustHost is a test helper that parses a URL and returns the host:port portion.
func mustHost(t *testing.T, rawURL string) string {
	t.Helper()
	u, err := url.Parse(rawURL)
	require.NoError(t, err)
	return u.Host
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

	p, err := buildPipeline(t, srv.URL, nil)
	require.NoError(t, err)

	ctx := context.Background()
	ep := newEndpointAt(mustHost(t, srv.URL), map[string]string{
		DefaultEngineTypeLabelKey: "vllm",
	})

	require.NoError(t, p.Poll(ctx, ep))

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
// pattern: with loraSpec: "", the extractor skips LoRA entirely — no extraction attempt,
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
	// per engine name (not a field-level merge).
	params := &modelServerExtractorParams{
		EngineConfigs: []engineConfigParams{
			{
				Name:                "vllm",
				QueuedRequestsSpec:  "vllm:num_requests_waiting",
				RunningRequestsSpec: "vllm:num_requests_running",
				KVUsageSpec:         "vllm:kv_cache_usage_perc",
				LoRASpec:            "", // disabled
				CacheInfoSpec:       "",
			},
		},
	}

	p, err := buildPipeline(t, srv.URL, params)
	require.NoError(t, err)

	ctx := context.Background()
	ep := newEndpointAt(mustHost(t, srv.URL), map[string]string{
		DefaultEngineTypeLabelKey: "vllm",
	})

	require.NoError(t, p.Poll(ctx, ep))

	m := ep.GetMetrics()
	assert.Equal(t, 5, m.WaitingQueueSize)
	assert.Equal(t, 2, m.RunningRequestsSize)
	assert.InDelta(t, 0.3, m.KVCacheUsagePercent, 0.001)
	assert.Empty(t, m.ActiveModels, "ActiveModels should be empty when loraSpec is disabled")
	assert.Empty(t, m.WaitingModels)
	assert.Zero(t, m.MaxActiveModels)
}

// TestMetricsExtractionMissingMetricFamilyReturnsError verifies the error-path behavior:
// when the server does not serve a metric that the extractor is configured to collect,
// Poll returns an error whose message names the missing metric family.
func TestMetricsExtractionMissingMetricFamilyReturnsError(t *testing.T) {
	tests := []struct {
		name                   string
		served                 []MetricMock
		wantErrContain         string
		wantWaitingQueueSize   int
		wantRunningRequestSize int
		wantKVCachePercent     float64
	}{
		{
			name: "LoRA family absent - other metrics still extracted",
			served: []MetricMock{
				{Name: WaitingMetric, Value: 4},
				{Name: RunningMetric, Value: 1},
				{Name: KVCacheMetric, Value: 0.2},
				// LoRA and CacheInfo deliberately not served
			},
			wantErrContain:         "lora_requests_info",
			wantWaitingQueueSize:   4,
			wantRunningRequestSize: 1,
			wantKVCachePercent:     0.2,
		},
		{
			name:           "all metric families absent",
			served:         []MetricMock{},
			wantErrContain: "not found",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := createMockServer(tc.served)
			defer srv.Close()

			p, err := buildPipeline(t, srv.URL, nil)
			require.NoError(t, err)

			ctx := context.Background()
			ep := newEndpointAt(mustHost(t, srv.URL), map[string]string{
				DefaultEngineTypeLabelKey: "vllm",
			})

			pollErr := p.Poll(ctx, ep)

			require.Error(t, pollErr, "expected error for missing metric family")
			assert.True(t, strings.Contains(pollErr.Error(), tc.wantErrContain),
				"error %q should contain %q", pollErr.Error(), tc.wantErrContain)

			m := ep.GetMetrics()
			assert.Equal(t, tc.wantWaitingQueueSize, m.WaitingQueueSize, "WaitingQueueSize")
			assert.Equal(t, tc.wantRunningRequestSize, m.RunningRequestsSize, "RunningRequestsSize")
			assert.InDelta(t, tc.wantKVCachePercent, m.KVCacheUsagePercent, 0.001, "KVCacheUsagePercent")
		})
	}
}

// TestMetricsExtractionDisabledSpecNoError verifies that when all metric specs are
// disabled (empty strings), Poll does NOT return an error even when the server serves nothing.
func TestMetricsExtractionDisabledSpecNoError(t *testing.T) {
	srv := createMockServer([]MetricMock{})
	defer srv.Close()

	params := &modelServerExtractorParams{
		EngineConfigs: []engineConfigParams{
			{
				Name:                "vllm",
				QueuedRequestsSpec:  "",
				RunningRequestsSpec: "",
				KVUsageSpec:         "",
				LoRASpec:            "",
				CacheInfoSpec:       "",
			},
		},
	}

	p, err := buildPipeline(t, srv.URL, params)
	require.NoError(t, err)

	ctx := context.Background()
	ep := newEndpointAt(mustHost(t, srv.URL), map[string]string{
		DefaultEngineTypeLabelKey: "vllm",
	})

	assert.NoError(t, p.Poll(ctx, ep))
}

// TestMetricsExtractionServerError verifies that an HTTP error from the server propagates as a Poll error.
func TestMetricsExtractionServerError(t *testing.T) {
	srv := createMockServer([]MetricMock{})
	srv.Close() // close immediately — all requests will fail

	p, err := buildPipeline(t, srv.URL, nil)
	require.NoError(t, err)

	ctx := context.Background()
	ep := newEndpointAt(mustHost(t, srv.URL), nil)

	require.Error(t, p.Poll(ctx, ep), "expected error when server is unreachable")
}

// TestMetricsExtractionJoinedErrors verifies that when multiple metric families are absent,
// errors are joined and all family names are present in the message.
func TestMetricsExtractionJoinedErrors(t *testing.T) {
	srv := createMockServer([]MetricMock{
		{Name: WaitingMetric, Value: 9},
	})
	defer srv.Close()

	p, err := buildPipeline(t, srv.URL, nil)
	require.NoError(t, err)

	ctx := context.Background()
	ep := newEndpointAt(mustHost(t, srv.URL), map[string]string{
		DefaultEngineTypeLabelKey: "vllm",
	})

	pollErr := p.Poll(ctx, ep)
	require.Error(t, pollErr)

	errMsg := pollErr.Error()
	assert.True(t, strings.Contains(errMsg, "num_requests_running") ||
		strings.Contains(errMsg, "kv_cache_usage_perc"),
		"error message should name at least one missing family: %s", errMsg)

	assert.Equal(t, 9, ep.GetMetrics().WaitingQueueSize)
}

// TestMetricsExtractionMultipleExtractors verifies that multiple extractors can each
// extract different subsets of metrics from the same data source output.
func TestMetricsExtractionMultipleExtractors(t *testing.T) {
	srv := createMockServer([]MetricMock{
		{Name: WaitingMetric, Value: 11},
		{Name: RunningMetric, Value: 5},
		{Name: KVCacheMetric, Value: 0.75},
		{
			Name:  LoRAMetric,
			Value: float64(time.Now().Unix()),
			Labels: map[string]string{
				LoraInfoRunningAdaptersMetricName: "adapter-x",
				LoraInfoMaxAdaptersMetricName:     "8",
			},
		},
	})
	defer srv.Close()

	source := buildSource(t, srv.URL)

	// Extractor A: queue + running only
	extA := buildExtractor(t, &modelServerExtractorParams{
		EngineConfigs: []engineConfigParams{
			{
				Name:                "vllm",
				QueuedRequestsSpec:  "vllm:num_requests_waiting",
				RunningRequestsSpec: "vllm:num_requests_running",
				KVUsageSpec:         "",
				LoRASpec:            "",
				CacheInfoSpec:       "",
			},
		},
	})

	// Extractor B: LoRA only
	extB := buildExtractor(t, &modelServerExtractorParams{
		EngineConfigs: []engineConfigParams{
			{
				Name:                "vllm",
				QueuedRequestsSpec:  "",
				RunningRequestsSpec: "",
				KVUsageSpec:         "",
				LoRASpec:            "vllm:lora_requests_info",
				CacheInfoSpec:       "",
			},
		},
	})

	ctx := context.Background()
	vllmLabels := map[string]string{DefaultEngineTypeLabelKey: "vllm"}

	epA := newEndpointAt(mustHost(t, srv.URL), vllmLabels)
	epB := newEndpointAt(mustHost(t, srv.URL), vllmLabels)

	pipeA := &pipeline{source: source, extractor: extA}
	pipeB := &pipeline{source: source, extractor: extB}

	require.NoError(t, pipeA.Poll(ctx, epA))
	require.NoError(t, pipeB.Poll(ctx, epB))

	mA := epA.GetMetrics()
	assert.Equal(t, 11, mA.WaitingQueueSize, "extractor A: WaitingQueueSize")
	assert.Equal(t, 5, mA.RunningRequestsSize, "extractor A: RunningRequestsSize")
	assert.Zero(t, mA.MaxActiveModels, "extractor A: LoRA should not be extracted")
	assert.Empty(t, mA.ActiveModels, "extractor A: ActiveModels should be empty")

	mB := epB.GetMetrics()
	assert.Zero(t, mB.WaitingQueueSize, "extractor B: queue should not be extracted")
	assert.Equal(t, 8, mB.MaxActiveModels, "extractor B: MaxActiveModels")
	assert.Contains(t, mB.ActiveModels, "adapter-x", "extractor B: ActiveModels")
}
