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

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/spf13/pflag"
	"github.com/stretchr/testify/assert"

	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	sourcemetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/source/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/server"
)

// vLLM metric names based on Model Server Protocol
const (
	WaitingMetric     = "vllm:num_requests_waiting"
	RunningMetric     = "vllm:num_requests_running"
	KVCacheMetric     = "vllm:kv_cache_usage_perc"
	LoRAMetric        = "vllm:lora_requests_info"
	CacheConfigMetric = "vllm:cache_config_info"
)

// TestMetricsParityStandard tests that both implementations produce identical results for standard metrics
func TestMetricsParityStandard(t *testing.T) {
	tests := []struct {
		name     string
		mockData []MetricMock
	}{
		{
			name: "Standard vLLM State",
			mockData: []MetricMock{
				{Name: WaitingMetric, Value: 5},
				{Name: RunningMetric, Value: 12},
				{Name: KVCacheMetric, Value: 0.85},
			},
		},
		{
			name: "Zero Values",
			mockData: []MetricMock{
				{Name: WaitingMetric, Value: 0},
				{Name: RunningMetric, Value: 0},
				{Name: KVCacheMetric, Value: 0.0},
			},
		},
		{
			name: "High Load Metrics",
			mockData: []MetricMock{
				{Name: WaitingMetric, Value: 100},
				{Name: RunningMetric, Value: 50},
				{Name: KVCacheMetric, Value: 0.99},
			},
		},
		{
			name: "Partial Metrics - Only Waiting",
			mockData: []MetricMock{
				{Name: WaitingMetric, Value: 3},
			},
		},
		{
			name: "Extra Metrics Ignored",
			mockData: []MetricMock{
				{Name: WaitingMetric, Value: 7},
				{Name: RunningMetric, Value: 3},
				{Name: KVCacheMetric, Value: 0.42},
				{Name: "vllm:extra_metric_not_in_mapping", Value: 999},
				{Name: "vllm:another_unknown_metric", Value: 123},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := createMockServer(tc.mockData)
			defer srv.Close()

			ctx := context.Background()
			pmMetrics, pmErr := parseWithBackendPodMetrics(t, ctx, srv.URL)
			dlMetrics, dlErr := parseWithDatalayerMetrics(t, ctx, srv.URL)

			assert.Equal(t, pmErr != nil, dlErr != nil,
				"Error mismatch!\nPodMetrics err: %v\nDatalayer err: %v", pmErr, dlErr)
			opts := cmpopts.IgnoreFields(fwkdl.Metrics{}, "UpdateTime") // ignore UpdateTime as it will differ
			if diff := cmp.Diff(pmMetrics, dlMetrics, opts); diff != "" {
				t.Errorf("Metrics mismatch (-podmetrics +datalayer):\n%s", diff)
			}
		})
	}
}

// TestMetricsParityMultiLoRA tests multi-LoRA adapter scenarios
func TestMetricsParityMultiLoRA(t *testing.T) {
	tests := []struct {
		name     string
		mockData []MetricMock
	}{
		{
			name: "Multi-LoRA Adapters",
			mockData: []MetricMock{
				{Name: WaitingMetric, Value: 2},
				{Name: RunningMetric, Value: 8},
				{
					Name:  LoRAMetric,
					Value: float64(time.Now().Unix()), // timestamp
					Labels: map[string]string{
						LoraInfoRunningAdaptersMetricName: "sql-expert,code-gen-v2",
						LoraInfoWaitingAdaptersMetricName: "unused-adapter",
						LoraInfoMaxAdaptersMetricName:     "5",
					},
				},
			},
		},
		{
			name: "LoRA with Cache Info",
			mockData: []MetricMock{
				{Name: WaitingMetric, Value: 1},
				{
					Name:  LoRAMetric,
					Value: float64(time.Now().Unix()),
					Labels: map[string]string{
						LoraInfoRunningAdaptersMetricName: "adapter1",
						LoraInfoMaxAdaptersMetricName:     "3",
					},
				},
				{
					Name:  CacheConfigMetric,
					Value: 1,
					Labels: map[string]string{
						CacheConfigBlockSizeInfoMetricName: "16",
						CacheConfigNumGPUBlocksMetricName:  "1024",
					},
				},
			},
		},
		{
			name: "Complex Label Characters",
			mockData: []MetricMock{
				{
					Name:  LoRAMetric,
					Value: float64(time.Now().Unix()),
					Labels: map[string]string{
						LoraInfoRunningAdaptersMetricName: "fine-tune:v1.0/test,model-v2",
						LoraInfoMaxAdaptersMetricName:     "10",
					},
				},
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := createMockServer(tc.mockData)
			defer srv.Close()

			ctx := context.Background()
			pmMetrics, pmErr := parseWithBackendPodMetrics(t, ctx, srv.URL)
			dlMetrics, dlErr := parseWithDatalayerMetrics(t, ctx, srv.URL)

			assert.Equal(t, pmErr != nil, dlErr != nil,
				"Error mismatch!\nPodMetrics err: %v\nDatalayer err: %v", pmErr, dlErr)
			opts := cmpopts.IgnoreFields(fwkdl.Metrics{}, "UpdateTime")
			if diff := cmp.Diff(pmMetrics, dlMetrics, opts); diff != "" {
				t.Errorf("Metrics mismatch (-podmetrics +datalayer):\n%s", diff)
			}
		})
	}
}

// TestMetricsParityTimeout tests timeout handling
func TestMetricsParityTimeout(t *testing.T) {
	srv := createSlowServer(1 * time.Second)
	defer srv.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()

	pmMetrics, pmErr := parseWithBackendPodMetrics(t, ctx, srv.URL)
	dlMetrics, dlErr := parseWithDatalayerMetrics(t, ctx, srv.URL)

	// Both should timeout and error
	assert.Error(t, pmErr, "PodMetrics should have timed out")
	assert.Error(t, dlErr, "Datalayer should have timed out")
	if pmErr != nil && dlErr != nil {
		t.Logf("Both implementations correctly timed out. PodMetrics: %v, Datalayer: %v", pmErr, dlErr)
		return
	}

	// If only one errored, that's a mismatch
	assert.Equal(t, pmErr != nil, dlErr != nil,
		"Timeout behavior mismatch!\nPodMetrics err: %v\nDatalayer err: %v", pmErr, dlErr)
	opts := cmpopts.IgnoreFields(fwkdl.Metrics{}, "UpdateTime")
	if diff := cmp.Diff(pmMetrics, dlMetrics, opts); diff != "" {
		t.Errorf("Partial results mismatch (-podmetrics +datalayer):\n%s", diff)
	}
}

// TestMetricsParityMalformed tests handling of malformed input
func TestMetricsParityMalformed(t *testing.T) {
	tests := []struct {
		name string
		data string
	}{
		{
			name: "Invalid Float",
			data: KVCacheMetric + " 12.abc.34\n",
		},
		{
			name: "Missing Value",
			data: WaitingMetric + " \n",
		},
		{
			name: "Completely Invalid",
			data: "this is not prometheus format at all",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			srv := createMalformedServer(tc.data)
			defer srv.Close()

			ctx := context.Background()
			pmMetrics, pmErr := parseWithBackendPodMetrics(t, ctx, srv.URL)
			dlMetrics, dlErr := parseWithDatalayerMetrics(t, ctx, srv.URL)

			// Both should handle errors similarly
			if (pmErr != nil) != (dlErr != nil) {
				t.Logf("Error handling differs (may be acceptable):\nPodMetrics err: %v\nDatalayer err: %v", pmErr, dlErr)
			}

			// If both succeeded (gracefully handled malformed data), compare results
			if pmErr == nil && dlErr == nil {
				opts := cmpopts.IgnoreFields(fwkdl.Metrics{}, "UpdateTime")
				if diff := cmp.Diff(pmMetrics, dlMetrics, opts); diff != "" {
					t.Errorf("Metrics mismatch (-podmetrics +datalayer):\n%s", diff)
				}
			}
		})
	}
}

// MetricMock defines a metric to be served by the test server
type MetricMock struct {
	Name   string
	Value  float64
	Labels map[string]string
}

// parseWithBackendPodMetrics uses the backend.PodMetrics implementation
func parseWithBackendPodMetrics(t *testing.T, ctx context.Context, urlStr string) (*fwkdl.Metrics, error) {
	t.Helper()

	mapping, err := backendmetrics.NewMetricMapping(
		WaitingMetric,
		RunningMetric,
		KVCacheMetric,
		LoRAMetric,
		CacheConfigMetric,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create metric mapping: %w", err)
	}

	parsedURL, err := url.Parse(urlStr)
	if err != nil {
		return nil, fmt.Errorf("failed to parse URL: %w", err)
	}

	client := &backendmetrics.PodMetricsClientImpl{
		MetricMapping:            mapping,
		ModelServerMetricsPath:   "",
		ModelServerMetricsScheme: parsedURL.Scheme,
		Client:                   &http.Client{},
	}

	metadata := &fwkdl.EndpointMetadata{
		MetricsHost: parsedURL.Host,
	}

	existing := fwkdl.NewMetrics()
	return client.FetchMetrics(ctx, metadata, existing)
}

// parseWithDatalayerMetrics uses the datalayer source + extractor implementation
func parseWithDatalayerMetrics(t *testing.T, ctx context.Context, urlStr string) (*fwkdl.Metrics, error) {
	t.Helper()

	parsedURL, err := url.Parse(urlStr)
	if err != nil {
		return nil, fmt.Errorf("failed to parse URL: %w", err)
	}

	cleanup := setupTestFlags(t) // set-up test flags and restore on cleanup
	defer cleanup()

	// CLI flags to match the test server URL
	if err := pflag.CommandLine.Set("model-server-metrics-scheme", parsedURL.Scheme); err != nil {
		return nil, fmt.Errorf("failed to set scheme flag: %w", err)
	}
	if err := pflag.CommandLine.Set("model-server-metrics-path", parsedURL.Path); err != nil {
		return nil, fmt.Errorf("failed to set path flag: %w", err)
	}

	mapping, err := NewMapping(
		WaitingMetric,
		RunningMetric,
		KVCacheMetric,
		LoRAMetric,
		CacheConfigMetric,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create mapping: %w", err)
	}

	registry := NewMappingRegistry()
	if err := registry.Register(DefaultEngineType, mapping); err != nil {
		return nil, fmt.Errorf("failed to register mapping: %w", err)
	}

	extractor, err := NewModelServerExtractor(registry, DefaultEngineTypeLabelKey)
	if err != nil {
		return nil, fmt.Errorf("failed to create extractor: %w", err)
	}

	plugin, err := sourcemetrics.MetricsDataSourceFactory(
		"test-metrics-source",
		nil, // use default parameters from flags
		nil, // no plugin handle needed for test
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create data source: %w", err)
	}

	dataSource, ok := plugin.(fwkdl.DataSource)
	if !ok {
		return nil, errors.New("plugin is not a DataSource")
	}

	if err := dataSource.AddExtractor(extractor); err != nil {
		return nil, fmt.Errorf("failed to add extractor: %w", err)
	}

	metadata := &fwkdl.EndpointMetadata{
		MetricsHost: parsedURL.Host,
		Labels: map[string]string{
			DefaultEngineTypeLabelKey: DefaultEngineType,
		},
	}
	endpoint := fwkdl.NewEndpoint(metadata, fwkdl.NewMetrics())

	if err := dataSource.Collect(ctx, endpoint); err != nil {
		return endpoint.GetMetrics(), err
	}

	return endpoint.GetMetrics(), nil
}

// setupTestFlags creates a temporary FlagSet for testing and returns a cleanup function
func setupTestFlags(t *testing.T) func() {
	t.Helper()
	originalFlags := pflag.CommandLine
	testFlags := pflag.NewFlagSet("test", pflag.ContinueOnError)
	pflag.CommandLine = testFlags

	opts := server.NewOptions()
	opts.AddFlags(testFlags)

	return func() {
		pflag.CommandLine = originalFlags
	}
}

// createMockServer creates an HTTP test server that serves Prometheus metrics
func createMockServer(metrics []MetricMock) *httptest.Server {
	reg := prometheus.NewRegistry()

	grouped := make(map[string][]MetricMock) // group by name to handle multiple label combinations
	for _, m := range metrics {
		grouped[m.Name] = append(grouped[m.Name], m)
	}

	for name, instances := range grouped {
		if len(instances) == 1 && len(instances[0].Labels) == 0 { // simple gauge without labels
			gauge := prometheus.NewGauge(prometheus.GaugeOpts{
				Name: name,
			})
			gauge.Set(instances[0].Value)
			reg.MustRegister(gauge)
		} else { // metrics with labels or multiple instances
			var labelKeys []string
			if len(instances[0].Labels) > 0 {
				for k := range instances[0].Labels {
					labelKeys = append(labelKeys, k)
				}
			}
			gv := prometheus.NewGaugeVec(prometheus.GaugeOpts{Name: name}, labelKeys)
			for _, inst := range instances {
				gv.With(inst.Labels).Set(inst.Value)
			}
			reg.MustRegister(gv)
		}
	}

	handler := promhttp.HandlerFor(reg, promhttp.HandlerOpts{})
	return httptest.NewServer(handler)
}

// createSlowServer creates a server that delays before responding
func createSlowServer(delay time.Duration) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(delay)
		fmt.Fprintf(w, "%s 10\n", WaitingMetric)
	}))
}

// createMalformedServer creates a server that returns invalid Prometheus format
func createMalformedServer(raw string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, raw)
	}))
}
