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

package epp

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/extractor/mocks"
	httpds "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/source/http"
)

func TestRuntimePollingDispatch(t *testing.T) {
	tests := []struct {
		name   string
		ticks  int
		verify func(*mocks.Extractor, fwkdl.Endpoint) bool
	}{
		{
			name:  "single poll cycle updates endpoint",
			ticks: 1,
			verify: func(ext *mocks.Extractor, ep fwkdl.Endpoint) bool {
				return ext.CallCount() >= 1
			},
		},
		{
			name:  "multiple ticks trigger multiple polls",
			ticks: 3,
			verify: func(ext *mocks.Extractor, ep fwkdl.Endpoint) bool {
				return ext.CallCount() >= 3
			},
		},
		{
			name:  "endpoint receives metrics on poll",
			ticks: 1,
			verify: func(ext *mocks.Extractor, ep fwkdl.Endpoint) bool {
				metrics := ep.GetMetrics()
				return metrics != nil && ext.CallCount() >= 1
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if testing.Short() {
				t.Skip("Skipping E2E test in short mode")
			}

			metrics := map[string]float64{
				"vllm:num_requests_waiting": 5,
				"vllm:num_requests_running": 2,
				"vllm:kv_cache_usage_perc":  0.75,
			}
			srv := createTestMetricsServer(t, metrics)
			defer srv.Close()

			pollingInterval := 50 * time.Millisecond
			r := datalayer.NewRuntime(pollingInterval)
			ext := mocks.NewPollingExtractor("test-extractor")

			httpSrc, err := httpds.NewHTTPDataSource("http", "/metrics", true, "test-http", "test-source",
				parsePrometheusMetrics, reflect.TypeFor[fwkdl.Metrics]())
			require.NoError(t, err)

			cfg := &datalayer.Config{
				Sources: []datalayer.DataSourceConfig{
					{Plugin: httpSrc, Extractors: []fwkdl.Extractor{ext}},
				},
			}

			require.NoError(t, r.Configure(cfg, false, "", logger))

			ctx, cancel := context.WithCancel(context.Background())
			t.Cleanup(cancel)

			endpointMeta := &fwkdl.EndpointMetadata{
				NamespacedName: types.NamespacedName{Name: "test-pod", Namespace: "test-ns"},
				MetricsHost:    srv.Listener.Addr().String(),
				Port:           "8000",
			}

			ep := r.NewEndpoint(ctx, endpointMeta, nil)
			require.NotNil(t, ep)

			// Wait for at least the expected number of polling cycles
			time.Sleep(time.Duration(tc.ticks+1) * pollingInterval)

			require.Eventually(t, func() bool {
				return tc.verify(ext, ep)
			}, 2*time.Second, 10*time.Millisecond, "Extractor was not called expected number of times")

			r.ReleaseEndpoint(ep)
		})
	}
}

func TestRuntimePollingMultipleExtractors(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	metrics := map[string]float64{
		"vllm:num_requests_waiting": 3,
	}
	srv := createTestMetricsServer(t, metrics)
	defer srv.Close()

	pollingInterval := 50 * time.Millisecond
	r := datalayer.NewRuntime(pollingInterval)
	ext1 := mocks.NewPollingExtractor("extractor-1")
	ext2 := mocks.NewPollingExtractor("extractor-2")

	httpSrc, err := httpds.NewHTTPDataSource("http", "/metrics", true, "test-http", "test-source",
		parsePrometheusMetrics, reflect.TypeFor[fwkdl.Metrics]())
	require.NoError(t, err)

	cfg := &datalayer.Config{
		Sources: []datalayer.DataSourceConfig{
			{Plugin: httpSrc, Extractors: []fwkdl.Extractor{ext1, ext2}},
		},
	}

	require.NoError(t, r.Configure(cfg, false, "", logger))

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	endpointMeta := &fwkdl.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "test-pod", Namespace: "test-ns"},
		MetricsHost:    srv.Listener.Addr().String(),
		Port:           "8000",
	}

	ep := r.NewEndpoint(ctx, endpointMeta, nil)
	require.NotNil(t, ep)

	// Wait for at least 2 polling cycles
	time.Sleep(2 * pollingInterval)

	require.Eventually(t, func() bool {
		return ext1.CallCount() >= 1 && ext2.CallCount() >= 1
	}, 2*time.Second, 10*time.Millisecond, "Both extractors should be called")

	r.ReleaseEndpoint(ep)
}

func TestRuntimePollingEndpointLifecycle(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	metrics := map[string]float64{
		"vllm:num_requests_waiting": 2,
	}
	srv := createTestMetricsServer(t, metrics)
	defer srv.Close()

	pollingInterval := 50 * time.Millisecond
	r := datalayer.NewRuntime(pollingInterval)
	ext := mocks.NewPollingExtractor("lifecycle-extractor")

	httpSrc, err := httpds.NewHTTPDataSource("http", "/metrics", true, "test-http", "test-source",
		parsePrometheusMetrics, reflect.TypeFor[fwkdl.Metrics]())
	require.NoError(t, err)

	cfg := &datalayer.Config{
		Sources: []datalayer.DataSourceConfig{
			{Plugin: httpSrc, Extractors: []fwkdl.Extractor{ext}},
		},
	}

	require.NoError(t, r.Configure(cfg, false, "", logger))

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	endpointMeta := &fwkdl.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "test-pod", Namespace: "test-ns"},
		MetricsHost:    srv.Listener.Addr().String(),
		Port:           "8000",
	}

	ep := r.NewEndpoint(ctx, endpointMeta, nil)
	require.NotNil(t, ep)

	// Wait for at least 2 polling cycles
	time.Sleep(2 * pollingInterval)

	require.Eventually(t, func() bool {
		return ext.CallCount() >= 1
	}, 2*time.Second, 10*time.Millisecond, "Extractor should be called before release")

	initialCount := ext.CallCount()

	r.ReleaseEndpoint(ep)

	// Wait for 2 more polling intervals to ensure no more calls happen
	time.Sleep(2 * pollingInterval)

	require.Equal(t, initialCount, ext.CallCount(), "Extractor should not be called after release")
}

func TestRuntimePollingWithoutExtractors(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	metrics := map[string]float64{
		"vllm:num_requests_waiting": 1,
	}
	srv := createTestMetricsServer(t, metrics)
	defer srv.Close()

	r := datalayer.NewRuntime(50 * time.Millisecond)

	httpSrc, err := httpds.NewHTTPDataSource("http", "/metrics", true, "test-http", "test-source",
		parsePrometheusMetrics, reflect.TypeFor[fwkdl.Metrics]())
	require.NoError(t, err)

	cfg := &datalayer.Config{
		Sources: []datalayer.DataSourceConfig{
			{Plugin: httpSrc, Extractors: nil},
		},
	}

	require.NoError(t, r.Configure(cfg, false, "", logger))

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	endpointMeta := &fwkdl.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "test-pod", Namespace: "test-ns"},
		MetricsHost:    srv.Listener.Addr().String(),
		Port:           "8000",
	}

	ep := r.NewEndpoint(ctx, endpointMeta, nil)
	require.NotNil(t, ep)
}

func TestRuntimePollingHTTPError(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping E2E test in short mode")
	}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	pollingInterval := 50 * time.Millisecond
	r := datalayer.NewRuntime(pollingInterval)
	ext := mocks.NewPollingExtractor("error-extractor")

	httpSrc, err := httpds.NewHTTPDataSource("http", "/metrics", true, "test-http", "test-source",
		parsePrometheusMetrics, reflect.TypeFor[fwkdl.Metrics]())
	require.NoError(t, err)

	cfg := &datalayer.Config{
		Sources: []datalayer.DataSourceConfig{
			{Plugin: httpSrc, Extractors: []fwkdl.Extractor{ext}},
		},
	}

	require.NoError(t, r.Configure(cfg, false, "", logger))

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	endpointMeta := &fwkdl.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "test-pod", Namespace: "test-ns"},
		MetricsHost:    srv.Listener.Addr().String(),
		Port:           "8000",
	}

	ep := r.NewEndpoint(ctx, endpointMeta, nil)
	require.NotNil(t, ep)

	// Wait for several polling cycles - polling should continue despite HTTP errors
	time.Sleep(5 * pollingInterval)

	// Extractor should NOT be called when HTTP errors occur (no data to extract)
	// This test verifies that the collector handles errors gracefully and continues polling
	require.Equal(t, 0, ext.CallCount(), "Extractor should not be called when HTTP errors occur")

	r.ReleaseEndpoint(ep)
}

func createTestMetricsServer(t *testing.T, metrics map[string]float64) *httptest.Server {
	t.Helper()
	reg := prometheus.NewRegistry()

	for name, value := range metrics {
		gauge := prometheus.NewGauge(prometheus.GaugeOpts{
			Name: name,
		})
		gauge.Set(value)
		reg.MustRegister(gauge)
	}

	handler := promhttp.HandlerFor(reg, promhttp.HandlerOpts{})
	return httptest.NewServer(handler)
}

func parsePrometheusMetrics(reader io.Reader) (any, error) {
	metrics := &fwkdl.Metrics{}
	parser := expfmt.NewTextParser(model.LegacyValidation)
	metricFamilies, err := parser.TextToMetricFamilies(reader)
	if err != nil {
		return nil, err
	}

	for _, mf := range metricFamilies {
		for _, m := range mf.GetMetric() {
			if m.Gauge != nil {
				value := m.Gauge.GetValue()
				switch mf.GetName() {
				case "vllm:num_requests_waiting":
					metrics.WaitingQueueSize = int(value)
				case "vllm:num_requests_running":
					metrics.RunningRequestsSize = int(value)
				case "vllm:kv_cache_usage_perc":
					metrics.KVCacheUsagePercent = value
				}
			}
		}
	}

	return metrics, nil
}
