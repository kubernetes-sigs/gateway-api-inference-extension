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
	"net/http/httptest"
	"sort"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// vLLM metric names based on the Model Server Protocol defaults.
const (
	WaitingMetric     = "vllm:num_requests_waiting"
	RunningMetric     = "vllm:num_requests_running"
	KVCacheMetric     = "vllm:kv_cache_usage_perc"
	LoRAMetric        = "vllm:lora_requests_info"
	CacheConfigMetric = "vllm:cache_config_info"
)

// MetricMock defines a metric to be served by the test server.
type MetricMock struct {
	Name   string
	Value  float64
	Labels map[string]string
}

// createMockServer creates an HTTP test server that serves Prometheus metrics.
func createMockServer(metrics []MetricMock) *httptest.Server {
	reg := prometheus.NewRegistry()

	grouped := make(map[string][]MetricMock) // group by name to handle multiple label combinations
	for _, m := range metrics {
		grouped[m.Name] = append(grouped[m.Name], m)
	}

	for name, instances := range grouped {
		if len(instances) == 1 && len(instances[0].Labels) == 0 { // simple gauge without labels
			gauge := prometheus.NewGauge(prometheus.GaugeOpts{Name: name})
			gauge.Set(instances[0].Value)
			reg.MustRegister(gauge)
		} else { // metrics with labels or multiple instances
			var labelKeys []string
			if len(instances[0].Labels) > 0 {
				for k := range instances[0].Labels {
					labelKeys = append(labelKeys, k)
				}
				sort.Strings(labelKeys) // deterministic order
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
