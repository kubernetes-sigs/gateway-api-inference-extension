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

package collectors

import (
	"github.com/prometheus/client_golang/prometheus"
	compbasemetrics "k8s.io/component-base/metrics"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	metricsutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/metrics"
)

var (
	descInferenceModelReady = prometheus.NewDesc(
		"inference_model_ready",
		metricsutil.HelpMsgWithStability("Indicates which InferenceModels are ready to serve by the epp. Value 1 indicates the model is tracked and ready, 0 indicates not ready.", compbasemetrics.ALPHA),
		[]string{
			"pool_name",
			"model_name",
		}, nil,
	)
)

type inferenceModelMetricsCollector struct {
	ds datastore.Datastore
}

// Check if inferenceModelMetricsCollector implements necessary interface
var _ prometheus.Collector = &inferenceModelMetricsCollector{}

// NewInferenceModelMetricsCollector implements the prometheus.Collector interface and
// exposes metrics about inference models tracked by the epp.
func NewInferenceModelMetricsCollector(ds datastore.Datastore) prometheus.Collector {
	return &inferenceModelMetricsCollector{
		ds: ds,
	}
}

// Describe implements the prometheus.Collector interface.
func (c *inferenceModelMetricsCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- descInferenceModelReady
}

// Collect implements the prometheus.Collector interface.
func (c *inferenceModelMetricsCollector) Collect(ch chan<- prometheus.Metric) {
	pool, err := c.ds.PoolGet()
	if err != nil {
		// Pool not synced yet, no metrics to expose
		return
	}

	models := c.ds.ModelGetAll()
	if len(models) == 0 {
		return
	}

	for _, model := range models {
		// Each model tracked by the datastore is considered "ready to serve" by the epp
		ch <- prometheus.MustNewConstMetric(
			descInferenceModelReady,
			prometheus.GaugeValue,
			1.0, // Value 1 indicates the model is ready to serve by the epp
			pool.Name,
			model.Spec.ModelName,
		)
	}
}
