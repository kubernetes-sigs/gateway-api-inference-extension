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
	descInferencePoolPerPodQueueSize = prometheus.NewDesc(
		"inference_pool_per_pod_queue_size",
		metricsutil.HelpMsgWithStability("The total number of requests pending in the model server queue for each underlying pod.", compbasemetrics.ALPHA),
		[]string{
			"name",
			"model_server_pod",
		}, nil,
	)
)

type inferencePoolMetricsCollector struct {
	ds datastore.Datastore
}

// Check if inferencePoolMetricsCollector implements necessary interface
var _ prometheus.Collector = &inferencePoolMetricsCollector{}

// NewInferencePoolMetricsCollector implements the prometheus.Collector interface and
// exposes metrics about inference pool.
func NewInferencePoolMetricsCollector(ds datastore.Datastore) prometheus.Collector {
	return &inferencePoolMetricsCollector{
		ds: ds,
	}
}

// DescribeWithStability implements the prometheus.Collector interface.
func (c *inferencePoolMetricsCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- descInferencePoolPerPodQueueSize
}

// CollectWithStability implements the prometheus.Collector interface.
func (c *inferencePoolMetricsCollector) Collect(ch chan<- prometheus.Metric) {
	pool, err := c.ds.PoolGet()
	if err != nil {
		return
	}

	podMetrics := c.ds.PodList(datastore.AllPodsPredicate)
	if len(podMetrics) == 0 {
		return
	}

	for _, pod := range podMetrics {
		ch <- prometheus.MustNewConstMetric(
			descInferencePoolPerPodQueueSize,
			prometheus.GaugeValue,
			float64(pod.GetMetrics().WaitingQueueSize),
			pool.Name,
			pod.GetMetadata().NamespacedName.Name,
		)
	}
}
