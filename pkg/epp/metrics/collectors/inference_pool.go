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
	"k8s.io/component-base/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
)

var (
	descInferencePoolPerPodQueueSize = metrics.NewDesc(
		"inference_pool_per_pod_queue_size",
		"The total number of requests pending in the model server queue for each underlying pod.",
		[]string{
			"name",
			"model_server_pod",
		}, nil,
		metrics.ALPHA,
		"",
	)
)

type inferencePoolMetricsCollector struct {
	metrics.BaseStableCollector

	ds datastore.Datastore
}

// Check if inferencePoolMetricsCollector implements necessary interface
var _ metrics.StableCollector = &inferencePoolMetricsCollector{}

// NewInferencePoolMetricsCollector implements the metrics.StableCollector interface and
// exposes metrics about inference pool.
func NewInferencePoolMetricsCollector(ds datastore.Datastore) metrics.StableCollector {
	return &inferencePoolMetricsCollector{
		ds: ds,
	}
}

// DescribeWithStability implements the metrics.StableCollector interface.
func (c *inferencePoolMetricsCollector) DescribeWithStability(ch chan<- *metrics.Desc) {
	ch <- descInferencePoolPerPodQueueSize
}

// CollectWithStability implements the metrics.StableCollector interface.
func (c *inferencePoolMetricsCollector) CollectWithStability(ch chan<- metrics.Metric) {
	pool, err := c.ds.PoolGet()
	if err != nil {
		return
	}

	podMetrics := c.ds.PodGetAll()
	if len(podMetrics) == 0 {
		return
	}

	for _, pod := range podMetrics {
		ch <- metrics.NewLazyConstMetric(
			descInferencePoolPerPodQueueSize,
			metrics.GaugeValue,
			float64(pod.GetMetrics().WaitingQueueSize),
			pool.Name,
			pod.GetPod().NamespacedName.Name,
		)
	}
}
