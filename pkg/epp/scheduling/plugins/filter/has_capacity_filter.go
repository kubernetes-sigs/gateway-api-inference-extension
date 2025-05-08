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

package filter

import (
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/config"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

// compile-time type validation
var _ plugins.Filter = &HasCapacityFilter{}

// NewHasCapacityFilter returns a new HasCapacityFilter.
func NewHasCapacityFilter() *HasCapacityFilter {
	return &HasCapacityFilter{
		queueThreshold:   config.Conf.QueueThresholdCritical,
		kvCacheThreshold: config.Conf.KVCacheThreshold,
	}
}

// HasCapacityFilter filters only pods that has capacity for sheddable requests.
type HasCapacityFilter struct {
	queueThreshold   int
	kvCacheThreshold float64
}

// Name returns the name of the filter.
func (f *HasCapacityFilter) Name() string {
	return "has-capacity"
}

// Filter filters out pods that doesn't meet the filter criteria.
func (f *HasCapacityFilter) Filter(ctx *types.SchedulingContext, pods []types.Pod) []types.Pod {
	filteredPods := []types.Pod{}

	for _, pod := range pods {
		if pod.GetMetrics().WaitingQueueSize <= f.queueThreshold && pod.GetMetrics().KVCacheUsagePercent <= f.kvCacheThreshold {
			filteredPods = append(filteredPods, pod)
		}
	}

	return filteredPods
}
