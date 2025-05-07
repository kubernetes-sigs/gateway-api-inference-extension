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

package util

import (
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

// GetMetricsFromPodInfo returns *metrics.Metrics object it one exists in PodInfo data.
// if it doesn't exist, the function return nil.
func GetMetricsFromPodInfo(pod types.Pod) *metrics.MetricsData {
	podMetrics, ok := pod.GetData()[metrics.MetricsDataKey]
	if !ok {
		return nil // no entry in the map with metrics key
	}
	return podMetrics.(*metrics.MetricsData)
}
