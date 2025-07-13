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

package datalayer

import (
	corev1 "k8s.io/api/core/v1"
)

// MetricsLoop is a temporary bridge interface to match backend.metrics.PodMetrics
// collection of metrics. It will be phased out as we move towards decoupled collection
// via a DataSource.
type MetricsLoop interface {
	StopRefreshLoop()
}

// EndpointPodState allows management of the Pod related attributes.
type EndpointPodState interface {
	GetPod() *PodInfo
	UpdatePod(*corev1.Pod)
}

// EndpointMetricsState allows management of the Metrics related attributes.
type EndpointMetricsState interface {
	GetMetrics() *Metrics
	UpdateMetrics(*Metrics)
}

// Endpoint represents an inference serving endpoint and its related attributes.
type Endpoint interface {
	MetricsLoop // TODO: remove once transition over to independent data source
	EndpointPodState
	EndpointMetricsState
	AttributeMap
}
