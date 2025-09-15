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
	"fmt"
	"sync/atomic"
)

// EndpointPodState allows management of the Pod related attributes.
type EndpointPodState interface {
	GetPod() *PodInfo
	UpdatePod(*PodInfo)
}

// EndpointMetricsState allows management of the Metrics related attributes.
type EndpointMetricsState interface {
	GetMetrics() *Metrics
	UpdateMetrics(*Metrics)
}

// EndpointRunningRequestsState allows management of the Pod related attributes.
type EndpointRunningRequestsState interface {
	GetRunningRequests() *RequestPriorityQueue
	AddRequest(requestID string, tpot float64) bool
	RemoveRequest(requestID string) bool
	UpdateRequest(requestID string, tpot float64) bool
	GetRequestCount() int
	ContainsRequest(requestID string) bool
	PeekRequestPriorityQueue() *Request
}

// Endpoint represents an inference serving endpoint and its related attributes.
type Endpoint interface {
	fmt.Stringer
	EndpointPodState
	EndpointMetricsState
	EndpointRunningRequestsState
	AttributeMap
}

// ModelServer is an implementation of the Endpoint interface.
type ModelServer struct {
	pod        atomic.Pointer[PodInfo]
	metrics    atomic.Pointer[Metrics]
	attributes *Attributes
}

// NewEndpoint return a new (uninitialized) ModelServer.
func NewEndpoint() *ModelServer {
	return &ModelServer{
		attributes: NewAttributes(),
	}
}

// String returns a representation of the ModelServer. For brevity, only names of
// extended attributes are returned and not their values.
func (srv *ModelServer) String() string {
	return fmt.Sprintf("Pod: %v; Metrics: %v; Attributes: %v", srv.GetPod(), srv.GetMetrics(), srv.Keys())
}

func (srv *ModelServer) GetPod() *PodInfo {
	return srv.pod.Load()
}

func (srv *ModelServer) UpdatePod(k8sPod *corev1.Pod) {
	currentPod := srv.GetPod()
	updatedPod := ToPodInfo(k8sPod)

	// Preserve the existing running requests queue if it exists
	if currentPod != nil && currentPod.GetRunningRequests() != nil {
		updatedPod.RunningRequests = currentPod.GetRunningRequests()
	}

	srv.pod.Store(updatedPod)
}

func (srv *ModelServer) GetMetrics() *Metrics {
	return srv.metrics.Load()
}

func (srv *ModelServer) UpdateMetrics(metrics *Metrics) {
	srv.metrics.Store(metrics)
}

// New methods for priority queue integration
func (srv *ModelServer) GetRunningRequests() *RequestPriorityQueue {
	pod := srv.GetPod()
	if pod == nil {
		return nil
	}
	return pod.RunningRequests
}

func (srv *ModelServer) AddRequest(requestID string, tpot float64) bool {
	pod := srv.GetPod()
	if pod == nil || pod.RunningRequests == nil {
		return false
	}
	success := pod.RunningRequests.Add(requestID, tpot)
	// No need to update metrics since we removed ActualRunningRequests
	return success
}

func (srv *ModelServer) RemoveRequest(requestID string) bool {
	pod := srv.GetPod()
	if pod == nil || pod.RunningRequests == nil {
		return false
	}
	_, success := pod.RunningRequests.Remove(requestID)
	// No need to update metrics since we removed ActualRunningRequests
	return success
}

func (srv *ModelServer) UpdateRequest(requestID string, tpot float64) bool {
	pod := srv.GetPod()
	if pod == nil || pod.RunningRequests == nil {
		return false
	}
	return pod.RunningRequests.Update(requestID, tpot)
}

func (srv *ModelServer) GetRequestCount() int {
	pod := srv.GetPod()
	if pod == nil || pod.RunningRequests == nil {
		return 0
	}
	return pod.RunningRequests.GetSize()
}

func (srv *ModelServer) ContainsRequest(requestID string) bool {
	pod := srv.GetPod()
	if pod == nil || pod.RunningRequests == nil {
		return false
	}
	return pod.RunningRequests.Contains(requestID)
}

func (srv *ModelServer) PeekRequestPriorityQueue() *Request {
	pod := srv.GetPod()
	if pod == nil || pod.RunningRequests == nil {
		return nil
	}
	return pod.RunningRequests.Peek()
}

func (srv *ModelServer) Put(key string, value Cloneable) {
	srv.attributes.Put(key, value)
}

func (srv *ModelServer) Get(key string) (Cloneable, bool) {
	return srv.attributes.Get(key)
}

func (srv *ModelServer) Keys() []string {
	return srv.attributes.Keys()
}

func (srv *ModelServer) Clone() *ModelServer {
	clone := &ModelServer{
		attributes: srv.attributes.Clone(),
	}
	clone.pod.Store(srv.pod.Load().Clone())
	clone.metrics.Store(srv.metrics.Load().Clone())
	return clone
}
