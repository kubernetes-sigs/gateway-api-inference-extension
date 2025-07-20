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

package metrics

import (
	"context"
	"fmt"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// FakePodMetrics is an implementation of PodMetrics that doesn't run the async refresh loop.
// FakePodMetrics implements the PodMetrics interface for testing
type FakePodMetrics struct {
	pod             *backend.Pod
	runningRequests *backend.RequestPriorityQueue
	stopped         bool
	mu              sync.RWMutex // Protect the stopped field and operations
}

func NewFakePodMetrics(k8sPod *corev1.Pod) *FakePodMetrics {
	pod := &backend.Pod{
		NamespacedName: types.NamespacedName{
			Name:      k8sPod.Name,
			Namespace: k8sPod.Namespace,
		},
		Address:         k8sPod.Status.PodIP,
		Labels:          make(map[string]string),
		RunningRequests: backend.NewRequestPriorityQueue(),
	}
	
	for k, v := range k8sPod.Labels {
		pod.Labels[k] = v
	}
	
	return &FakePodMetrics{
		pod:             pod,
		runningRequests: pod.RunningRequests,
		stopped:         false,
	}
}

func (f *FakePodMetrics) GetPod() *backend.Pod {
	return f.pod
}

func (f *FakePodMetrics) GetMetrics() *MetricsState {
	return &MetricsState{
		ActiveModels:  make(map[string]int),
		WaitingModels: make(map[string]int),
		UpdateTime:    time.Now(),
	}
}

func (f *FakePodMetrics) UpdatePod(k8sPod *corev1.Pod) {
	f.pod.NamespacedName = types.NamespacedName{Name: k8sPod.Name, Namespace: k8sPod.Namespace}
	f.pod.Address = k8sPod.Status.PodIP
	f.pod.Labels = make(map[string]string)
	for k, v := range k8sPod.Labels {
		f.pod.Labels[k] = v
	}
}

func (f *FakePodMetrics) StopRefreshLoop() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.stopped = true
}

func (f *FakePodMetrics) String() string {
	return fmt.Sprintf("FakePodMetrics{%s}", f.pod.NamespacedName)
}

func (f *FakePodMetrics) GetRunningRequests() *backend.RequestPriorityQueue {
	f.mu.RLock()
	defer f.mu.RUnlock()
	if f.stopped {
		return nil // Return nil for stopped pod metrics
	}
	return f.runningRequests
}

func (f *FakePodMetrics) AddRequest(requestID string, tpot float64) bool {
	f.mu.RLock()
	defer f.mu.RUnlock()
	if f.stopped {
		return false // Reject operations after stopped
	}
	return f.runningRequests.Add(requestID, tpot)
}

func (f *FakePodMetrics) RemoveRequest(requestID string) bool {
	f.mu.RLock()
	defer f.mu.RUnlock()
	if f.stopped {
		return false // Reject operations after stopped
	}
	_, success := f.runningRequests.Remove(requestID)
	return success
}

func (f *FakePodMetrics) UpdateRequest(requestID string, tpot float64) bool {
	f.mu.RLock()
	defer f.mu.RUnlock()
	if f.stopped {
		return false // Reject operations after stopped
	}
	return f.runningRequests.Update(requestID, tpot)
}

func (f *FakePodMetrics) GetRequestCount() int {
	f.mu.RLock()
	defer f.mu.RUnlock()
	if f.stopped {
		return 0 // Return 0 after stopped
	}
	return f.runningRequests.GetSize()
}

func (f *FakePodMetrics) ContainsRequest(requestID string) bool {
	f.mu.RLock()
	defer f.mu.RUnlock()
	if f.stopped {
		return false // Return false after stopped
	}
	return f.runningRequests.Contains(requestID)
}

// IsStopped returns whether the pod metrics has been stopped (useful for testing)
func (f *FakePodMetrics) IsStopped() bool {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.stopped
}

// FakePodMetricsClient allows controlling metrics responses for testing
type FakePodMetricsClient struct {
	errMu sync.RWMutex
	Err   map[types.NamespacedName]error
	resMu sync.RWMutex
	Res   map[types.NamespacedName]*MetricsState
}

// NewFakePodMetricsClient creates a new fake pod metrics client
func NewFakePodMetricsClient() *FakePodMetricsClient {
	return &FakePodMetricsClient{
		Err: make(map[types.NamespacedName]error),
		Res: make(map[types.NamespacedName]*MetricsState),
	}
}

func (f *FakePodMetricsClient) FetchMetrics(ctx context.Context, pod *backend.Pod, existing *MetricsState, _ int32) (*MetricsState, error) {
	f.errMu.RLock()
	err, ok := f.Err[pod.NamespacedName]
	f.errMu.RUnlock()
	if ok {
		return nil, err
	}
	
	f.resMu.RLock()
	res, ok := f.Res[pod.NamespacedName]
	f.resMu.RUnlock()
	if !ok {
		// Return a default metrics state if none configured
		return &MetricsState{
			ActiveModels:  make(map[string]int),
			WaitingModels: make(map[string]int),
			UpdateTime:    time.Now(),
		}, nil
	}
	
	log.FromContext(ctx).V(logutil.VERBOSE).Info("Fetching metrics for pod", "existing", existing, "new", res)
	return res.Clone(), nil
}

func (f *FakePodMetricsClient) SetRes(new map[types.NamespacedName]*MetricsState) {
	f.resMu.Lock()
	defer f.resMu.Unlock()
	f.Res = new
}

func (f *FakePodMetricsClient) SetErr(new map[types.NamespacedName]error) {
	f.errMu.Lock()
	defer f.errMu.Unlock()
	f.Err = new
}

// SetPodMetrics sets metrics for a specific pod
func (f *FakePodMetricsClient) SetPodMetrics(podName types.NamespacedName, metrics *MetricsState) {
	f.resMu.Lock()
	defer f.resMu.Unlock()
	f.Res[podName] = metrics
}

// SetPodError sets an error for a specific pod
func (f *FakePodMetricsClient) SetPodError(podName types.NamespacedName, err error) {
	f.errMu.Lock()
	defer f.errMu.Unlock()
	f.Err[podName] = err
}

// ClearPodMetrics removes metrics for a specific pod
func (f *FakePodMetricsClient) ClearPodMetrics(podName types.NamespacedName) {
	f.resMu.Lock()
	defer f.resMu.Unlock()
	delete(f.Res, podName)
}

// ClearPodError removes error for a specific pod
func (f *FakePodMetricsClient) ClearPodError(podName types.NamespacedName) {
	f.errMu.Lock()
	defer f.errMu.Unlock()
	delete(f.Err, podName)
}