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

	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// FakePodMetrics is an implementation of PodMetrics that doesn't run the async refresh loop.
type FakePodMetrics struct {
	Pod             *backend.Pod
	Metrics         *MetricsState
	runningRequests *datalayer.RequestPriorityQueue
	stopped         bool
	mu              sync.RWMutex // Protect the stopped field and operations
}

func (fpm *FakePodMetrics) String() string {
	return fmt.Sprintf("Pod: %v; Metrics: %v", fpm.GetPod(), fpm.GetMetrics())
}

func (fpm *FakePodMetrics) GetPod() *backend.Pod {
	return fpm.Pod
}

func (fpm *FakePodMetrics) GetMetrics() *MetricsState {
	return fpm.Metrics
}

func (fpm *FakePodMetrics) UpdatePod(pod *corev1.Pod) {
	fpm.Pod = toInternalPod(pod, nil)
}

func (f *FakePodMetrics) StopRefreshLoop() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.stopped = true
}

func (f *FakePodMetrics) GetRunningRequests() *datalayer.RequestPriorityQueue {
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
	pod := f.GetPod()
	if pod == nil || pod.RunningRequests == nil {
		return false
	}
	return pod.RunningRequests.Contains(requestID)
}

func (srv *FakePodMetrics) PeekRequestPriorityQueue() *datalayer.Request {
	pod := srv.GetPod()
	if pod == nil || pod.RunningRequests == nil {
		return nil
	}
	return pod.RunningRequests.Peek()
}

func NewFakePodMetrics(k8sPod *corev1.Pod) *FakePodMetrics {
	labels := make(map[string]string)
	for k, v := range k8sPod.Labels {
		labels[k] = v
	}

	pod := &backend.Pod{
		NamespacedName: types.NamespacedName{
			Name:      k8sPod.Name,
			Namespace: k8sPod.Namespace,
		},
		Address:         k8sPod.Status.PodIP,
		Labels:          labels,
		RunningRequests: datalayer.NewRequestPriorityQueue(),
	}

	return &FakePodMetrics{
		Pod:             pod,
		Metrics:         &MetricsState{UpdateTime: time.Now()},
		runningRequests: datalayer.NewRequestPriorityQueue(),
		stopped:         false,
	}
}

func (*FakePodMetrics) Put(string, datalayer.Cloneable)        {}
func (*FakePodMetrics) Get(string) (datalayer.Cloneable, bool) { return nil, false }
func (*FakePodMetrics) Keys() []string                         { return nil }

func (fpm *FakePodMetrics) UpdateMetrics(updated *MetricsState) {
	updated.UpdateTime = time.Now()
	fpm.Metrics = updated
}

type FakePodMetricsClient struct {
	errMu sync.RWMutex
	Err   map[types.NamespacedName]error
	resMu sync.RWMutex
	Res   map[types.NamespacedName]*MetricsState
}

func (f *FakePodMetricsClient) FetchMetrics(ctx context.Context, pod *backend.Pod, existing *MetricsState) (*MetricsState, error) {
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
		return nil, fmt.Errorf("no pod found: %v", pod.NamespacedName)
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
