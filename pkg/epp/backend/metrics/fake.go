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

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// FakePodMetrics is an implementation of PodMetrics that doesn't run the async refresh loop.
type FakePodMetrics struct {
	Pod     *Pod
	Metrics *Metrics
}

func (fpm *FakePodMetrics) String() string {
	return fmt.Sprintf("Pod: %v; Metrics: %v", fpm.GetPod(), fpm.GetMetrics())
}

func (fpm *FakePodMetrics) GetPod() *Pod {
	return fpm.Pod
}
func (fpm *FakePodMetrics) GetMetrics() *Metrics {
	return fpm.Metrics
}
func (fpm *FakePodMetrics) UpdatePod(pod *corev1.Pod) {
	fpm.Pod = toInternalPod(pod)
}
func (fpm *FakePodMetrics) StopRefreshLoop() {} // noop

type FakePodMetricsClient struct {
	errMu sync.RWMutex
	Err   map[types.NamespacedName]error
	resMu sync.RWMutex
	Res   map[types.NamespacedName]*Metrics
}

func (f *FakePodMetricsClient) FetchMetrics(ctx context.Context, pod *Pod, existing *Metrics, port int32) (*Metrics, error) {
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

func (f *FakePodMetricsClient) SetRes(new map[types.NamespacedName]*Metrics) {
	f.resMu.Lock()
	defer f.resMu.Unlock()
	f.Res = new
}

func (f *FakePodMetricsClient) SetErr(new map[types.NamespacedName]error) {
	f.errMu.Lock()
	defer f.errMu.Unlock()
	f.Err = new
}
