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
	Pod        *backend.Pod
	Metrics    *MetricsState
	Attributes *datalayer.Attributes
}

func (fpm *FakePodMetrics) String() string {
	return fmt.Sprintf("Metadata: %v; Metrics: %v", fpm.GetMetadata(), fpm.GetMetrics())
}

func (fpm *FakePodMetrics) GetMetadata() *backend.Pod {
	return fpm.Pod
}

func (fpm *FakePodMetrics) GetMetrics() *MetricsState {
	return fpm.Metrics
}

func (fpm *FakePodMetrics) UpdateMetadata(metadata *datalayer.EndpointMetadata) {
	fpm.Pod = metadata
}
func (fpm *FakePodMetrics) GetAttributes() *datalayer.Attributes {
	return fpm.Attributes
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
