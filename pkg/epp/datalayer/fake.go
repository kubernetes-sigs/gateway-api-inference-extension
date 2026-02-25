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
	"context"
	"reflect"
	"sync"
	"sync/atomic"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

const (
	fakeSource = "fake-data-source"
)

type FakeDataSource struct {
	typedName *plugin.TypedName
	callCount int64
	Metrics   map[types.NamespacedName]*fwkdl.Metrics
	errMu     sync.RWMutex
	Errors    map[types.NamespacedName]error
}

func (fds *FakeDataSource) TypedName() plugin.TypedName {
	if fds.typedName != nil {
		return *fds.typedName
	}
	return plugin.TypedName{
		Type: fakeSource,
		Name: fakeSource,
	}
}

func (fds *FakeDataSource) OutputType() reflect.Type {
	return reflect.TypeOf(fwkdl.Metrics{})
}

func (fds *FakeDataSource) ExtractorType() reflect.Type {
	return reflect.TypeOf((*fwkdl.Extractor)(nil)).Elem()
}

func (fds *FakeDataSource) Extractors() []string                 { return []string{} }
func (fds *FakeDataSource) AddExtractor(_ fwkdl.Extractor) error { return nil }

func (fds *FakeDataSource) Collect(ctx context.Context, ep fwkdl.Endpoint) error {
	atomic.AddInt64(&fds.callCount, 1)
	key := ep.GetMetadata().Clone().NamespacedName
	fds.errMu.RLock()
	err, hasErr := fds.Errors[key]
	fds.errMu.RUnlock()
	if hasErr {
		return err
	}
	if metrics, ok := fds.Metrics[key]; ok {
		ep.UpdateMetrics(metrics)
	}
	return nil
}

// SetErrors sets the error map for the fake data source (thread-safe).
func (fds *FakeDataSource) SetErrors(errors map[types.NamespacedName]error) {
	fds.errMu.Lock()
	defer fds.errMu.Unlock()
	fds.Errors = errors
}

// FakeNotificationSource implements both DataSource and NotificationSource for testing.
type FakeNotificationSource struct {
	typedName plugin.TypedName
	gvk       schema.GroupVersionKind
}

func (m *FakeNotificationSource) TypedName() plugin.TypedName {
	return m.typedName
}

func (m *FakeNotificationSource) OutputType() reflect.Type {
	return reflect.TypeOf(fwkdl.NotificationEvent{})
}

func (m *FakeNotificationSource) ExtractorType() reflect.Type {
	return reflect.TypeOf((*fwkdl.NotificationExtractor)(nil)).Elem()
}

func (m *FakeNotificationSource) GVK() schema.GroupVersionKind {
	return m.gvk
}

func (m *FakeNotificationSource) Notify(_ context.Context, _ fwkdl.NotificationEvent) error {
	return nil
}

func (m *FakeNotificationSource) Extractors() []string {
	return []string{}
}

func (m *FakeNotificationSource) AddExtractor(_ fwkdl.Extractor) error {
	return nil
}

func (m *FakeNotificationSource) Collect(_ context.Context, _ fwkdl.Endpoint) error {
	return nil
}
