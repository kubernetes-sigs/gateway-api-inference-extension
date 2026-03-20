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

package mocks

import (
	"context"
	"reflect"
	"sync/atomic"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

var _ fwkdl.DataSource = (*MetricsDataSource)(nil)
var _ fwkdl.PollingDataSource = (*MetricsDataSource)(nil)

type MetricsDataSource struct {
	typedName plugin.TypedName
	CallCount int64
	Metrics   map[types.NamespacedName]*fwkdl.Metrics
	Errors    map[types.NamespacedName]error
}

func NewDataSource(typedName plugin.TypedName) *MetricsDataSource {
	return &MetricsDataSource{typedName: typedName}
}

func (fds *MetricsDataSource) TypedName() plugin.TypedName {
	return fds.typedName
}

func (fds *MetricsDataSource) OutputType() reflect.Type {
	return reflect.TypeOf(fwkdl.Metrics{})
}

func (fds *MetricsDataSource) ExtractorType() reflect.Type {
	return reflect.TypeOf((*fwkdl.Extractor)(nil)).Elem()
}

func (fds *MetricsDataSource) Extractors() []string                 { return []string{} }
func (fds *MetricsDataSource) AddExtractor(_ fwkdl.Extractor) error { return nil }

func (fds *MetricsDataSource) Poll(ctx context.Context, ep fwkdl.Endpoint) (any, error) {
	atomic.AddInt64(&fds.CallCount, 1)
	if metrics, ok := fds.Metrics[ep.GetMetadata().Clone().NamespacedName]; ok {
		if _, ok := fds.Errors[ep.GetMetadata().Clone().NamespacedName]; !ok {
			ep.UpdateMetrics(metrics)
		}
	}
	return nil, nil
}

// NotificationSource implements both DataSource and NotificationSource for testing.
type NotificationSource struct {
	typedName plugin.TypedName
	gvk       schema.GroupVersionKind
}

func NewNotificationSource(pluginType, name string, gvk schema.GroupVersionKind) *NotificationSource {
	return &NotificationSource{
		typedName: plugin.TypedName{Type: pluginType, Name: name},
		gvk:       gvk,
	}
}

func (m *NotificationSource) TypedName() plugin.TypedName {
	return m.typedName
}

func (m *NotificationSource) OutputType() reflect.Type {
	return reflect.TypeOf(fwkdl.NotificationEvent{})
}

func (m *NotificationSource) ExtractorType() reflect.Type {
	return reflect.TypeOf((*fwkdl.NotificationExtractor)(nil)).Elem()
}

func (m *NotificationSource) GVK() schema.GroupVersionKind {
	return m.gvk
}

func (m *NotificationSource) Notify(_ context.Context, event fwkdl.NotificationEvent) (*fwkdl.NotificationEvent, error) {
	return &event, nil
}

func (m *NotificationSource) Extractors() []string {
	return []string{}
}

func (m *NotificationSource) AddExtractor(_ fwkdl.Extractor) error {
	return nil
}

func (m *NotificationSource) Collect(_ context.Context, _ fwkdl.Endpoint) error {
	return nil
}
