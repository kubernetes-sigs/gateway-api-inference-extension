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
	"sync/atomic"

	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
)

const (
	fakeSource = "fake-data-source"
)

type FakeDataSource struct {
	typedName *plugins.TypedName
	callCount int64
	Metrics   map[types.NamespacedName]*Metrics
	Errors    map[types.NamespacedName]error
}

func (fds *FakeDataSource) TypedName() plugins.TypedName {
	if fds.typedName != nil {
		return *fds.typedName
	}
	return plugins.TypedName{
		Type: fakeSource,
		Name: fakeSource,
	}
}
func (fds *FakeDataSource) Extractors() []string           { return []string{} }
func (fds *FakeDataSource) AddExtractor(_ Extractor) error { return nil }

func (fds *FakeDataSource) Collect(ctx context.Context, ep Endpoint) error {
	atomic.AddInt64(&fds.callCount, 1)
	if metrics, ok := fds.Metrics[ep.GetMetadata().Clone().NamespacedName]; ok {
		if _, ok := fds.Errors[ep.GetMetadata().Clone().NamespacedName]; !ok {
			ep.UpdateMetrics(metrics)
		}
	}
	return nil
}
