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

// EndpointMetaState allows management of the EndpointMetadata related attributes.
type EndpointMetaState interface {
	GetMetadata() *EndpointMetadata
	UpdateMetadata(*EndpointMetadata)
	GetAttributes() *Attributes
}

// EndpointMetricsState allows management of the Metrics related attributes.
type EndpointMetricsState interface {
	GetMetrics() *Metrics
	UpdateMetrics(*Metrics)
}

// Endpoint represents an inference serving endpoint and its related attributes.
type Endpoint interface {
	fmt.Stringer
	EndpointMetaState
	EndpointMetricsState
	AttributeMap
}

// ModelServer is an implementation of the Endpoint interface.
type ModelServer struct {
	pod        atomic.Pointer[EndpointMetadata]
	metrics    atomic.Pointer[Metrics]
	attributes *Attributes
}

// NewEndpoint returns a new ModelServer with the given EndpointMetadata and Metrics.
func NewEndpoint(meta *EndpointMetadata, metrics *Metrics) *ModelServer {
	if meta == nil {
		meta = &EndpointMetadata{}
	}
	if metrics == nil {
		metrics = NewMetrics()
	}
	ep := &ModelServer{
		attributes: NewAttributes(),
	}
	ep.UpdateMetadata(meta)
	ep.UpdateMetrics(metrics)
	return ep
}

// String returns a representation of the ModelServer. For brevity, only names of
// extended attributes are returned and not their values.
func (srv *ModelServer) String() string {
	return fmt.Sprintf("Metadata: %v; Metrics: %v; Attributes: %v", srv.GetMetadata(), srv.GetMetrics(), srv.Keys())
}

func (srv *ModelServer) GetMetadata() *EndpointMetadata {
	return srv.pod.Load()
}

func (srv *ModelServer) UpdateMetadata(pod *EndpointMetadata) {
	srv.pod.Store(pod)
}

func (srv *ModelServer) GetMetrics() *Metrics {
	return srv.metrics.Load()
}

func (srv *ModelServer) UpdateMetrics(metrics *Metrics) {
	srv.metrics.Store(metrics)
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

func (srv *ModelServer) GetAttributes() *Attributes {
	return srv.attributes
}

func (srv *ModelServer) Clone() *ModelServer {
	clone := &ModelServer{
		attributes: srv.attributes.Clone(),
	}
	clone.pod.Store(srv.pod.Load().Clone())
	clone.metrics.Store(srv.metrics.Load().Clone())
	return clone
}
