/*
Copyright 2026 The Kubernetes Authors.

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
	"sync"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

var _ fwkdl.EndpointExtractor = (*EndpointExtractor)(nil)

// EndpointExtractor is a mock EndpointExtractor for testing.
// It records all events it receives and provides helper methods for test assertions.
type EndpointExtractor struct {
	name       string
	events     []fwkdl.EndpointEvent
	mu         sync.Mutex
	extractErr error
}

// NewEndpointExtractor creates a new mock EndpointExtractor with the given name.
func NewEndpointExtractor(name string) *EndpointExtractor {
	return &EndpointExtractor{name: name}
}

// WithExtractError configures the extractor to return an error on ExtractEndpoint.
func (m *EndpointExtractor) WithExtractError(err error) *EndpointExtractor {
	m.extractErr = err
	return m
}

func (m *EndpointExtractor) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "mock-endpoint-extractor", Name: m.name}
}

func (m *EndpointExtractor) ExpectedInputType() reflect.Type {
	return fwkdl.EndpointEventReflectType
}

// Extract is the base Extractor interface method — no-op for endpoint extractors.
func (m *EndpointExtractor) Extract(_ context.Context, _ any, _ fwkdl.Endpoint) error {
	return nil
}

// ExtractEndpoint records the event and returns any configured error.
func (m *EndpointExtractor) ExtractEndpoint(_ context.Context, event fwkdl.EndpointEvent) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.events = append(m.events, event)
	return m.extractErr
}

// GetEvents returns a copy of all recorded events.
func (m *EndpointExtractor) GetEvents() []fwkdl.EndpointEvent {
	m.mu.Lock()
	defer m.mu.Unlock()
	result := make([]fwkdl.EndpointEvent, len(m.events))
	copy(result, m.events)
	return result
}

// Reset clears all recorded events.
func (m *EndpointExtractor) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.events = nil
}
