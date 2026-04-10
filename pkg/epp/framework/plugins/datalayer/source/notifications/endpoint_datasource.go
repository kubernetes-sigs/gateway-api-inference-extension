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

package notifications

import (
	"context"
	"reflect"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

var (
	_ fwkdl.DataSource     = (*EndpointDataSource)(nil)
	_ fwkdl.EndpointSource = (*EndpointDataSource)(nil)
)

// EndpointNotificationSourceType is the plugin type identifier for endpoint notification sources.
const EndpointNotificationSourceType = "endpoint-notification-source"

// EndpointDataSource is an EndpointSource that passes endpoint lifecycle events
// through to registered EndpointExtractors without modification.
type EndpointDataSource struct {
	typedName fwkplugin.TypedName
}

// NewEndpointDataSource returns a new EndpointDataSource with the given plugin type and name.
func NewEndpointDataSource(pluginType, pluginName string) *EndpointDataSource {
	return &EndpointDataSource{
		typedName: fwkplugin.TypedName{Type: pluginType, Name: pluginName},
	}
}

// TypedName returns the plugin type and name.
func (s *EndpointDataSource) TypedName() fwkplugin.TypedName {
	return s.typedName
}

// OutputType returns the type of data this DataSource produces (EndpointEvent).
func (s *EndpointDataSource) OutputType() reflect.Type {
	return fwkdl.EndpointEventReflectType
}

// ExtractorType returns the type of Extractor this DataSource expects (EndpointExtractor).
func (s *EndpointDataSource) ExtractorType() reflect.Type {
	return fwkdl.EndpointExtractorType
}

// NotifyEndpoint passes the event through unchanged for the Runtime to dispatch to extractors.
func (s *EndpointDataSource) NotifyEndpoint(_ context.Context, event fwkdl.EndpointEvent) (*fwkdl.EndpointEvent, error) {
	return &event, nil
}
