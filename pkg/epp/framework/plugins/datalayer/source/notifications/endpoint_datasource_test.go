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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

func verifyEndpointSource(t *testing.T, plugin fwkplugin.Plugin, wantName string) {
	t.Helper()

	src, ok := plugin.(fwkdl.EndpointSource)
	require.True(t, ok, "plugin should implement EndpointSource")
	assert.Equal(t, wantName, src.TypedName().Name, "name mismatch")
	assert.Equal(t, EndpointNotificationSourceType, src.TypedName().Type, "type mismatch")

	_, ok = plugin.(fwkdl.DataSource)
	assert.True(t, ok, "plugin should implement DataSource")
}

func TestNewEndpointDataSource(t *testing.T) {
	src := NewEndpointDataSource("test-type", "test-name")
	assert.Equal(t, "test-type", src.TypedName().Type)
	assert.Equal(t, "test-name", src.TypedName().Name)
	assert.Equal(t, fwkdl.EndpointEventReflectType, src.OutputType())
	assert.Equal(t, fwkdl.EndpointExtractorType, src.ExtractorType())
}

func TestEndpointNotifyReturnsEvent(t *testing.T) {
	src := NewEndpointDataSource(EndpointNotificationSourceType, "test")
	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{}, nil)

	event, err := src.NotifyEndpoint(context.Background(), fwkdl.EndpointEvent{
		Type:     fwkdl.EventAddOrUpdate,
		Endpoint: ep,
	})

	require.NoError(t, err)
	require.NotNil(t, event)
	assert.Equal(t, fwkdl.EventAddOrUpdate, event.Type)
	assert.Equal(t, ep, event.Endpoint)

	event, err = src.NotifyEndpoint(context.Background(), fwkdl.EndpointEvent{
		Type:     fwkdl.EventDelete,
		Endpoint: ep,
	})

	require.NoError(t, err)
	require.NotNil(t, event)
	assert.Equal(t, fwkdl.EventDelete, event.Type)
}

func TestEndpointSourceFactory(t *testing.T) {
	testCases := []struct {
		name       string
		pluginName string
		wantName   string
	}{
		{
			name:       "explicit name",
			pluginName: "my-endpoint-source",
			wantName:   "my-endpoint-source",
		},
		{
			name:       "name defaults to EndpointNotificationSourceType",
			pluginName: "",
			wantName:   EndpointNotificationSourceType,
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			plugin, err := EndpointSourceFactory(tt.pluginName, nil, nil)
			require.NoError(t, err)
			verifyEndpointSource(t, plugin, tt.wantName)
		})
	}
}
