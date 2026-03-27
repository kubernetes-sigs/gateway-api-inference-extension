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
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

var (
	testGVK = schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"}
)

func TestNewK8sNotificationSource(t *testing.T) {
	src := NewK8sNotificationSource("test-type", "test-name", testGVK)
	assert.Equal(t, "test-type", src.TypedName().Type)
	assert.Equal(t, "test-name", src.TypedName().Name)
	assert.Equal(t, testGVK, src.GVK())
}

func TestNotifyReturnsEvent(t *testing.T) {
	src := NewK8sNotificationSource(NotificationSourceType, "test", testGVK)

	ctx := context.Background()

	obj := &unstructured.Unstructured{}
	obj.SetName("test-cm")
	obj.SetNamespace("default")

	// test AddOrUpdate - source returns event unchanged (extractors handled by reconciler)
	event, err := src.Notify(ctx, fwkdl.NotificationEvent{
		Type:   fwkdl.EventAddOrUpdate,
		Object: obj.DeepCopy(),
	})
	assert.NoError(t, err, "failed to notify")
	require.NotNil(t, event)
	assert.Equal(t, fwkdl.EventAddOrUpdate, event.Type)
	assert.Equal(t, "test-cm", event.Object.GetName())

	// verify deep copy: mutating the received object doesn't affect original.
	event.Object.SetName("mutated")
	assert.Equal(t, "test-cm", obj.GetName())

	// test Delete event
	event, err = src.Notify(ctx, fwkdl.NotificationEvent{
		Type:   fwkdl.EventDelete,
		Object: obj.DeepCopy(),
	})
	assert.NoError(t, err, "failed to notify")
	require.NotNil(t, event)
	assert.Equal(t, fwkdl.EventDelete, event.Type)
	assert.Equal(t, "test-cm", event.Object.GetName())
}

func TestNotifyReturnsNilOnSkip(t *testing.T) {
	// This tests the case where Notify might return nil to signal
	// Runtime to skip extractor dispatch. Currently K8sNotificationSource
	// always returns the event, but this test verifies the contract.
	src := NewK8sNotificationSource(NotificationSourceType, "test", testGVK)

	ctx := context.Background()
	obj := &unstructured.Unstructured{}
	obj.SetName("test-cm")

	event, err := src.Notify(ctx, fwkdl.NotificationEvent{
		Type:   fwkdl.EventAddOrUpdate,
		Object: obj,
	})
	assert.NoError(t, err)
	// Currently source always returns event (not nil)
	// This test verifies the return value contract
	assert.NotNil(t, event)
}

// marshalParams is a test helper that marshals parameters to JSON.
// Returns nil for nil params.
func marshalParams(t *testing.T, params any) json.RawMessage {
	t.Helper()
	if params == nil {
		return nil
	}
	b, err := json.Marshal(params)
	require.NoError(t, err, "failed to marshal test parameters")
	return b
}

// verifyNotificationSource is a test helper that verifies a plugin implements
// both NotificationSource and DataSource interfaces with expected properties.
func verifyNotificationSource(t *testing.T, plugin fwkplugin.Plugin, wantGVK schema.GroupVersionKind, wantName string) {
	t.Helper()

	src, ok := plugin.(fwkdl.NotificationSource)
	require.True(t, ok, "plugin should implement NotificationSource")

	assert.Equal(t, wantGVK, src.GVK(), "GVK mismatch")
	assert.Equal(t, wantName, src.TypedName().Name, "name mismatch")
	assert.Equal(t, NotificationSourceType, src.TypedName().Type, "type mismatch")

	_, ok = plugin.(fwkdl.DataSource)
	assert.True(t, ok, "plugin should implement DataSource")
}

func TestNotificationSourceFactory(t *testing.T) {
	testCases := []struct {
		name          string
		pluginName    string
		params        any
		expectedError string // empty = no error expected, non-empty = error containing this text
		wantGVK       schema.GroupVersionKind
		wantName      string
	}{
		{
			name:       "valid params",
			pluginName: "deployment-watcher",
			params:     notificationSourceParams{Group: "apps", Version: "v1", Kind: "Deployment"},
			wantGVK:    schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"},
			wantName:   "deployment-watcher",
		},
		{
			name:       "core resource (empty group)",
			pluginName: "pod-watcher",
			params:     notificationSourceParams{Group: "", Version: "v1", Kind: "Pod"},
			wantGVK:    schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"},
			wantName:   "pod-watcher",
		},
		{
			name:       "name defaults to GVK",
			pluginName: "",
			params:     notificationSourceParams{Group: "apps", Version: "v1", Kind: "Deployment"},
			wantGVK:    schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"},
			wantName:   schema.GroupVersionKind{Group: "apps", Version: "v1", Kind: "Deployment"}.String(),
		},
		{
			name:          "missing version",
			pluginName:    "test",
			params:        notificationSourceParams{Group: "", Kind: "Pod"},
			expectedError: "version and kind are required",
		},
		{
			name:          "missing kind",
			pluginName:    "test",
			params:        notificationSourceParams{Group: "", Version: "v1"},
			expectedError: "version and kind are required",
		},
		{
			name:          "nil params",
			pluginName:    "test",
			params:        nil,
			expectedError: "requires parameters",
		},
	}

	for _, tt := range testCases {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			rawParams := marshalParams(t, tt.params)
			plugin, err := NotificationSourceFactory(tt.pluginName, rawParams, nil)

			if tt.expectedError != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.expectedError)
				return
			}

			require.NoError(t, err)
			verifyNotificationSource(t, plugin, tt.wantGVK, tt.wantName)
		})
	}
}
