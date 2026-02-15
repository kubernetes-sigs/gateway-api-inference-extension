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
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/extractor/mocks"
)

var (
	testGVK = schema.GroupVersionKind{Group: "", Version: "v1", Kind: "ConfigMap"}
)

// plainExtractor implements Extractor but NOT NotificationExtractor.
type plainExtractor struct{}

func (p *plainExtractor) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "plain", Name: "plain"}
}

func (p *plainExtractor) ExpectedInputType() reflect.Type {
	return nil
}

func (p *plainExtractor) Extract(_ context.Context, _ any, _ fwkdl.Endpoint) error {
	return nil
}

func TestNewK8sNotificationSource(t *testing.T) {
	src := NewK8sNotificationSource("test-type", "test-name", testGVK)
	assert.Equal(t, "test-type", src.TypedName().Type)
	assert.Equal(t, "test-name", src.TypedName().Name)
	assert.Equal(t, testGVK, src.GVK())
}

func TestAddExtractor(t *testing.T) {
	src := NewK8sNotificationSource(NotificationSourceType, "test", testGVK)

	ext1 := mocks.NewNotificationExtractor("ext1")
	ext2 := mocks.NewNotificationExtractor("ext2")

	require.NoError(t, src.AddExtractor(ext1))
	require.NoError(t, src.AddExtractor(ext2))

	err := src.AddExtractor(ext1) // error on duplicate
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "duplicate")

	err = src.AddExtractor(nil) // error on nil
	assert.Error(t, err)

	names := src.Extractors()
	assert.Len(t, names, 2)
}

func TestAddExtractorWrongType(t *testing.T) {
	src := NewK8sNotificationSource(NotificationSourceType, "test", testGVK)
	err := src.AddExtractor(&plainExtractor{})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "NotificationExtractor")
}

func TestNotify(t *testing.T) {
	src := NewK8sNotificationSource(NotificationSourceType, "test", testGVK)
	ext := mocks.NewNotificationExtractor("ext1")
	_ = src.AddExtractor(ext)

	ctx := context.Background()

	obj := &unstructured.Unstructured{}
	obj.SetName("test-cm")
	obj.SetNamespace("default")

	// test AddOrUpdate mutation
	err := src.Notify(ctx, fwkdl.NotificationEvent{
		Type:   fwkdl.EventAddOrUpdate,
		Object: obj.DeepCopy(),
	})
	assert.NoError(t, err, "failed to notify")
	events := ext.GetEvents()
	require.Len(t, events, 1)
	assert.Equal(t, fwkdl.EventAddOrUpdate, events[0].Type)
	assert.Equal(t, "test-cm", events[0].Object.GetName())

	// verify deep copy: mutating the received object doesn't affect original.
	events[0].Object.SetName("mutated")
	assert.Equal(t, "test-cm", obj.GetName())

	// test Delete event.
	err = src.Notify(ctx, fwkdl.NotificationEvent{
		Type:   fwkdl.EventDelete,
		Object: obj.DeepCopy(),
	})
	assert.NoError(t, err, "failed to notify")
	events = ext.GetEvents()
	require.Len(t, events, 2)
	assert.Equal(t, fwkdl.EventDelete, events[1].Type)
	assert.Equal(t, "test-cm", events[1].Object.GetName())
}

func TestNotifyMultipleExtractors(t *testing.T) {
	src := NewK8sNotificationSource(NotificationSourceType, "test", testGVK)
	ext1 := mocks.NewNotificationExtractor("ext1")
	ext2 := mocks.NewNotificationExtractor("ext2")
	_ = src.AddExtractor(ext1)
	_ = src.AddExtractor(ext2)

	obj := &unstructured.Unstructured{}
	obj.SetName("cm1")

	err := src.Notify(context.Background(), fwkdl.NotificationEvent{
		Type:   fwkdl.EventAddOrUpdate,
		Object: obj,
	})
	assert.NoError(t, err, "failed to notify")
	assert.Len(t, ext1.GetEvents(), 1)
	assert.Len(t, ext2.GetEvents(), 1)
}

// marshalParams is a test helper that marshals parameters to JSON.
// Returns nil for nil params.
func marshalParams(t *testing.T, params interface{}) json.RawMessage {
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
		params        interface{}
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
			pluginName: "cm-watcher",
			params:     notificationSourceParams{Group: "", Version: "v1", Kind: "ConfigMap"},
			wantGVK:    schema.GroupVersionKind{Group: "", Version: "v1", Kind: "ConfigMap"},
			wantName:   "cm-watcher",
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
			params:        notificationSourceParams{Group: "", Kind: "ConfigMap"},
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
