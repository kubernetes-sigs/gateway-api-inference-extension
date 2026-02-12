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
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

var (
	testGVK = schema.GroupVersionKind{Group: "", Version: "v1", Kind: "ConfigMap"}
)

// mockNotificationExtractor implements both Extractor and NotificationExtractor.
type mockNotificationExtractor struct {
	name       string
	events     []fwkdl.NotificationEvent
	mu         sync.Mutex
	extractErr error
}

func (m *mockNotificationExtractor) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "mock-extractor", Name: m.name}
}

func (m *mockNotificationExtractor) ExpectedInputType() reflect.Type {
	return reflect.TypeOf(unstructured.Unstructured{})
}

// Extract is the Extractor interface method — no-op for notification extractors.
func (m *mockNotificationExtractor) Extract(_ context.Context, _ any, _ fwkdl.Endpoint) error {
	return nil
}

// ExtractNotification is the NotificationExtractor method — the real work.
func (m *mockNotificationExtractor) ExtractNotification(_ context.Context, event fwkdl.NotificationEvent) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.events = append(m.events, event)
	return m.extractErr
}

func (m *mockNotificationExtractor) getEvents() []fwkdl.NotificationEvent {
	m.mu.Lock()
	defer m.mu.Unlock()
	result := make([]fwkdl.NotificationEvent, len(m.events))
	copy(result, m.events)
	return result
}

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

	ext1 := &mockNotificationExtractor{name: "ext1"}
	ext2 := &mockNotificationExtractor{name: "ext2"}

	require.NoError(t, src.AddExtractor(ext1))
	require.NoError(t, src.AddExtractor(ext2))

	// Duplicate.
	err := src.AddExtractor(ext1)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "duplicate")

	// Nil.
	err = src.AddExtractor(nil)
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
	ext := &mockNotificationExtractor{name: "ext1"}
	_ = src.AddExtractor(ext)

	ctx := context.Background()

	obj := &unstructured.Unstructured{}
	obj.SetName("test-cm")
	obj.SetNamespace("default")

	// Test AddOrUpdate event.
	err := src.Notify(ctx, fwkdl.NotificationEvent{
		Type:   fwkdl.EventAddOrUpdate,
		Object: obj.DeepCopy(),
	})
	assert.NoError(t, err, "failed to notify")
	events := ext.getEvents()
	require.Len(t, events, 1)
	assert.Equal(t, fwkdl.EventAddOrUpdate, events[0].Type)
	assert.Equal(t, "test-cm", events[0].Object.GetName())

	// Verify deep copy: mutating the received object doesn't affect original.
	events[0].Object.SetName("mutated")
	assert.Equal(t, "test-cm", obj.GetName())

	// Test Delete event.
	err = src.Notify(ctx, fwkdl.NotificationEvent{
		Type:   fwkdl.EventDelete,
		Object: obj.DeepCopy(),
	})
	assert.NoError(t, err, "failed to notify")
	events = ext.getEvents()
	require.Len(t, events, 2)
	assert.Equal(t, fwkdl.EventDelete, events[1].Type)
	assert.Equal(t, "test-cm", events[1].Object.GetName())
}

func TestNotifyMultipleExtractors(t *testing.T) {
	src := NewK8sNotificationSource(NotificationSourceType, "test", testGVK)
	ext1 := &mockNotificationExtractor{name: "ext1"}
	ext2 := &mockNotificationExtractor{name: "ext2"}
	_ = src.AddExtractor(ext1)
	_ = src.AddExtractor(ext2)

	obj := &unstructured.Unstructured{}
	obj.SetName("cm1")

	err := src.Notify(context.Background(), fwkdl.NotificationEvent{
		Type:   fwkdl.EventAddOrUpdate,
		Object: obj,
	})
	assert.NoError(t, err, "failed to notify")
	assert.Len(t, ext1.getEvents(), 1)
	assert.Len(t, ext2.getEvents(), 1)
}

func TestNotificationSourceFactory(t *testing.T) {
	tests := []struct {
		name       string
		pluginName string
		params     interface{}
		wantErr    bool
		wantGVK    schema.GroupVersionKind
		wantName   string
		errContain string
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
			name:       "missing version",
			pluginName: "test",
			params:     notificationSourceParams{Group: "", Kind: "ConfigMap"},
			wantErr:    true,
			errContain: "version and kind are required",
		},
		{
			name:       "missing kind",
			pluginName: "test",
			params:     notificationSourceParams{Group: "", Version: "v1"},
			wantErr:    true,
			errContain: "version and kind are required",
		},
		{
			name:       "nil params",
			pluginName: "test",
			params:     nil,
			wantErr:    true,
			errContain: "requires parameters",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var rawParams json.RawMessage
			if tt.params != nil {
				b, err := json.Marshal(tt.params)
				require.NoError(t, err)
				rawParams = b
			}

			plugin, err := NotificationSourceFactory(tt.pluginName, rawParams, nil)
			if tt.wantErr {
				assert.Error(t, err)
				if tt.errContain != "" {
					assert.Contains(t, err.Error(), tt.errContain)
				}
				return
			}

			require.NoError(t, err)
			src, ok := plugin.(fwkdl.NotificationSource)
			require.True(t, ok, "should implement NotificationSource")
			assert.Equal(t, tt.wantGVK, src.GVK())
			assert.Equal(t, tt.wantName, src.TypedName().Name)
			assert.Equal(t, NotificationSourceType, src.TypedName().Type)

			// Should also satisfy DataSource.
			_, ok = plugin.(fwkdl.DataSource)
			assert.True(t, ok, "should implement DataSource")
		})
	}
}
