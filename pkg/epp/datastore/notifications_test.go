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

package datastore

import (
	"context"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer/endpoints"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer/mocks"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

func TestDatastoreRegisterObserverSuccess(t *testing.T) {
	ctx := context.Background()
	ds := NewDatastore(ctx, nil, 0)

	// Create an EndpointDataSource
	endpointDS := endpoints.NewEndpointDataSource("test-endpoint-source")

	// Register it as an observer
	endpointType := reflect.TypeOf((*fwkdl.Endpoint)(nil)).Elem()
	err := ds.RegisterObserver(endpointType, endpointDS)

	// Verify no error
	assert.NoError(t, err, "Should successfully register observer")
}

func TestDatastoreRegisterObserverTypeMismatch(t *testing.T) {
	ctx := context.Background()
	ds := NewDatastore(ctx, nil, 0)

	// Create a mock observer with wrong expected type
	wrongTypeObserver := &mockObserverWrongType{}

	// Try to register it
	endpointType := reflect.TypeOf((*fwkdl.Endpoint)(nil)).Elem()
	err := ds.RegisterObserver(endpointType, wrongTypeObserver)

	// Verify error is returned
	assert.Error(t, err, "Should reject observer with wrong expected type")
	assert.Contains(t, err.Error(), "expects", "Error should mention type expectation")
}

func TestDatastoreRegisterObserverDuplicate(t *testing.T) {
	ctx := context.Background()
	ds := NewDatastore(ctx, nil, 0)

	// Create an EndpointDataSource
	endpointDS := endpoints.NewEndpointDataSource("test-endpoint-source")

	// Register it first time
	endpointType := reflect.TypeOf((*fwkdl.Endpoint)(nil)).Elem()
	err := ds.RegisterObserver(endpointType, endpointDS)
	require.NoError(t, err, "First registration should succeed")

	// Try to register again
	err = ds.RegisterObserver(endpointType, endpointDS)

	// Verify error is returned
	assert.Error(t, err, "Should reject duplicate observer registration")
	assert.Contains(t, err.Error(), "already registered", "Error should mention already registered")
}

// mockObserverWrongType is a test observer that expects string instead of *corev1.Pod
type mockObserverWrongType struct{}

func (m *mockObserverWrongType) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "mock", Name: "wrong-type"}
}

func (m *mockObserverWrongType) Extractors() []string {
	return nil
}

func (m *mockObserverWrongType) AddExtractor(extractor fwkdl.Extractor) error {
	return nil
}

func (m *mockObserverWrongType) ExpectedEventDataType() reflect.Type {
	return reflect.TypeOf("") // Wrong type - expects string instead of *corev1.Pod
}

// MockReader is a simple mock client.Reader that returns empty pod lists
type MockReader struct{}

func (m *MockReader) Get(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
	return nil
}

func (m *MockReader) List(ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
	// Return empty list for pod resync
	if podList, ok := list.(*corev1.PodList); ok {
		podList.Items = []corev1.Pod{}
	}
	return nil
}

// MockEndpointFactory is a simple mock that creates real endpoints
type MockEndpointFactory struct{}

func NewMockEndpointFactory() *MockEndpointFactory {
	return &MockEndpointFactory{}
}

func (m *MockEndpointFactory) SetSources(sources []fwkdl.DataSource) {
	// No-op for mock
}

func (m *MockEndpointFactory) NewEndpoint(parent context.Context, metadata *fwkdl.EndpointMetadata, poolinfo datalayer.PoolInfo) fwkdl.Endpoint {
	// Return a real endpoint using the framework's NewEndpoint
	return fwkdl.NewEndpoint(metadata, nil)
}

func (m *MockEndpointFactory) ReleaseEndpoint(ep fwkdl.Endpoint) {
	// No-op for mock
}

// Helper function to create a test pod
func createNotificationTestPod(name, namespace, ip string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels: map[string]string{
				"app": "test-model",
			},
		},
		Status: corev1.PodStatus{
			PodIP: ip,
		},
	}
}

// Helper function to create a test endpoint pool
func createNotificationTestPool() *datalayer.EndpointPool {
	return &datalayer.EndpointPool{
		Name:        "test-pool",
		Namespace:   "default",
		Selector:    map[string]string{"app": "test-model"},
		TargetPorts: []int{8080},
	}
}

// setupDatastoreWithObserver creates a datastore with registered observer and mock extractor
func setupDatastoreWithObserver(t *testing.T, pool *datalayer.EndpointPool) (Datastore, *mocks.MockEndpointExtractor) {
	t.Helper()
	ctx := context.Background()
	mockFactory := NewMockEndpointFactory()
	ds := NewDatastore(ctx, mockFactory, 0)

	// Create EndpointDataSource with mock extractor
	endpointDS := endpoints.NewEndpointDataSource("test-endpoint-source")
	mockExtractor := mocks.NewMockEndpointExtractor("test-extractor")
	err := endpointDS.AddExtractor(mockExtractor)
	require.NoError(t, err, "Failed to add extractor")

	// Register the observer
	endpointType := reflect.TypeOf((*fwkdl.Endpoint)(nil)).Elem()
	err = ds.RegisterObserver(endpointType, endpointDS)
	require.NoError(t, err, "Failed to register observer")

	// Set the pool
	if pool != nil {
		mockReader := &MockReader{}
		err = ds.PoolSet(ctx, mockReader, pool)
		require.NoError(t, err, "Failed to set pool")
	}

	return ds, mockExtractor
}

func TestDatastorePodUpdateOrAddIfNotExistNotifiesObserver(t *testing.T) {
	ds, mockExtractor := setupDatastoreWithObserver(t, createNotificationTestPool())

	// Create a test pod
	testPod := createNotificationTestPod("test-pod", "default", "10.0.0.1")

	// Call PodUpdateOrAddIfNotExist
	isExisting := ds.PodUpdateOrAddIfNotExist(testPod)

	// Verify the pod was added (not existing)
	assert.False(t, isExisting, "Pod should be newly added")

	// Verify the extractor was called
	calls := mockExtractor.GetExtractCalls()
	require.Len(t, calls, 1, "Extractor should be called once")

	// Verify the notification details
	call := calls[0]
	assert.Equal(t, fwkdl.EventAddedOrModified, call.ChangeNotification.Type, "Event type should be AddedOrModified")

	// Verify the data is the pod
	pod, ok := call.ChangeNotification.Data.(*corev1.Pod)
	require.True(t, ok, "Event data should be *corev1.Pod")
	assert.Equal(t, "test-pod", pod.Name, "Pod name should match")
	assert.Equal(t, "default", pod.Namespace, "Pod namespace should match")

	// Verify the endpoint was created with correct metadata
	assert.NotNil(t, call.Endpoint, "Endpoint should not be nil")
	epMetadata := call.Endpoint.GetMetadata()
	assert.Equal(t, "test-pod-rank-0", epMetadata.GetNamespacedName().Name, "Endpoint name should include rank")
	assert.Equal(t, "default", epMetadata.GetNamespacedName().Namespace, "Endpoint namespace should match")
	assert.Equal(t, "10.0.0.1", epMetadata.Address, "Endpoint address should match pod IP")
}

func TestDatastorePodUpdateOrAddIfNotExistNotifiesObserverOnUpdate(t *testing.T) {
	ds, mockExtractor := setupDatastoreWithObserver(t, createNotificationTestPool())

	// Add pod first time
	testPod := createNotificationTestPod("test-pod", "default", "10.0.0.1")
	ds.PodUpdateOrAddIfNotExist(testPod)

	// Clear extractor calls
	mockExtractor.Reset()

	// Update the pod (same pod, potentially different state)
	updatedPod := createNotificationTestPod("test-pod", "default", "10.0.0.1")
	updatedPod.Labels["version"] = "v2"
	isExisting := ds.PodUpdateOrAddIfNotExist(updatedPod)

	// Verify the pod was existing
	assert.True(t, isExisting, "Pod should already exist")

	// Verify the extractor was called again for the update
	calls := mockExtractor.GetExtractCalls()
	require.Len(t, calls, 1, "Extractor should be called once for update")

	// Verify it's still an AddedOrModified event
	call := calls[0]
	assert.Equal(t, fwkdl.EventAddedOrModified, call.ChangeNotification.Type, "Event type should be AddedOrModified")
}

func TestDatastorePodDeleteNotifiesObserver(t *testing.T) {
	ds, mockExtractor := setupDatastoreWithObserver(t, createNotificationTestPool())

	// Add a pod first
	testPod := createNotificationTestPod("test-pod", "default", "10.0.0.1")
	ds.PodUpdateOrAddIfNotExist(testPod)

	// Clear extractor calls from the add operation
	mockExtractor.Reset()

	// Delete the pod
	ds.PodDelete("test-pod")

	// Verify the extractor was called for deletion
	calls := mockExtractor.GetExtractCalls()
	require.Len(t, calls, 1, "Extractor should be called once for deletion")

	// Verify the notification details
	call := calls[0]
	assert.Equal(t, fwkdl.EventDeleted, call.ChangeNotification.Type, "Event type should be Deleted")

	// Verify the data is a pod with deletion timestamp
	pod, ok := call.ChangeNotification.Data.(*corev1.Pod)
	require.True(t, ok, "Event data should be *corev1.Pod")
	assert.Equal(t, "test-pod", pod.Name, "Pod name should match")
	assert.NotNil(t, pod.DeletionTimestamp, "Pod should have deletion timestamp")

	// Verify the endpoint is provided
	assert.NotNil(t, call.Endpoint, "Endpoint should not be nil")

	// Verify the pod is removed from datastore
	pods := ds.PodList(func(ep fwkdl.Endpoint) bool {
		return ep.GetMetadata().PodName == "test-pod"
	})
	assert.Empty(t, pods, "Pod should be removed from datastore")
}

func TestDatastorePodDeleteMultipleEndpoints(t *testing.T) {
	// Set a pool with multiple target ports
	pool := &datalayer.EndpointPool{
		Name:        "test-pool",
		Namespace:   "default",
		Selector:    map[string]string{"app": "test-model"},
		TargetPorts: []int{8080, 8081}, // Multiple ports
	}
	ds, mockExtractor := setupDatastoreWithObserver(t, pool)

	// Add a pod (will create 2 endpoints due to 2 target ports)
	testPod := createNotificationTestPod("test-pod", "default", "10.0.0.1")
	ds.PodUpdateOrAddIfNotExist(testPod)

	// Clear extractor calls from the add operations
	mockExtractor.Reset()

	// Delete the pod
	ds.PodDelete("test-pod")

	// Verify the extractor was called twice (once for each endpoint)
	calls := mockExtractor.GetExtractCalls()
	assert.Len(t, calls, 2, "Extractor should be called twice for two endpoints")

	// Verify both are delete events
	for i, call := range calls {
		assert.Equal(t, fwkdl.EventDeleted, call.ChangeNotification.Type, "Event %d type should be Deleted", i)
		pod, ok := call.ChangeNotification.Data.(*corev1.Pod)
		require.True(t, ok, "Event %d data should be *corev1.Pod", i)
		assert.Equal(t, "test-pod", pod.Name, "Event %d pod name should match", i)
	}

	// Verify all endpoints are removed
	pods := ds.PodList(func(ep fwkdl.Endpoint) bool {
		return ep.GetMetadata().PodName == "test-pod"
	})
	assert.Empty(t, pods, "All endpoints should be removed from datastore")
}

func TestDatastoreNoObserverRegistered(t *testing.T) {
	ctx := context.Background()
	mockFactory := NewMockEndpointFactory()
	ds := NewDatastore(ctx, mockFactory, 0)

	// Set a pool (no observer registered)
	pool := createNotificationTestPool()
	mockReader := &MockReader{}
	err := ds.PoolSet(ctx, mockReader, pool)
	require.NoError(t, err, "Failed to set pool")

	// Add a pod without registering an observer - should not panic
	testPod := createNotificationTestPod("test-pod", "default", "10.0.0.1")
	assert.NotPanics(t, func() {
		ds.PodUpdateOrAddIfNotExist(testPod)
	}, "Should not panic when no observer is registered")

	// Delete a pod without observer - should not panic
	assert.NotPanics(t, func() {
		ds.PodDelete("test-pod")
	}, "Should not panic when no observer is registered")
}
func (m *mockObserverWrongType) AttachToStore(notifier fwkdl.StoreNotifier) error {
	return nil
}

func (m *mockObserverWrongType) Notify(ctx context.Context, event fwkdl.ChangeNotification, ep fwkdl.Endpoint) error {
	return nil
}
