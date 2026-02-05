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

package endpoints

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer/mocks"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

func TestNewEndpointDataSource(t *testing.T) {
	tests := []struct {
		name         string
		inputName    string
		expectedName string
		expectedType string
	}{
		{
			name:         "default name when empty string provided",
			inputName:    "",
			expectedName: EndpointsDataSourceType,
			expectedType: EndpointsDataSourceType,
		},
		{
			name:         "custom name when provided",
			inputName:    "custom-endpoint-source",
			expectedName: "custom-endpoint-source",
			expectedType: EndpointsDataSourceType,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ds := NewEndpointDataSource(tt.inputName)

			// Verify data source is not nil
			assert.NotNil(t, ds, "NewEndpointDataSource should not return nil")

			// Verify TypedName
			typedName := ds.TypedName()
			assert.Equal(t, tt.expectedType, typedName.Type, "Type should match")
			assert.Equal(t, tt.expectedName, typedName.Name, "Name should match")

			// Verify initial state has no extractors
			extractors := ds.Extractors()
			assert.Empty(t, extractors, "New data source should have no extractors")
		})
	}
}

func TestEndpointDataSourceExpectedEventDataType(t *testing.T) {
	ds := NewEndpointDataSource("test-source")

	expectedType := ds.ExpectedEventDataType()

	// Verify it returns *corev1.Pod type
	assert.NotNil(t, expectedType, "ExpectedEventDataType should not return nil")
	assert.Equal(t, "*v1.Pod", expectedType.String(), "Should expect *corev1.Pod type")
}

// setupDataSourceWithExtractor creates a data source and adds an extractor to it.
// Uses require to fail fast if setup fails.
func setupDataSourceWithExtractor(t *testing.T, dsName, extractorName string) (*EndpointDataSource, *mocks.MockEndpointExtractor) {
	t.Helper()
	ds := NewEndpointDataSource(dsName)
	extractor := mocks.NewMockEndpointExtractor(extractorName)
	err := ds.AddExtractor(extractor)
	require.NoError(t, err, "Failed to add extractor during setup")
	return ds, extractor
}

// createTestPod creates a test Pod with the given name and namespace
func createTestPod(name, namespace, ip string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Status: corev1.PodStatus{
			PodIP: ip,
		},
	}
}

// createTestEndpoint creates a test endpoint with the given name and namespace
func createTestEndpoint(name, namespace string) fwkdl.Endpoint {
	return fwkdl.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: types.NamespacedName{
				Name:      name,
				Namespace: namespace,
			},
		},
		nil,
	)
}

func TestEndpointDataSourceAddExtractorSuccess(t *testing.T) {
	ds, extractor := setupDataSourceWithExtractor(t, "test-source", "test-extractor")

	// Verify extractor is in the list
	extractors := ds.Extractors()
	assert.Len(t, extractors, 1, "Should have one extractor")
	assert.Contains(t, extractors, extractor.TypedName().String(), "Should contain the added extractor")
}


func TestEndpointDataSourceAddExtractorWrongType(t *testing.T) {
	ds := NewEndpointDataSource("test-source")

	// Create an inline mock extractor that expects string type (wrong type)
	wrongTypeExtractor := &wrongTypeExtractor{
		name:         "wrong-type-extractor",
		expectedType: reflect.TypeOf(""),
	}

	// Try to add the extractor with wrong type
	err := ds.AddExtractor(wrongTypeExtractor)

	// Verify error is returned
	assert.Error(t, err, "Adding extractor with wrong type should return error")
	assert.Contains(t, err.Error(), "cannot handle", "Error should mention type mismatch")

	// Verify extractor is NOT in the list
	extractors := ds.Extractors()
	assert.Empty(t, extractors, "Should have no extractors after failed add")
}

// wrongTypeExtractor is a test-only mock that expects the wrong input type
type wrongTypeExtractor struct {
	name         string
	expectedType reflect.Type
}

func (w *wrongTypeExtractor) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{
		Type: "wrong-type-extractor",
		Name: w.name,
	}
}

func (w *wrongTypeExtractor) ExpectedInputType() reflect.Type {
	return w.expectedType
}

func (w *wrongTypeExtractor) Extract(ctx context.Context, data any, ep fwkdl.Endpoint) error {
	return nil
}


func TestEndpointDataSourceAddExtractorDuplicate(t *testing.T) {
	ds, extractor := setupDataSourceWithExtractor(t, "test-source", "test-extractor")

	// Try to add the same extractor again - should fail
	err := ds.AddExtractor(extractor)
	assert.Error(t, err, "Adding duplicate extractor should return error")
	assert.Contains(t, err.Error(), "duplicate", "Error should mention duplicate")

	// Verify only one extractor in the list
	extractors := ds.Extractors()
	assert.Len(t, extractors, 1, "Should have only one extractor")
}


func TestEndpointDataSourceNotifyAddEvent(t *testing.T) {
	ds, extractor := setupDataSourceWithExtractor(t, "test-source", "test-extractor")

	// Create test Pod and endpoint
	pod := createTestPod("test-pod", "default", "10.0.0.1")
	endpoint := createTestEndpoint("test-pod", "default")

	// Create change notification
	event := fwkdl.ChangeNotification{
		Type: fwkdl.EventAddedOrModified,
		Data: pod,
	}

	// Call Notify
	ctx := context.Background()
	err := ds.Notify(ctx, event, endpoint)
	assert.NoError(t, err, "Notify should not return error")

	// Verify extractor was called
	calls := extractor.GetExtractCalls()
	assert.Len(t, calls, 1, "Extractor should be called once")

	// Verify the call details
	call := calls[0]
	assert.Equal(t, fwkdl.EventAddedOrModified, call.ChangeNotification.Type, "Event type should match")
	assert.Equal(t, pod, call.ChangeNotification.Data, "Pod data should match")
	assert.Equal(t, endpoint, call.Endpoint, "Endpoint should match")
}

func TestEndpointDataSourceNotifyDeleteEvent(t *testing.T) {
	ds, extractor := setupDataSourceWithExtractor(t, "test-source", "test-extractor")

	// Create test Pod with deletion timestamp
	now := metav1.Now()
	pod := createTestPod("test-pod", "default", "10.0.0.1")
	pod.DeletionTimestamp = &now

	endpoint := createTestEndpoint("test-pod", "default")

	// Create delete notification
	event := fwkdl.ChangeNotification{
		Type: fwkdl.EventDeleted,
		Data: pod,
	}

	// Call Notify
	ctx := context.Background()
	err := ds.Notify(ctx, event, endpoint)
	assert.NoError(t, err, "Notify should not return error")

	// Verify extractor was called
	calls := extractor.GetExtractCalls()
	assert.Len(t, calls, 1, "Extractor should be called once")

	// Verify the call details
	call := calls[0]
	assert.Equal(t, fwkdl.EventDeleted, call.ChangeNotification.Type, "Event type should be EventDeleted")
	assert.Equal(t, pod, call.ChangeNotification.Data, "Pod data should match")
	assert.NotNil(t, call.ChangeNotification.Data.(*corev1.Pod).DeletionTimestamp, "Pod should have deletion timestamp")
}


func TestEndpointDataSourceNotifyMultipleExtractors(t *testing.T) {
	ds, extractor1 := setupDataSourceWithExtractor(t, "test-source", "extractor-1")

	// Add a second extractor
	extractor2 := mocks.NewMockEndpointExtractor("extractor-2")
	err := ds.AddExtractor(extractor2)
	require.NoError(t, err)

	// Create test Pod and endpoint
	pod := createTestPod("test-pod", "default", "10.0.0.1")
	endpoint := createTestEndpoint("test-pod", "default")

	// Create change notification
	event := fwkdl.ChangeNotification{
		Type: fwkdl.EventAddedOrModified,
		Data: pod,
	}

	// Call Notify once
	ctx := context.Background()
	err = ds.Notify(ctx, event, endpoint)
	assert.NoError(t, err, "Notify should not return error")

	// Verify both extractors were called
	calls1 := extractor1.GetExtractCalls()
	calls2 := extractor2.GetExtractCalls()
	
	assert.Len(t, calls1, 1, "Extractor 1 should be called once")
	assert.Len(t, calls2, 1, "Extractor 2 should be called once")

	// Verify both received the same event and endpoint
	assert.Equal(t, calls1[0].ChangeNotification.Type, calls2[0].ChangeNotification.Type, "Both should receive same event type")
	assert.Equal(t, calls1[0].ChangeNotification.Data, calls2[0].ChangeNotification.Data, "Both should receive same Pod data")
	assert.Equal(t, calls1[0].Endpoint, calls2[0].Endpoint, "Both should receive same endpoint")
}

func TestEndpointDataSourceNotifyExtractorError(t *testing.T) {
	ds, extractor1 := setupDataSourceWithExtractor(t, "test-source", "extractor-1")

	// Add a second extractor that will succeed
	extractor2 := mocks.NewMockEndpointExtractor("extractor-2")
	err := ds.AddExtractor(extractor2)
	require.NoError(t, err)

	// Configure first extractor to return an error
	expectedErr := errors.New("extractor error")
	extractor1.SetExtractError(expectedErr)

	// Create test Pod and endpoint
	pod := createTestPod("test-pod", "default", "10.0.0.1")
	endpoint := createTestEndpoint("test-pod", "default")

	// Create change notification
	event := fwkdl.ChangeNotification{
		Type: fwkdl.EventAddedOrModified,
		Data: pod,
	}

	// Call Notify
	ctx := context.Background()
	err = ds.Notify(ctx, event, endpoint)

	// Verify error is returned
	assert.Error(t, err, "Notify should return error from extractor")
	assert.Contains(t, err.Error(), "extractor error", "Error should contain extractor error message")

	// Verify both extractors were still called despite the error
	calls1 := extractor1.GetExtractCalls()
	calls2 := extractor2.GetExtractCalls()
	
	assert.Len(t, calls1, 1, "Extractor 1 should be called even though it errors")
	assert.Len(t, calls2, 1, "Extractor 2 should be called even though extractor 1 errored")
	assert.Equal(t, calls1[0].ChangeNotification.Data, calls2[0].ChangeNotification.Data, "Both should receive same Pod data")
	assert.Equal(t, calls1[0].Endpoint, calls2[0].Endpoint, "Both should receive same endpoint")
}
