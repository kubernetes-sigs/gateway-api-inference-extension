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
	"fmt"
	"reflect"
	"sync"

	corev1 "k8s.io/api/core/v1"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// EndpointDataSource is a notification-based data source that receives
// Endpoint change events from the datastore and notifies registered extractors.
type EndpointDataSource struct {
	typedName  fwkplugin.TypedName
	extractors sync.Map // key: name, value: extractor
}

// Verify that EndpointDataSource implements NotificationDataSource
var _ fwkdl.NotificationDataSource = (*EndpointDataSource)(nil)

// NewEndpointDataSource creates a new endpoint notification data source.
func NewEndpointDataSource(name string) *EndpointDataSource {
	if name == "" {
		name = EndpointsDataSourceType
	}
	return &EndpointDataSource{
		typedName: fwkplugin.TypedName{
			Type: EndpointsDataSourceType,
			Name: name,
		},
	}
}

// TypedName returns the data source type and name.
func (ds *EndpointDataSource) TypedName() fwkplugin.TypedName {
	return ds.typedName
}

// Extractors returns a list of registered Extractor names.
func (ds *EndpointDataSource) Extractors() []string {
	extractors := []string{}
	ds.extractors.Range(func(_, val any) bool {
		if ex, ok := val.(fwkdl.Extractor); ok {
			extractors = append(extractors, ex.TypedName().String())
		}
		return true
	})
	return extractors
}

// AddExtractor adds an extractor to the data source, validating it can process
// the ChangeNotification type.
func (ds *EndpointDataSource) AddExtractor(extractor fwkdl.Extractor) error {
	// Validate that the extractor expects ChangeNotification as input
	expectedType := reflect.TypeOf(fwkdl.ChangeNotification{})
	if err := datalayer.ValidateExtractorType(expectedType, extractor.ExpectedInputType()); err != nil {
		return err
	}
	if _, loaded := ds.extractors.LoadOrStore(extractor.TypedName().Name, extractor); loaded {
		return fmt.Errorf("attempt to add duplicate extractor %s to %s", extractor.TypedName(), ds.TypedName())
	}
	return nil
}

// ExpectedEventDataType returns the type expected in notification events (*corev1.Pod).
func (ds *EndpointDataSource) ExpectedEventDataType() reflect.Type {
	return reflect.TypeOf((*corev1.Pod)(nil))
}

// AttachToStore registers this data source with the datastore to receive
// Endpoint change notifications.
func (ds *EndpointDataSource) AttachToStore(notifier fwkdl.StoreNotifier) error {
	endpointType := reflect.TypeOf((*fwkdl.Endpoint)(nil)).Elem()
	return notifier.RegisterObserver(endpointType, ds)
}

// Notify is called when a Pod affecting an endpoint changes.
// It forwards the notification to all registered extractors.
func (ds *EndpointDataSource) Notify(ctx context.Context, event fwkdl.ChangeNotification, ep fwkdl.Endpoint) error {
	var errs []error
	ds.extractors.Range(func(_, val any) bool {
		if ex, ok := val.(fwkdl.Extractor); ok {
			// Pass the ChangeNotification to the extractor
			if err := ex.Extract(ctx, event, ep); err != nil {
				errs = append(errs, err)
			}
		}
		return true
	})

	if len(errs) != 0 {
		return errors.Join(errs...)
	}
	return nil
}
