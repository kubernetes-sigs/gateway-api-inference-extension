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
	"errors"
	"fmt"
	"reflect"
	"sync"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/log"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

var (
	_ fwkdl.DataSource         = (*K8sNotificationSource)(nil)
	_ fwkdl.NotificationSource = (*K8sNotificationSource)(nil)
)

// K8sNotificationSource watches a single GVK and dispatches events to
// registered NotificationExtractors.
type K8sNotificationSource struct {
	typedName  fwkplugin.TypedName
	gvk        schema.GroupVersionKind
	extractors sync.Map // key: name (string), value: fwkdl.NotificationExtractor
}

// NewK8sNotificationSource returns a new notification source for the given GVK.
func NewK8sNotificationSource(pluginType, pluginName string,
	gvk schema.GroupVersionKind) *K8sNotificationSource {
	return &K8sNotificationSource{
		typedName: fwkplugin.TypedName{Type: pluginType, Name: pluginName},
		gvk:       gvk,
	}
}

// TypedName returns the plugin type and name.
func (s *K8sNotificationSource) TypedName() fwkplugin.TypedName {
	return s.typedName
}

// GVK returns the GroupVersionKind this source watches.
func (s *K8sNotificationSource) GVK() schema.GroupVersionKind {
	return s.gvk
}

// OutputType returns the type of data this DataSource produces (NotificationEvent).
func (s *K8sNotificationSource) OutputType() reflect.Type {
	return fwkdl.NotificationEventType
}

// ExtractorType returns the type of Extractor this DataSource expects (NotificationExtractor).
func (s *K8sNotificationSource) ExtractorType() reflect.Type {
	return fwkdl.NotificationExtractorType
}

// Extractors returns names of registered extractors.
func (s *K8sNotificationSource) Extractors() []string {
	var names []string
	s.extractors.Range(func(_, val any) bool {
		if ext, ok := val.(fwkdl.NotificationExtractor); ok {
			names = append(names, ext.TypedName().String())
		}
		return true
	})
	return names
}

// AddExtractor registers an extractor.
// Validation of extractor compatibility is done by the runtime via datalayer.WithConfig.
func (s *K8sNotificationSource) AddExtractor(ext fwkdl.Extractor) error {
	if _, loaded := s.extractors.LoadOrStore(ext.TypedName().Name, ext); loaded {
		return fmt.Errorf("duplicate extractor %s on notification source %s",
			ext.TypedName(), s.TypedName())
	}
	return nil
}

// Notify dispatches a notification event to all registered extractors
// synchronously, preserving event ordering.
func (s *K8sNotificationSource) Notify(ctx context.Context, event fwkdl.NotificationEvent) error {
	logger := log.FromContext(ctx).WithValues("gvk", s.gvk, "eventType", event.Type)

	var errs []error
	s.extractors.Range(func(_, val any) bool {
		ext, ok := val.(fwkdl.NotificationExtractor)
		if !ok {
			errs = append(errs, fmt.Errorf("extractor %s does not implement NotificationExtractor", ext.TypedName()))
			return true
		}
		if err := ext.ExtractNotification(ctx, event); err != nil {
			errs = append(errs, fmt.Errorf("extractor %s: %w", ext.TypedName(), err))
		}
		return true
	})

	if len(errs) > 0 {
		logger.Error(errors.Join(errs...), "extractor(s) failed processing notification")
	}
	return nil
}
