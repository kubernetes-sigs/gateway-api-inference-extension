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
	"encoding/json"
	"errors"
	"fmt"

	"k8s.io/apimachinery/pkg/runtime/schema"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// NotificationSourceType is the plugin type identifier for k8s notification sources.
const NotificationSourceType = "k8s-notification-source"

// notificationSourceParams holds the JSON parameters for the notification source factory.
type notificationSourceParams struct {
	// Group is the API group of the GVK to watch (empty string for core resources).
	Group string `json:"group"`
	// Version is the API version (e.g., "v1", "v1beta1").
	Version string `json:"version"`
	// Kind is the resource kind (e.g., "ConfigMap", "Deployment").
	Kind string `json:"kind"`
}

// NotificationSourceFactory is the factory function for k8s notification source plugins.
func NotificationSourceFactory(name string, parameters json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	if parameters == nil {
		return nil, errors.New("k8s-notification-source requires parameters (group, version, kind)")
	}

	var params notificationSourceParams
	if err := json.Unmarshal(parameters, &params); err != nil {
		return nil, fmt.Errorf("failed to parse notification source parameters: %w", err)
	}

	if params.Version == "" || params.Kind == "" {
		return nil, errors.New("version and kind are required for k8s-notification-source")
	}

	gvk := schema.GroupVersionKind{
		Group:   params.Group,
		Version: params.Version,
		Kind:    params.Kind,
	}

	// Default the plugin name to the GVK string if not explicitly provided.
	if name == "" {
		name = gvk.String()
	}

	return NewK8sNotificationSource(NotificationSourceType, name, gvk), nil
}
