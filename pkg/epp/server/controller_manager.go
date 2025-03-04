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

package server

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
)

var scheme = runtime.NewScheme()

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(v1alpha2.AddToScheme(scheme))
}

// NewDefaultManager creates a new controller manager with default configuration.
func NewDefaultManager(namespace, name string, restConfig *rest.Config) (ctrl.Manager, error) {
	manager, err := ctrl.NewManager(restConfig, ctrl.Options{
		Scheme: scheme,
		Cache: cache.Options{
			ByObject: map[client.Object]cache.ByObject{
				&corev1.Pod{}: {
					Namespaces: map[string]cache.Config{
						namespace: {},
					},
				},
				&v1alpha2.InferencePool{}: {
					Namespaces: map[string]cache.Config{
						namespace: {
							FieldSelector: fields.SelectorFromSet(fields.Set{
								"metadata.name": name,
							}),
						},
					},
				},
				&v1alpha2.InferenceModel{}: {
					Namespaces: map[string]cache.Config{
						namespace: {},
					},
				},
			},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create controller manager: %v", err)
	}
	return manager, nil
}
