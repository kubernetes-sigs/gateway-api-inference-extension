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
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
)

type ControllerConfig struct {
	startCrdReconcilers       bool
	hasInferenceObjective     bool
	hasInferenceModelRewrites bool
}

func NewControllerConfig(startCrdReconcilers bool) ControllerConfig {
	return ControllerConfig{
		startCrdReconcilers: startCrdReconcilers,
	}
}

func (cc *ControllerConfig) PopulateControllerConfig(cfg *rest.Config) error {
	if !cc.startCrdReconcilers {
		return nil
	}
	dc, err := discovery.NewDiscoveryClientForConfig(cfg)
	if err != nil {
		return err
	}
	cc.populateWithDiscovery(dc)
	return nil
}

func (cc *ControllerConfig) populateWithDiscovery(dc discovery.DiscoveryInterface) {
	inferenceObjectiveGVK := schema.GroupVersionKind{
		Group:   v1alpha2.GroupVersion.Group,
		Version: v1alpha2.GroupVersion.Version,
		Kind:    "InferenceObjective",
	}
	cc.hasInferenceObjective = gvkExists(dc, inferenceObjectiveGVK)

	inferenceModelRewriteGVK := schema.GroupVersionKind{
		Group:   v1alpha2.GroupVersion.Group,
		Version: v1alpha2.GroupVersion.Version,
		Kind:    "InferenceModelRewrite",
	}
	cc.hasInferenceModelRewrites = gvkExists(dc, inferenceModelRewriteGVK)
}

func gvkExists(dc discovery.DiscoveryInterface, gvk schema.GroupVersionKind) bool {
	apiResourceList, err := dc.ServerResourcesForGroupVersion(gvk.GroupVersion().String())
	if err != nil {
		ctrl.Log.WithName("controllerConfig").Error(err, "Checking server resources error.")
		return false
	}
	for _, r := range apiResourceList.APIResources {
		if r.Kind == gvk.Kind {
			return true
		}
	}
	return false
}
