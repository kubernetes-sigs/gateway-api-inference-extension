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

package utils

import (
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework"
	bbrplugins "sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/plugins"
)

// RegisterAllFactories registers all factories for all plugins implementing BBRPlugin interface
// As more plugins are developed, this function is extended to make sure that the registry is bootstrapped to have
// all constructors upfront upon initialization when utils.init() is called.
// Whether instances of these plugins are created, depends on how the plugin chains for request and response are configured.
// By default, a request plugins chain is always present, containing "MetaDataExtractor/simple-model-extractor" plugin
// That pulls model from the body into the X-Gateway-Model-Name header
// In extended implementations (outside of IGW), if there are plugins, which are not upstream in IGW,
// this function should be extended to include factories for the plugins unknown to the base IGW code
func RegisterAllFactories(registry framework.PluginRegistry) error {
	//default plugin factory registration
	err := registry.RegisterFactory(bbrplugins.DefaultPluginType,
		func() bbrplugins.BBRPlugin {
			return bbrplugins.NewDefaultMetaDataExtractor()
		})
	if err != nil {
		return err
	}
	//another plugin factory registration here. etc.
	return nil
}
