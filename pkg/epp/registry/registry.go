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

package registry

import (
	"encoding/json"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
)

// Factory is the definition of the factory functions that are used to instantiate plugins
// specified in a configuration.
type Factory func(parameters json.RawMessage) (plugins.Plugin, error)

// Register is a static function that can be called to register plugin factory functions.
func Register(name string, factory Factory) {
	Registry[name] = factory
}

// Registry is a mapping from plugin name to Factory function
var Registry map[string]Factory = map[string]Factory{}
