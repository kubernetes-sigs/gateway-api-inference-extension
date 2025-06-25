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

package plugins

import (
	"encoding/json"
)

// Factory is the definition of the factory functions that are used to instantiate plugins
// specified in a configuration.
type FactoryFunc func(name string, parameters json.RawMessage, handle Handle) (Plugin, error)

// Register is a static function that can be called to register plugin factory functions.
func Register(pluginType string, factory FactoryFunc) {
	Registry[pluginType] = factory
}

// Registry is a mapping from plugin name to Factory function
var Registry map[string]FactoryFunc = map[string]FactoryFunc{}
