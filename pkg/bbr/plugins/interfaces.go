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

package bbrplugins

import (
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
)

// ------------------------------------ Defaults ------------------------------------------
const (
	//The deafult plugin implementation of this plugin type will always be configured for request plugins chain
	//Even though BBRPlugin type is not a K8s resource, it's logically akin to `kind`
	//Shoud start wit an upper case letter, use CamelNotation, only aplhanumericals after the first letter
	PluginTypePattern   = `^[A-Z][A-Za-z0-9]*$`
	MaxPluginTypeLength = 63
	DefaultPluginType   = "MetaDataExtractor"
	// Even though BBRPlugin is not a K8s resource yet, let's make its naming compliant with K8s resource naming
	// Allows: lowercase letters, digits, hyphens, dots.
	// Must start and end with a lowercase alphanumeric character.
	// Middle characters group can contain lowercase alphanumerics, hyphens, and dots
	// Middle and rightmost groups are optional
	PluginNamePattern   = `^[a-z0-9]([-a-z0-9.]*[a-z0-9])?$`
	DefaultPluginName   = "simple-model-extractor"
	MaxPluginNameLength = 253
	//Well-known custom header set to a model name
	ModelHeader = "X-Gateway-Model-Name"
)

// BBRPlugin defines the interface for plugins in the BBR framework should never mutate the body directly.
type BBRPlugin interface {
	plugins.Plugin

	// RequiresFullParsing indicates whether full body parsing is required
	// to facilitate efficient memory sharing across plugins in a chain.
	RequiresFullParsing() bool

	// Execute runs the plugin logic on the request body.
	// A plugin's imnplementation logic CAN mutate the body of the message.
	// A plugin's implementation MUST return a map of headers
	// If no headers are set by the implementation, the map must be empty
	// A value of a header in an extended implementation NEED NOT to be identical to the value of that same header as would be set
	// in a default implementation.
	// Example: in the body of a request model is set to "semantic-model-selector",
	// which, say, stands for "select a best model for this request at minimal cost"
	// A plugin implementation of "semantic-model-selector" sets X-Gateway-Model-Name to any valid
	// model name from the inventory of the backend models and also mutates the body accordingly
	// In contrast,
	Execute(requestBodyBytes []byte) (
		headers map[string]string,
		mutatedBodyBytes []byte,
		err error,
	)
}
