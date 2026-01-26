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

package plugins

import (
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

type BBRPlugin interface {
	plugin.Plugin

	// Execute runs the plugin's logic on the request body.
	// A plugin's implementation logic CAN mutate the body of the message.
	// A plugin's implementation MUST return a map of headers.
	// If no headers are set by the implementation, the return headers map is nil.
	// In the future, a headers map can be added to the plugin interface to allow different plugins on a chain to share information on the headers.
	Execute(requestBodyBytes []byte) (mutatedBodyBytes []byte, headers map[string][]string, err error)
}
