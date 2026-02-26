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

package framework

import (
	"context"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// BBRPlugin defines the interface for a plugin.
// This interface should be embedded in all plugins across the code.
type BBRPlugin plugin.Plugin // alias

type PayloadProcessor interface {
	BBRPlugin
	// Execute runs the payload processor plugin.
	// Payload processor can mutate the headers and/or the body of the message.
	Execute(ctx context.Context, headers map[string]string, body map[string]any) (updatedHeaders map[string]string, updatedBody map[string]any, err error)
}

type Guardrail interface {
	BBRPlugin
	// Execute runs guardrail plugin
	// Guardrail plugin can inspect the request, including headers and payload and decide
	// whether the request should be blocked or not.
	Execute(ctx context.Context, headers map[string]string, body map[string]any) (bool, error)
}
