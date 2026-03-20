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
// This interface should be embedded in all plugins across bbr code.
type BBRPlugin plugin.Plugin // alias

type RequestProcessor interface {
	BBRPlugin
	// ProcessRequest runs the RequestProcessor plugin.
	// RequestProcessor can mutate the headers and/or the body of the request.
	ProcessRequest(ctx context.Context, cycleState *CycleState, request *InferenceRequest) error
}

type ResponseProcessor interface {
	BBRPlugin
	// ProcessResponse runs the ResponseProcessor plugin.
	// ResponseProcessor can mutate the headers and/or the body of the response.
	ProcessResponse(ctx context.Context, cycleState *CycleState, response *InferenceResponse) error
}
