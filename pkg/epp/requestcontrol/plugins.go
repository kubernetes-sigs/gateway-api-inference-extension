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

package requestcontrol

import (
	"context"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

const (
	PreRequestExtensionPoint        = "PreRequest"
	ResponseReceivedExtensionPoint  = "ResponseReceived"
	ResponseStreamingExtensionPoint = "ResponseStreaming"
	ResponseCompleteExtensionPoint  = "ResponseComplete"
)

// PreRequest is called by the director after a getting result from scheduling layer but
// before a request is sent to the selected model server.
type PreRequest interface {
	plugins.Plugin
	PreRequest(ctx context.Context, request *types.LLMRequest, schedulingResult *types.SchedulingResult, targetPort int)
}

// ResponseReceived is called by the director after the response headers are successfully received
// which indicates the beginning of the response handling by the model server.
// The given pod argument is the pod that served the request.
type ResponseReceived interface {
	plugins.Plugin
	ResponseReceived(ctx context.Context, request *types.LLMRequest, response *Response, targetPod *backend.Pod)
}

// ResponseStreaming is called by the director after each chunk of streaming response is sent.
type ResponseStreaming interface {
	plugins.Plugin
	ResponseStreaming(ctx context.Context, request *types.LLMRequest, response *Response, targetPod *backend.Pod)
}

// ResponseComplete is called by the director after the complete response is sent.
type ResponseComplete interface {
	plugins.Plugin
	ResponseComplete(ctx context.Context, request *types.LLMRequest, response *Response, targetPod *backend.Pod)
}
