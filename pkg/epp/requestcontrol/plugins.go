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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugin"
	types "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/scheduling"
)

const (
	PreRequestExtensionPoint        = "PreRequest"
	ResponseReceivedExtensionPoint  = "ResponseReceived"
	ResponseStreamingExtensionPoint = "ResponseStreaming"
	ResponseCompleteExtensionPoint  = "ResponseComplete"
)

// PreRequest is called by the director after a getting result from scheduling layer and
// before a request is sent to the selected model server.
type PreRequest interface {
	plugin.Plugin
	PreRequest(ctx context.Context, request *types.LLMRequest, schedulingResult *types.SchedulingResult)
}

// ResponseReceived is called by the director after the response headers are successfully received
// which indicates the beginning of the response handling by the model server.
// The given pod argument is the pod that served the request.
type ResponseReceived interface {
	plugin.Plugin
	ResponseReceived(ctx context.Context, request *types.LLMRequest, response *Response, targetEndpoint *datalayer.EndpointMetadata)
}

// ResponseStreaming is called by the director after each chunk of streaming response is sent.
type ResponseStreaming interface {
	plugin.Plugin
	ResponseStreaming(ctx context.Context, request *types.LLMRequest, response *Response, targetEndpoint *datalayer.EndpointMetadata)
}

// ResponseComplete is called by the director when the request lifecycle terminates.
// This occurs after a response is fully sent, OR if the request fails/disconnects after a pod was scheduled.
//
// Plugins should assume this is the final cleanup hook for a request.
//
// TODO(https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/2079):
// Update signature to pass error/termination state. This is a breaking change required for plugins to distinguish
// between success, errors, and disconnects.
type ResponseComplete interface {
	plugin.Plugin
	ResponseComplete(ctx context.Context, request *types.LLMRequest, response *Response, targetEndpoint *datalayer.EndpointMetadata)
}

// PrepareRequestData is called by the director before scheduling requests.
// PrepareDataPlugin plugin is implemented by data producers which produce data from different sources.
type PrepareDataPlugin interface {
	plugin.ProducerPlugin
	plugin.ConsumerPlugin
	PrepareRequestData(ctx context.Context, request *types.LLMRequest, pods []types.Endpoint) error
}

// AdmissionPlugin is called by the director after the prepare data phase and before scheduling.
// When a request has to go through multiple AdmissionPlugin,
// the request is admitted only if all plugins say that the request should be admitted.
type AdmissionPlugin interface {
	plugin.Plugin
	// AdmitRequest returns the denial reason, wrapped as error if the request is denied.
	// If the request is allowed, it returns nil.
	AdmitRequest(ctx context.Context, request *types.LLMRequest, pods []types.Endpoint) error
}
