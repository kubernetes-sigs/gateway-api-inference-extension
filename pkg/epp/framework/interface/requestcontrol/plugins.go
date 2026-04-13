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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	types "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
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
	PreRequest(ctx context.Context, request *types.InferenceRequest, schedulingResult *types.SchedulingResult)
}

// ResponseHeaderProcessor is called by the director after the response headers are successfully received
// which indicates the beginning of the response handling by the model server.
// The given pod argument is the pod that served the request.
type ResponseHeaderProcessor interface {
	plugin.Plugin
	ResponseHeader(ctx context.Context, request *types.InferenceRequest, response *Response, targetEndpoint *datalayer.EndpointMetadata)
}

// ResponseBodyProcessor is the primary hook for processing response data.
// It is called by the director for every data chunk in a streaming response, or exactly once
// for non-streaming responses.
//
// Lifecycle & Termination:
//   - For streams: Invoked multiple times. The final call will have response.EndOfStream set to true.
//   - For non-streaming: Invoked once with response.EndOfStream set to true.
//   - Plugins must treat the call where response.EndOfStream == true as the final lifecycle hook
//     to perform cleanup or final logging.
//
// TODO(https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/2079):
// Update signature to pass error/termination state. This is a breaking change required for plugins to distinguish
// between success, errors, and disconnects.
type ResponseBodyProcessor interface {
	plugin.Plugin
	ResponseBody(ctx context.Context, request *types.InferenceRequest, response *Response, targetEndpoint *datalayer.EndpointMetadata)
}

// DataProducer is implemented by data producers which produce data from different sources.
// PrepareRequestData is called by the director before scheduling requests.
type DataProducer interface {
	plugin.ProducerPlugin
	plugin.ConsumerPlugin
	PrepareRequestData(ctx context.Context, request *types.InferenceRequest, pods []types.Endpoint) error
}

// Admitter is called by the director after the data producer and before scheduling.
// When a request has to go through multiple Admitter,
// the request is admitted only if all plugins say that the request should be admitted.
type Admitter interface {
	plugin.Plugin
	// AdmitRequest returns the denial reason, wrapped as error if the request is denied.
	// If the request is allowed, it returns nil.
	AdmitRequest(ctx context.Context, request *types.InferenceRequest, pods []types.Endpoint) error
}
