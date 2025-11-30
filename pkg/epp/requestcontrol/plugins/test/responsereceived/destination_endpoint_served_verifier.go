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

package responsereceived

import (
	"context"
	"encoding/json"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol/plugins/test"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	// DestinationEndpointServedVerifierType is the ResponseReceived type that is used in plugins registry.
	DestinationEndpointServedVerifierType = "destination-endpoint-served-verifier"
)

var _ requestcontrol.ResponseReceived = &DestinationEndpointServedVerifier{}

// DestinationEndpointServedVerifier is a test-only plugin for conformance tests.
// It verifies that the request was served by the expected endpoint.
// It works by reading Envoy's dynamic metadata, which is passed in the
// Response.ReqMetadata field. This metadata should contain the
// address of the backend endpoint that served the request. The verifier then
// writes this address to the "x-conformance-test-served-endpoint" response header.
// The conformance test client can then validate this header to ensure the request
// was routed correctly.
type DestinationEndpointServedVerifier struct {
	typedName plugins.TypedName
}

// TypedName returns the type and name tuple of this plugin instance.
func (f *DestinationEndpointServedVerifier) TypedName() plugins.TypedName {
	return f.typedName
}

// WithName sets the name of the filter.
func (f *DestinationEndpointServedVerifier) WithName(name string) *DestinationEndpointServedVerifier {
	f.typedName.Name = name
	return f
}

// DestinationEndpointServedVerifierFactory defines the factory function for DestinationEndpointServedVerifier.
func DestinationEndpointServedVerifierFactory(name string, _ json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	return NewDestinationEndpointServedVerifier().WithName(name), nil
}

func NewDestinationEndpointServedVerifier() *DestinationEndpointServedVerifier {
	return &DestinationEndpointServedVerifier{}
}

// ResponseReceived is the handler for the ResponseReceived extension point.
func (p *DestinationEndpointServedVerifier) ResponseReceived(ctx context.Context, request *schedulingtypes.LLMRequest, response *requestcontrol.Response, _ *backend.Pod) {
	logger := log.FromContext(ctx).WithName(p.TypedName().String())
	logger.V(logging.DEBUG).Info("Verifying destination endpoint served")

	reqMetadata := response.ReqMetadata
	lbMetadata, ok := reqMetadata[metadata.DestinationEndpointNamespace].(map[string]any)
	if !ok {
		logger.V(logging.DEBUG).Info("Response does not contain envoy lb metadata, skipping verification")
		response.Headers[test.ConformanceTestResultHeader] = "fail: missing envoy lb metadata"
		return
	}

	actualEndpoint, ok := lbMetadata[metadata.DestinationEndpointServedKey].(string)
	if !ok {
		logger.V(logging.DEBUG).Info("Response does not contain destination endpoint served metadata, skipping verification")
		response.Headers[test.ConformanceTestResultHeader] = "fail: missing destination endpoint served metadata"
		return
	}
	response.Headers[test.ConformanceTestResultHeader] = actualEndpoint
}
