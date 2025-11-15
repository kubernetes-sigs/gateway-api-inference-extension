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
	"testing"

	"github.com/stretchr/testify/require"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol/plugins/test"
)

func TestDestinationEndpointServedVerifier_ResponseReceived(t *testing.T) {
	testCases := []struct {
		name                string
		response            *requestcontrol.Response
		expectedHeaderValue string
	}{
		{
			name: "success - endpoint is correctly reported",
			response: &requestcontrol.Response{
				Headers: make(map[string]string),
				ReqMetadata: map[string]any{
					metadata.DestinationEndpointNamespace: map[string]any{
						metadata.DestinationEndpointServedKey: "10.0.0.1:8080",
					},
				},
			},
			expectedHeaderValue: "10.0.0.1:8080",
		},
		{
			name: "failure - missing lb metadata",
			response: &requestcontrol.Response{
				Headers:     make(map[string]string),
				ReqMetadata: map[string]any{},
			},
			expectedHeaderValue: "fail: missing envoy lb metadata",
		},
		{
			name: "failure - missing served endpoint key",
			response: &requestcontrol.Response{
				Headers: make(map[string]string),
				ReqMetadata: map[string]any{
					metadata.DestinationEndpointNamespace: map[string]any{
						"some-other-key": "some-value",
					},
				},
			},
			expectedHeaderValue: "fail: missing destination endpoint served metadata",
		},
		{
			name: "failure - nil metadata",
			response: &requestcontrol.Response{
				Headers:     make(map[string]string),
				ReqMetadata: nil,
			},
			expectedHeaderValue: "fail: missing envoy lb metadata",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			plugin := NewDestinationEndpointServedVerifier().WithName("test-verifier")

			plugin.ResponseReceived(context.Background(), nil, tc.response, nil)

			actualHeader, ok := tc.response.Headers[test.ConformanceTestResultHeader]
			require.True(t, ok, "Expected header %s to be set", test.ConformanceTestResultHeader)
			require.Equal(t, tc.expectedHeaderValue, actualHeader)
		})
	}
}
