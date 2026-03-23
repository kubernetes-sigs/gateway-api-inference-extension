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

package epp

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap/zapcore"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/test/integration"
)

var reqLogger = zap.New(zap.UseDevMode(true), zap.Level(-1*zapcore.Level(logutil.DEFAULT)))

const requestAttributeReporterTestConfig = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: openai-parser
  - name: total-tokens-cost-reporter
    type: request-attribute-reporter
    parameters:
      attributes:
        - key:
            namespace: envoy.lb
            name: x-gateway-inference-request-cost
          expression: |
            usage.completion_tokens
          condition: "has(usage.completion_tokens)"
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: openai-parser
      - pluginRef: total-tokens-cost-reporter
parser:
  pluginRef: openai-parser
`

func TestRequestAttributeReporter(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Use our new WithConfigText harness option to provide the custom config.
	h := NewTestHarness(t, ctx, WithStandardMode(), WithConfigText(requestAttributeReporterTestConfig)).WithBaseResources()

	// Pods must exist in datastore so that there's no early failure
	pods := []podState{P(0, 0, 0.1, modelMyModelTarget)}
	h.WithPods(pods).WaitForSync(len(pods), modelMyModel)
	h.WaitForReadyPodsMetric(len(pods))

	// Send the complete simulated transaction (Request headers -> target selection -> Response headers -> Response body)
	// 1. Envoy sends the request (Headers + Body)
	requests := integration.ReqLLM(reqLogger, "hello", "modelName", "modelName")

	// 2. Envoy sends the upstream response (Headers + Body)
	respRequests := ReqResponseOnly(
		map[string]string{"content-type": "application/json", "status": "200"},
		`{"choices":[{"finish_reason":"stop","index":0,"logprobs":null,"message":{"content":"Hello! How can I help you today?","role":"assistant"}}],"created":1773675992,"id":"chatcmpl-63a9a231-a267-4d15-a799-39e9dceef8a5","model":"modelName","object":"chat.completion","usage":{"completion_tokens":10,"prompt_tokens":32,"total_tokens":42}}`,
	)
	requests = append(requests, respRequests...)

	// Since we send RequestHeaders + RequestBody + ResponseHeaders + ResponseBody, EPP will send:
	// 1. request header response
	// 2. request body response
	// 3. response header response
	// 4. response body response (this is where dynamic metadata is attached)
	responses, err := integration.StreamedRequest(t, h.Client, requests, 4)
	require.NoError(t, err)
	require.Len(t, responses, 4)

	// Verify that the response contains the dynamic metadata.
	// It should be attached to the response body response (index 3)
	res := responses[3]
	require.NotNil(t, res.DynamicMetadata, "expected DynamicMetadata in ext_proc response")

	envoyLbMap, ok := res.DynamicMetadata.Fields["envoy.lb"]
	require.True(t, ok, "expected envoy.lb namespace in DynamicMetadata")

	costMetric, ok := envoyLbMap.GetStructValue().Fields["x-gateway-inference-request-cost"]
	require.True(t, ok, "expected x-gateway-inference-request-cost in envoy.lb namespace")
	require.Equal(t, float64(10), costMetric.GetNumberValue(), "expected cost metric to be 10 based on completion_tokens")
}
