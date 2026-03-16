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

	"sigs.k8s.io/gateway-api-inference-extension/test/integration"
)

func TestRequestAttributeReporterStreaming(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	h := NewTestHarness(t, ctx, WithStandardMode(), WithConfigText(requestAttributeReporterTestConfig)).WithBaseResources()

	pods := []podState{P(0, 0, 0.1, modelMyModelTarget)}
	h.WithPods(pods).WaitForSync(len(pods), modelMyModel)
	h.WaitForReadyPodsMetric(len(pods))

	requests := integration.ReqLLM(reqLogger, "hello", "modelName", "modelName")

	respRequests := ReqResponseOnly(
		map[string]string{"content-type": "text/event-stream", "status": "200"},
		`data: {"choices":[{"delta":{"content":"Hello! "},"index":0,"finish_reason":null}],"id":"123","created":1,"model":"modelName","object":"chat.completion.chunk"}
`,
		`data: {"choices":[{"delta":{},"index":0,"finish_reason":"stop"}],"id":"123","created":1,"model":"modelName","object":"chat.completion.chunk","usage":{"completion_tokens":10,"prompt_tokens":32,"total_tokens":42}}
`,
	)
	requests = append(requests, respRequests...)

	// We send ReqHeaders, ReqBody, RespHeaders, RespBody(chunk1), RespBody(chunk2) -> 5 responses
	responses, err := integration.StreamedRequest(t, h.Client, requests, 5)
	require.NoError(t, err)

	res := responses[4]
	require.NotNil(t, res.DynamicMetadata, "expected DynamicMetadata in ext_proc response")
	envoyLbMap, ok := res.DynamicMetadata.Fields["envoy.lb"]
	require.True(t, ok, "expected envoy.lb namespace in DynamicMetadata")
	costMetric, ok := envoyLbMap.GetStructValue().Fields["x-gateway-inference-request-cost"]
	require.True(t, ok)
	require.Equal(t, float64(10), costMetric.GetNumberValue())
}
