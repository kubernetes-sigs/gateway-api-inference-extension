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

package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/testing/protocmp"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	epp "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

const testPluginValue = "done"

// fakeResponsePlugin implements framework.PayloadProcessor for testing response plugin execution.
type fakeResponsePlugin struct {
	name     string
	mutateFn func(ctx context.Context, headers map[string]string, body map[string]any) (map[string]string, map[string]any, error)
}

func (p *fakeResponsePlugin) TypedName() epp.TypedName {
	return epp.TypedName{Type: "fake", Name: p.name}
}

func (p *fakeResponsePlugin) Execute(ctx context.Context, headers map[string]string, body map[string]any) (map[string]string, map[string]any, error) {
	return p.mutateFn(ctx, headers, body)
}

var _ framework.PayloadProcessor = &fakeResponsePlugin{}

func newTestRequestContext() *RequestContext {
	return &RequestContext{
		Request:  &Request{Headers: make(map[string]string)},
		Response: &Response{Headers: make(map[string]string)},
	}
}

func TestHandleResponseBody_NoPlugins(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	server := NewServer(false, &fakeDatastore{}, []framework.PayloadProcessor{}, []framework.PayloadProcessor{})
	responseBody := []byte(`{"choices":[{"text":"Hello!"}]}`)
	resp, err := server.HandleResponseBody(ctx, newTestRequestContext(), responseBody)
	if err != nil {
		t.Fatalf("HandleResponseBody returned unexpected error: %v", err)
	}

	want := []*extProcPb.ProcessingResponse{
		{
			Response: &extProcPb.ProcessingResponse_ResponseBody{
				ResponseBody: &extProcPb.BodyResponse{},
			},
		},
	}

	if diff := cmp.Diff(want, resp, protocmp.Transform()); diff != "" {
		t.Errorf("HandleResponseBody returned unexpected response, diff(-want, +got): %v", diff)
	}
}

func TestHandleResponseBody_SinglePlugin(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	mutatePlugin := &fakeResponsePlugin{
		name: "mutator",
		mutateFn: func(_ context.Context, headers map[string]string, body map[string]any) (map[string]string, map[string]any, error) {
			body["mutated"] = true
			return headers, body, nil
		},
	}

	server := NewServer(false, &fakeDatastore{}, []framework.PayloadProcessor{}, []framework.PayloadProcessor{mutatePlugin})
	responseBody := []byte(`{"choices":[{"text":"Hello!"}]}`)
	resp, err := server.HandleResponseBody(ctx, newTestRequestContext(), responseBody)
	if err != nil {
		t.Fatalf("HandleResponseBody returned unexpected error: %v", err)
	}

	// The mutated body should contain the new "mutated" field.
	var gotBody map[string]any
	bodyResp := resp[0].GetResponseBody().GetResponse().GetBodyMutation().GetBody()
	if err := json.Unmarshal(bodyResp, &gotBody); err != nil {
		t.Fatalf("Failed to unmarshal response body: %v", err)
	}
	if gotBody["mutated"] != true {
		t.Errorf("Expected mutated field to be true, got: %v", gotBody)
	}
}

func TestHandleResponseBody_PluginChaining(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	plugin1 := &fakeResponsePlugin{
		name: "plugin1",
		mutateFn: func(_ context.Context, headers map[string]string, body map[string]any) (map[string]string, map[string]any, error) {
			body["p1"] = testPluginValue
			return headers, body, nil
		},
	}
	plugin2 := &fakeResponsePlugin{
		name: "plugin2",
		mutateFn: func(_ context.Context, headers map[string]string, body map[string]any) (map[string]string, map[string]any, error) {
			body["p2"] = testPluginValue
			return headers, body, nil
		},
	}

	server := NewServer(false, &fakeDatastore{}, []framework.PayloadProcessor{}, []framework.PayloadProcessor{plugin1, plugin2})
	responseBody := []byte(`{"original":true}`)
	resp, err := server.HandleResponseBody(ctx, newTestRequestContext(), responseBody)
	if err != nil {
		t.Fatalf("HandleResponseBody returned unexpected error: %v", err)
	}

	var gotBody map[string]any
	bodyResp := resp[0].GetResponseBody().GetResponse().GetBodyMutation().GetBody()
	if err := json.Unmarshal(bodyResp, &gotBody); err != nil {
		t.Fatalf("Failed to unmarshal response body: %v", err)
	}
	if gotBody["p1"] != testPluginValue || gotBody["p2"] != testPluginValue {
		t.Errorf("Expected both plugins to have mutated the body, got: %v", gotBody)
	}
}

func TestHandleResponseBody_PluginError(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	failingPlugin := &fakeResponsePlugin{
		name: "failing",
		mutateFn: func(_ context.Context, _ map[string]string, _ map[string]any) (map[string]string, map[string]any, error) {
			return nil, nil, errors.New("guardrail violation detected")
		},
	}

	server := NewServer(false, &fakeDatastore{}, []framework.PayloadProcessor{}, []framework.PayloadProcessor{failingPlugin})
	responseBody := []byte(`{"choices":[{"text":"bad content"}]}`)
	_, err := server.HandleResponseBody(ctx, newTestRequestContext(), responseBody)
	if err == nil {
		t.Fatal("HandleResponseBody should have returned an error")
	}

	if got := err.Error(); got == "" {
		t.Error("Expected non-empty error message")
	}
}

func TestHandleResponseBody_StreamingWithPlugin(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	noopPlugin := &fakeResponsePlugin{
		name: "noop",
		mutateFn: func(_ context.Context, headers map[string]string, body map[string]any) (map[string]string, map[string]any, error) {
			return headers, body, nil
		},
	}

	server := NewServer(true, &fakeDatastore{}, []framework.PayloadProcessor{}, []framework.PayloadProcessor{noopPlugin})
	responseBody := []byte(`{"choices":[{"text":"Hello!"}]}`)
	resp, err := server.HandleResponseBody(ctx, newTestRequestContext(), responseBody)
	if err != nil {
		t.Fatalf("HandleResponseBody returned unexpected error: %v", err)
	}

	// In streaming mode with plugins, should return a StreamedBodyResponse.
	if len(resp) == 0 {
		t.Fatal("Expected at least one response in streaming mode")
	}
	streamedResp := resp[len(resp)-1].GetResponseBody().GetResponse().GetBodyMutation().GetStreamedResponse()
	if streamedResp == nil {
		t.Fatal("Expected streaming response with StreamedBodyResponse")
	}
	if !streamedResp.EndOfStream {
		t.Error("Expected EndOfStream to be true for the last streamed response chunk")
	}
}

func TestProcessResponseBody_Streaming(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	noopPlugin := &fakeResponsePlugin{
		name: "noop",
		mutateFn: func(_ context.Context, headers map[string]string, body map[string]any) (map[string]string, map[string]any, error) {
			return headers, body, nil
		},
	}

	server := NewServer(true, &fakeDatastore{}, []framework.PayloadProcessor{}, []framework.PayloadProcessor{noopPlugin})

	chunk1 := &extProcPb.HttpBody{
		Body: []byte(`{"choices":[{"te`),
	}
	chunk2 := &extProcPb.HttpBody{
		Body:        []byte(`xt":"Hello!"}]}`),
		EndOfStream: true,
	}

	reqCtx := newTestRequestContext()
	respStreamedBody := &streamedBody{}

	resp1, err := server.processResponseBody(ctx, reqCtx, chunk1, respStreamedBody)
	if err != nil {
		t.Fatalf("processResponseBody chunk1 returned unexpected error: %v", err)
	}
	if resp1 != nil {
		t.Fatalf("processResponseBody chunk1 should return nil while buffering, got: %v", resp1)
	}

	resp2, err := server.processResponseBody(ctx, reqCtx, chunk2, respStreamedBody)
	if err != nil {
		t.Fatalf("processResponseBody chunk2 returned unexpected error: %v", err)
	}
	if resp2 == nil {
		t.Fatal("processResponseBody chunk2 should return a response on EoS")
	}
}
