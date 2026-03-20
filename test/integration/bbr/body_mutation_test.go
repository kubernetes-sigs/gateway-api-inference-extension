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

package bbr

import (
	"context"
	"encoding/json"
	"strconv"
	"testing"

	envoyCorev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/testing/protocmp"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/plugins/test"
	envoytest "sigs.k8s.io/gateway-api-inference-extension/pkg/common/envoy/test"
	epp "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/test/integration"
)

// bodyMutatingPlugin is a test plugin that injects a field into the request body.
type bodyMutatingPlugin struct {
	fieldName  string
	fieldValue any
}

func (p *bodyMutatingPlugin) TypedName() epp.TypedName {
	return epp.TypedName{Type: "test-body-mutator", Name: "test-body-mutator"}
}

func (p *bodyMutatingPlugin) ProcessRequest(_ context.Context, _ *framework.CycleState, request *framework.InferenceRequest) error {
	request.SetBodyField(p.fieldName, p.fieldValue)
	return nil
}

var _ framework.RequestProcessor = &bodyMutatingPlugin{}

// TestBodyMutation_Unary verifies that when a plugin mutates the body, the response
// includes a BodyMutation with the serialized body and an updated Content-Length header.
func TestBodyMutation_Unary(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	plugin := &bodyMutatingPlugin{fieldName: "injected", fieldValue: "test-value"}
	baseModelToHeaderPlugin, err := test.NewTestBaseModelPlugin()
	require.NoError(t, err, "failed to create base model plugin")
	h := NewBBRHarnessWithPlugins(t, ctx, false, []framework.RequestProcessor{plugin, baseModelToHeaderPlugin})

	body := map[string]any{"prompt": "hello"}
	bodyBytes, _ := json.Marshal(body)

	req := &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestBody{
			RequestBody: &extProcPb.HttpBody{
				Body:        bodyBytes,
				EndOfStream: true,
			},
		},
	}

	resp, err := integration.SendRequest(t, h.Client, req)
	require.NoError(t, err, "unexpected error during request processing")

	wantBody, _ := json.Marshal(map[string]any{
		"prompt":   "hello",
		"injected": "test-value",
	})
	want := &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_RequestBody{
			RequestBody: &extProcPb.BodyResponse{
				Response: &extProcPb.CommonResponse{
					ClearRouteCache: true,
					HeaderMutation: &extProcPb.HeaderMutation{
						SetHeaders: []*envoyCorev3.HeaderValueOption{
							{
								Header: &envoyCorev3.HeaderValue{
									Key:      "X-Gateway-Base-Model-Name",
									RawValue: []byte(""),
								},
							},
							{
								Header: &envoyCorev3.HeaderValue{
									Key:      "Content-Length",
									RawValue: []byte(strconv.Itoa(len(wantBody))),
								},
							},
						},
					},
					BodyMutation: &extProcPb.BodyMutation{
						Mutation: &extProcPb.BodyMutation_Body{
							Body: wantBody,
						},
					},
				},
			},
		},
	}

	envoytest.SortSetHeadersInResponses([]*extProcPb.ProcessingResponse{want})
	envoytest.SortSetHeadersInResponses([]*extProcPb.ProcessingResponse{resp})
	if diff := cmp.Diff(want, resp, protocmp.Transform()); diff != "" {
		t.Errorf("Response mismatch (-want +got): %v", diff)
	}
}

// TestBodyMutation_Streaming verifies the streaming path: when a plugin mutates the body,
// the header response includes Content-Length and the body response carries the mutated bytes.
func TestBodyMutation_Streaming(t *testing.T) {
	t.Parallel()
	ctx := context.Background()

	plugin := &bodyMutatingPlugin{fieldName: "injected", fieldValue: "test-value"}
	baseModelToHeaderPlugin, err := test.NewTestBaseModelPlugin()
	require.NoError(t, err, "failed to create base model plugin")
	h := NewBBRHarnessWithPlugins(t, ctx, true, []framework.RequestProcessor{plugin, baseModelToHeaderPlugin})

	body := map[string]any{"prompt": "hello"}
	bodyBytes, _ := json.Marshal(body)

	reqs := []*extProcPb.ProcessingRequest{
		{
			Request: &extProcPb.ProcessingRequest_RequestHeaders{
				RequestHeaders: &extProcPb.HttpHeaders{
					Headers: &envoyCorev3.HeaderMap{
						Headers: []*envoyCorev3.HeaderValue{
							{Key: "content-type", RawValue: []byte("application/json")},
						},
					},
				},
			},
		},
		{
			Request: &extProcPb.ProcessingRequest_RequestBody{
				RequestBody: &extProcPb.HttpBody{
					Body:        bodyBytes,
					EndOfStream: true,
				},
			},
		},
	}

	wantBody, _ := json.Marshal(map[string]any{
		"prompt":   "hello",
		"injected": "test-value",
	})
	wantResponses := []*extProcPb.ProcessingResponse{
		{
			Response: &extProcPb.ProcessingResponse_RequestHeaders{
				RequestHeaders: &extProcPb.HeadersResponse{
					Response: &extProcPb.CommonResponse{
						ClearRouteCache: true,
						HeaderMutation: &extProcPb.HeaderMutation{
							SetHeaders: []*envoyCorev3.HeaderValueOption{
								{
									Header: &envoyCorev3.HeaderValue{
										Key:      "X-Gateway-Base-Model-Name",
										RawValue: []byte(""),
									},
								},
								{
									Header: &envoyCorev3.HeaderValue{
										Key:      "Content-Length",
										RawValue: []byte(strconv.Itoa(len(wantBody))),
									},
								},
							},
						},
					},
				},
			},
		},
		{
			Response: &extProcPb.ProcessingResponse_RequestBody{
				RequestBody: &extProcPb.BodyResponse{
					Response: &extProcPb.CommonResponse{
						BodyMutation: &extProcPb.BodyMutation{
							Mutation: &extProcPb.BodyMutation_StreamedResponse{
								StreamedResponse: &extProcPb.StreamedBodyResponse{
									Body:        wantBody,
									EndOfStream: true,
								},
							},
						},
					},
				},
			},
		},
	}

	responses, err := integration.StreamedRequest(t, h.Client, reqs, len(wantResponses))
	require.NoError(t, err, "unexpected stream error")

	envoytest.SortSetHeadersInResponses(wantResponses)
	envoytest.SortSetHeadersInResponses(responses)
	if diff := cmp.Diff(wantResponses, responses, protocmp.Transform()); diff != "" {
		t.Errorf("Response mismatch (-want +got): %v", diff)
	}
}
