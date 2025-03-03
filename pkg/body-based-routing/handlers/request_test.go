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
	"testing"

	basepb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/testing/protocmp"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	bodyWithModel = `
	{
		"model": "foo",
		"prompt": "Tell me a joke"
	}
	`
	bodyWithModelNoStr = `
	{
		"model": 1,
		"prompt": "Tell me a joke"
	}
	`
	bodyWithoutModel = `
	{
		"prompt": "Tell me a joke"
	}
	`
)

func TestHandleRequestBody(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	tests := []struct {
		name    string
		body    *extProcPb.HttpBody
		want    *extProcPb.ProcessingResponse
		wantErr bool
	}{
		{
			name: "malformed body",
			body: &extProcPb.HttpBody{
				Body: []byte("malformed json"),
			},
			wantErr: true,
		},
		{
			name: "model not found",
			body: &extProcPb.HttpBody{
				Body: []byte(bodyWithoutModel),
			},
			want: &extProcPb.ProcessingResponse{
				Response: &extProcPb.ProcessingResponse_RequestBody{
					RequestBody: &extProcPb.BodyResponse{},
				},
			},
		},
		{
			name: "model is not string",
			body: &extProcPb.HttpBody{
				Body: []byte(bodyWithModelNoStr),
			},
			wantErr: true,
		},
		{
			name: "success",
			body: &extProcPb.HttpBody{
				Body: []byte(bodyWithModel),
			},
			want: &extProcPb.ProcessingResponse{
				Response: &extProcPb.ProcessingResponse_RequestBody{
					RequestBody: &extProcPb.BodyResponse{
						Response: &extProcPb.CommonResponse{
							// Necessary so that the new headers are used in the routing decision.
							ClearRouteCache: true,
							HeaderMutation: &extProcPb.HeaderMutation{
								SetHeaders: []*basepb.HeaderValueOption{
									{
										Header: &basepb.HeaderValue{
											Key:      "X-Gateway-Model-Name",
											RawValue: []byte("foo"),
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := &Server{}
			resp, err := server.HandleRequestBody(ctx, test.body)
			if err != nil {
				if !test.wantErr {
					t.Fatalf("HandleRequestBody returned unexpected error: %v, want %v", err, test.wantErr)
				}
				return
			}

			if diff := cmp.Diff(test.want, resp, protocmp.Transform()); diff != "" {
				t.Errorf("HandleRequestBody returned unexpected response, diff(-want, +got): %v", diff)
			}
		})
	}
}
