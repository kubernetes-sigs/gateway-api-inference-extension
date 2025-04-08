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

// Package bbr contains integration tests for the body-based routing extension.
package bbr

import (
	"context"
	"fmt"
	"testing"
	"time"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/testing/protocmp"
	runserver "sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/server"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	integrationutils "sigs.k8s.io/gateway-api-inference-extension/test/integration"
)

var logger = logutil.NewTestLogger().V(logutil.VERBOSE)

func TestBodyBasedRouting(t *testing.T) {
	tests := []struct {
		name        string
		req         *extProcPb.ProcessingRequest
		wantHeaders []*configPb.HeaderValueOption
		wantErr     bool
	}{
		{
			name: "success adding model parameter to header",
			req:  integrationutils.GenerateRequest(logger, "test", "llama"),
			wantHeaders: []*configPb.HeaderValueOption{
				{
					Header: &configPb.HeaderValue{
						Key:      "X-Gateway-Model-Name",
						RawValue: []byte("llama"),
					},
				},
			},
			wantErr: false,
		},
		{
			name:        "no model parameter",
			req:         integrationutils.GenerateRequest(logger, "test1", ""),
			wantHeaders: []*configPb.HeaderValueOption{},
			wantErr:     false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client, cleanup := setUpHermeticServer(false)
			t.Cleanup(cleanup)

			want := &extProcPb.ProcessingResponse{}
			if len(test.wantHeaders) > 0 {
				want.Response = &extProcPb.ProcessingResponse_RequestBody{
					RequestBody: &extProcPb.BodyResponse{
						Response: &extProcPb.CommonResponse{
							HeaderMutation: &extProcPb.HeaderMutation{
								SetHeaders: test.wantHeaders,
							},
							ClearRouteCache: true,
						},
					},
				}
			} else {
				want.Response = &extProcPb.ProcessingResponse_RequestBody{
					RequestBody: &extProcPb.BodyResponse{},
				}
			}

			res, err := integrationutils.SendRequest(t, client, test.req)
			if err != nil && !test.wantErr {
				t.Errorf("Unexpected error, got: %v, want error: %v", err, test.wantErr)
			}
			if diff := cmp.Diff(want, res, protocmp.Transform()); diff != "" {
				t.Errorf("Unexpected response, (-want +got): %v", diff)
			}
		})
	}
}

func TestFullDuplexStreamed_BodyBasedRouting(t *testing.T) {
	tests := []struct {
		name          string
		reqs          []*extProcPb.ProcessingRequest
		wantResponses []*extProcPb.ProcessingResponse
		wantErr       bool
	}{
		{
			name: "success adding model parameter to header",
			reqs: integrationutils.GenerateStreamedRequestSet(logger, "test", "foo"),
			wantResponses: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_RequestHeaders{
						RequestHeaders: &extProcPb.HeadersResponse{
							Response: &extProcPb.CommonResponse{
								ClearRouteCache: true,
								HeaderMutation: &extProcPb.HeaderMutation{
									SetHeaders: []*configPb.HeaderValueOption{
										{
											Header: &configPb.HeaderValue{
												Key:      "X-Gateway-Model-Name",
												RawValue: []byte("foo"),
											},
										},
									}},
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
											Body:        []byte("{\"max_tokens\":100,\"model\":\"foo\",\"prompt\":\"test\",\"temperature\":0}"),
											EndOfStream: true,
										},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "success adding model parameter to header with multiple body chunks",
			reqs: []*extProcPb.ProcessingRequest{
				{
					Request: &extProcPb.ProcessingRequest_RequestHeaders{
						RequestHeaders: &extProcPb.HttpHeaders{
							Headers: &configPb.HeaderMap{
								Headers: []*configPb.HeaderValue{
									{
										Key:   "hi",
										Value: "mom",
									},
								},
							},
						},
					},
				},
				{
					Request: &extProcPb.ProcessingRequest_RequestBody{
						RequestBody: &extProcPb.HttpBody{Body: []byte("{\"max_tokens\":100,\"model\":\"sql-lo"), EndOfStream: false},
					},
				},
				{
					Request: &extProcPb.ProcessingRequest_RequestBody{
						RequestBody: &extProcPb.HttpBody{Body: []byte("ra-sheddable\",\"prompt\":\"test\",\"temperature\":0}"), EndOfStream: true},
					},
				},
			},
			wantResponses: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_RequestHeaders{
						RequestHeaders: &extProcPb.HeadersResponse{
							Response: &extProcPb.CommonResponse{
								ClearRouteCache: true,
								HeaderMutation: &extProcPb.HeaderMutation{
									SetHeaders: []*configPb.HeaderValueOption{
										{
											Header: &configPb.HeaderValue{
												Key:      "X-Gateway-Model-Name",
												RawValue: []byte("sql-lora-sheddable"),
											},
										},
									}},
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
											Body:        []byte("{\"max_tokens\":100,\"model\":\"sql-lora-sheddable\",\"prompt\":\"test\",\"temperature\":0}"),
											EndOfStream: true,
										},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name: "no model parameter",
			reqs: integrationutils.GenerateStreamedRequestSet(logger, "test", ""),
			wantResponses: []*extProcPb.ProcessingResponse{
				{
					Response: &extProcPb.ProcessingResponse_RequestHeaders{
						RequestHeaders: &extProcPb.HeadersResponse{},
					},
				},
				{
					Response: &extProcPb.ProcessingResponse_RequestBody{
						RequestBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{
								BodyMutation: &extProcPb.BodyMutation{
									Mutation: &extProcPb.BodyMutation_StreamedResponse{
										StreamedResponse: &extProcPb.StreamedBodyResponse{
											Body:        []byte("{\"max_tokens\":100,\"prompt\":\"test\",\"temperature\":0}"),
											EndOfStream: true,
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
			client, cleanup := setUpHermeticServer(true)
			t.Cleanup(cleanup)

			responses, err := integrationutils.StreamedRequest(t, client, test.reqs, len(test.wantResponses))
			if err != nil && !test.wantErr {
				t.Errorf("Unexpected error, got: %v, want error: %v", err, test.wantErr)
			}

			if diff := cmp.Diff(test.wantResponses, responses, protocmp.Transform()); diff != "" {
				t.Errorf("Unexpected response, (-want +got): %v", diff)
			}
		})
	}
}

func setUpHermeticServer(streaming bool) (client extProcPb.ExternalProcessor_ProcessClient, cleanup func()) {
	port := 9004

	serverCtx, stopServer := context.WithCancel(context.Background())
	serverRunner := runserver.NewDefaultExtProcServerRunner(port, false)
	serverRunner.SecureServing = false
	serverRunner.Streaming = streaming

	go func() {
		if err := serverRunner.AsRunnable(logger.WithName("ext-proc")).Start(serverCtx); err != nil {
			logutil.Fatal(logger, err, "Failed to start ext-proc server")
		}
	}()

	address := fmt.Sprintf("localhost:%v", port)
	// Create a grpc connection
	conn, err := grpc.NewClient(address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		logutil.Fatal(logger, err, "Failed to connect", "address", address)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	client, err = extProcPb.NewExternalProcessorClient(conn).Process(ctx)
	if err != nil {
		logutil.Fatal(logger, err, "Failed to create client")
	}
	return client, func() {
		cancel()
		conn.Close()
		stopServer()

		// wait a little until the goroutines actually exit
		time.Sleep(5 * time.Second)
	}
}
