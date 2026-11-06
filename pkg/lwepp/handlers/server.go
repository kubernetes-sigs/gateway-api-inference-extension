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
	"io"
	"net"
	"sync/atomic"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/structpb"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/common/envoy"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/lwepp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/lwepp/metadata"
)

// Datastore defines the interface required by the server.
type Datastore interface {
	PoolGet() (*datastore.EndpointPool, error)
	PodList(predicate func(*datastore.Endpoint) bool) []*datastore.Endpoint
}

func NewStreamingServer(datastore Datastore) *StreamingServer {
	return &StreamingServer{
		datastore: datastore,
		picker:    &RoundRobinPicker{},
	}
}

// StreamingServer implements the Envoy external processing server.
type StreamingServer struct {
	datastore Datastore
	picker    EndpointPicker
}

// RequestContext stores context information during the life time of an HTTP request.
type RequestContext struct {
	TargetEndpoint string
	SelectedPodIP  string
	Candidates     []*datastore.Endpoint
	Headers        map[string][]string
}

// PickRequest contains metadata from the incoming request to guide endpoint selection.
type PickRequest struct {
	Headers map[string][]string
	Body    []byte
	Model   string
}

// PickResult holds the selected backend endpoint information.
type PickResult struct {
	Endpoint     string            // Primary endpoint (ip:port)
	Fallbacks    []string          // Optional fallback endpoints
	MutatedBody  []byte            // If non-nil, replaces the request body forwarded to Envoy
	ExtraHeaders map[string]string // Additional headers to set on the request
}

// EndpointPicker defines a strategy for selecting backend inference endpoints.
type EndpointPicker interface {
	Pick(ctx context.Context, req *PickRequest, endpoints []*datastore.Endpoint) (*PickResult, error)
}

// RoundRobinPicker implements a basic round-robin endpoint selection strategy.
type RoundRobinPicker struct {
	rrIndex uint64
}

// Pick selects an endpoint based on round-robin ordering.
func (p *RoundRobinPicker) Pick(ctx context.Context, req *PickRequest, endpoints []*datastore.Endpoint) (*PickResult, error) {
	if len(endpoints) == 0 {
		return nil, status.Errorf(codes.Unavailable, "no endpoints available")
	}

	index := atomic.AddUint64(&p.rrIndex, 1)
	selectedPod := endpoints[index%uint64(len(endpoints))]

	return &PickResult{
		Endpoint: net.JoinHostPort(selectedPod.Address, selectedPod.Port),
	}, nil
}

const maxRequestBodySize = 10 * 1024 * 1024 // 10MB

// sendUnavailable answers the request with an ImmediateResponse carrying HTTP 503, as the
// endpoint picker protocol requires when there is no ready endpoint or none of the endpoints
// in the subset hint is eligible. Failing the stream instead lets a data plane configured to
// fail open route the request to an endpoint outside the subset.
func sendUnavailable(ctx context.Context, srv extProcPb.ExternalProcessor_ProcessServer, err error) error {
	log.FromContext(ctx).Info("No endpoint available, responding with 503", "reason", status.Convert(err).Message())
	resp := &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &extProcPb.ImmediateResponse{
				Status: &envoyTypePb.HttpStatus{Code: envoyTypePb.StatusCode_ServiceUnavailable},
				Body:   []byte(status.Convert(err).Message()),
			},
		},
	}
	if sendErr := srv.Send(resp); sendErr != nil {
		return status.Errorf(codes.Unknown, "failed to send immediate response back to Envoy: %v", sendErr)
	}
	return nil
}

func (s *StreamingServer) Process(srv extProcPb.ExternalProcessor_ProcessServer) error {
	ctx := srv.Context()
	logger := log.FromContext(ctx).WithName("ext-proc")
	ctx = log.IntoContext(ctx, logger)
	logger.Info("Processing new stream")

	reqCtx := &RequestContext{}
	var body []byte
	// Request body chunks received while the request headers response is still deferred.
	var pendingBody []byte
	var err error
	headersDeferred := false

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		req, recvErr := srv.Recv()
		if recvErr == io.EOF || status.Code(recvErr) == codes.Canceled {
			return nil
		}
		if recvErr != nil {
			return status.Errorf(codes.Unknown, "cannot receive stream request: %v", recvErr)
		}

		switch v := req.Request.(type) {
		case *extProcPb.ProcessingRequest_RequestHeaders:
			logger.Info("Received request headers")
			err = s.handleRequestHeaders(ctx, reqCtx, req, v)
			if status.Code(err) == codes.Unavailable {
				return sendUnavailable(ctx, srv, err)
			}
			if err != nil {
				logger.Error(err, "Failed to handle request headers")
				return status.Errorf(codes.Internal, "internal error: %v", err)
			}

			// Keep correlation local to this ext_proc stream, including asynchronous responses.
			if ids := reqCtx.Headers[metadata.ConformanceRequestIDHeader]; len(ids) == 1 && len(ids[0]) <= 128 {
				ctx = log.IntoContext(ctx, logger.WithValues(metadata.ConformanceRequestIDLogKey, ids[0]))
			}

			var resp *extProcPb.ProcessingResponse
			if v.RequestHeaders.EndOfStream {
				err = s.pickEndpoint(ctx, reqCtx, nil)
				if status.Code(err) == codes.Unavailable {
					return sendUnavailable(ctx, srv, err)
				}
				if err != nil {
					logger.Error(err, "Failed to pick endpoint")
					return status.Errorf(codes.Internal, "internal error: %v", err)
				}

				resp = &extProcPb.ProcessingResponse{
					Response: &extProcPb.ProcessingResponse_RequestHeaders{
						RequestHeaders: &extProcPb.HeadersResponse{
							Response: &extProcPb.CommonResponse{
								ClearRouteCache: true,
								HeaderMutation: &extProcPb.HeaderMutation{
									SetHeaders: []*configPb.HeaderValueOption{
										{
											Header: &configPb.HeaderValue{
												Key:      metadata.DestinationEndpointKey,
												RawValue: []byte(reqCtx.TargetEndpoint),
											},
										},
										{
											Header: &configPb.HeaderValue{
												Key:      "X-Echo-Set-Header",
												RawValue: []byte(metadata.ConformanceTestResultHeader + ":" + reqCtx.TargetEndpoint),
											},
										},
									},
								},
							},
						},
					},
					DynamicMetadata: &structpb.Struct{
						Fields: map[string]*structpb.Value{
							metadata.DestinationEndpointNamespace: structpb.NewStructValue(&structpb.Struct{
								Fields: map[string]*structpb.Value{
									metadata.DestinationEndpointKey: structpb.NewStringValue(reqCtx.TargetEndpoint),
								},
							}),
						},
					},
				}
			} else {
				headersDeferred = true
			}

			if resp != nil {
				if err := srv.Send(resp); err != nil {
					return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
				}
			}

		case *extProcPb.ProcessingRequest_RequestBody:
			logger.V(1).Info("Received request body", "endOfStream", v.RequestBody.EndOfStream)
			if len(body)+len(v.RequestBody.Body) > maxRequestBodySize {
				logger.Error(nil, "Request body size limit exceeded", "limit", maxRequestBodySize, "currentSize", len(body))
				return status.Errorf(codes.ResourceExhausted, "request body size limit of %d bytes exceeded", maxRequestBodySize)
			}
			body = append(body, v.RequestBody.Body...)
			pendingBody = append(pendingBody, v.RequestBody.Body...)

			// Envoy treats a request body response that arrives before the response to the
			// request headers as out of order, marks it spurious and tears the stream down.
			// The endpoint can only be picked once the whole body has arrived, so while the
			// headers response is still deferred the body chunks are held back and flushed
			// after it has been sent.
			if headersDeferred && !v.RequestBody.EndOfStream {
				continue
			}

			if v.RequestBody.EndOfStream {
				err = s.pickEndpoint(ctx, reqCtx, body)
				if status.Code(err) == codes.Unavailable {
					return sendUnavailable(ctx, srv, err)
				}
				if err != nil {
					logger.Error(err, "Failed to pick endpoint")
					return status.Errorf(codes.Internal, "internal error: %v", err)
				}
				body = nil

				if headersDeferred {
					headerResp := &extProcPb.ProcessingResponse{
						Response: &extProcPb.ProcessingResponse_RequestHeaders{
							RequestHeaders: &extProcPb.HeadersResponse{
								Response: &extProcPb.CommonResponse{
									ClearRouteCache: true,
									HeaderMutation: &extProcPb.HeaderMutation{
										SetHeaders: []*configPb.HeaderValueOption{
											{
												Header: &configPb.HeaderValue{
													Key:      metadata.DestinationEndpointKey,
													RawValue: []byte(reqCtx.TargetEndpoint),
												},
											},
											{
												Header: &configPb.HeaderValue{
													Key:      "X-Echo-Set-Header",
													RawValue: []byte(metadata.ConformanceTestResultHeader + ":" + reqCtx.TargetEndpoint),
												},
											},
										},
									},
								},
							},
						},
						DynamicMetadata: &structpb.Struct{
							Fields: map[string]*structpb.Value{
								metadata.DestinationEndpointNamespace: structpb.NewStructValue(&structpb.Struct{
									Fields: map[string]*structpb.Value{
										metadata.DestinationEndpointKey: structpb.NewStringValue(reqCtx.TargetEndpoint),
									},
								}),
							},
						},
					}
					if err := srv.Send(headerResp); err != nil {
						return status.Errorf(codes.Unknown, "failed to send deferred headers response back to Envoy: %v", err)
					}
					headersDeferred = false
				}
			}

			for _, commonResp := range envoy.BuildChunkedBodyResponses(pendingBody, v.RequestBody.EndOfStream) {
				resp := &extProcPb.ProcessingResponse{
					Response: &extProcPb.ProcessingResponse_RequestBody{
						RequestBody: &extProcPb.BodyResponse{
							Response: commonResp,
						},
					},
				}

				if err := srv.Send(resp); err != nil {
					return status.Errorf(codes.Unknown, "failed to send body response back to Envoy: %v", err)
				}
			}
			pendingBody = nil

		case *extProcPb.ProcessingRequest_ResponseHeaders:
			logger.Info("Received response headers")
			resp := s.handleResponseHeaders(ctx, req, v)
			if err := srv.Send(resp); err != nil {
				return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
			}

		case *extProcPb.ProcessingRequest_ResponseBody:
			logger.V(1).Info("Received response body", "endOfStream", v.ResponseBody.EndOfStream)
			for _, commonResp := range envoy.BuildChunkedBodyResponses(v.ResponseBody.Body, v.ResponseBody.EndOfStream) {
				resp := &extProcPb.ProcessingResponse{
					Response: &extProcPb.ProcessingResponse_ResponseBody{
						ResponseBody: &extProcPb.BodyResponse{
							Response: commonResp,
						},
					},
				}

				if err := srv.Send(resp); err != nil {
					return status.Errorf(codes.Unknown, "failed to send response body back to Envoy: %v", err)
				}
			}

		default:
			// Ignore other request types (Trailers)
			logger.V(1).Info("Ignoring request type", "type", v)
		}
	}
}
