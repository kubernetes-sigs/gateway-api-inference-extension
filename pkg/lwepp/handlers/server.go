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
	"io"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/structpb"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/lwepp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/lwepp/metadata"
)

// Datastore defines the interface required by the server.
type Datastore interface {
	PoolGet() (*datastore.EndpointPool, error)
	PodList(predicate func(*datastore.Endpoint) bool) []*datastore.Endpoint
}

  func NewStreamingServer(datastore Datastore, disableEndpointSubsetFilter bool) *StreamingServer {
      return &StreamingServer{
          datastore:                   datastore,
          disableEndpointSubsetFilter: disableEndpointSubsetFilter,
      }
  }

// Server implements the Envoy external processing server.
  type StreamingServer struct {
      datastore                   Datastore
      rrIndex                     uint64
      disableEndpointSubsetFilter bool
  }

// RequestContext stores context information during the life time of an HTTP request.
type RequestContext struct {
	TargetEndpoint string
	SelectedPodIP  string
}

func (s *StreamingServer) Process(srv extProcPb.ExternalProcessor_ProcessServer) error {
	ctx := srv.Context()
	logger := log.FromContext(ctx)
	logger.Info("Processing new stream")

	reqCtx := &RequestContext{}

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
			err := s.handleRequestHeaders(ctx, reqCtx, req, v)
			if err != nil {
				logger.Error(err, "Failed to handle request headers")
				return status.Errorf(codes.Internal, "internal error: %v", err)
			}

			resp := &extProcPb.ProcessingResponse{
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
			if err := srv.Send(resp); err != nil {
				return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
			}

		case *extProcPb.ProcessingRequest_ResponseHeaders:
			logger.Info("Received response headers")
			resp, err := s.handleResponseHeaders(ctx, reqCtx, v)
			if err != nil {
				return err
			}
			if err := srv.Send(resp); err != nil {
				return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
			}

		default:
			// Ignore other request types (Body, Trailers)
			logger.V(1).Info("Ignoring request type", "type", v)
		}
	}
}
