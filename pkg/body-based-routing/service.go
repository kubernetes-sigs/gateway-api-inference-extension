// Package service provides the implementation of the external processor service.
package service

import (
	"encoding/json"
	"fmt"

	basepb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	eppb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

// GRPCCalloutService implements the gRPC ExternalProcessorServer.
type GRPCCalloutService struct {
	eppb.UnimplementedExternalProcessorServer
}

// Process processes incoming gRPC streams.
func (s *GRPCCalloutService) Process(stream eppb.ExternalProcessor_ProcessServer) error {
	for {
		req, err := stream.Recv()
		if err != nil {
			return err
		}

		fmt.Printf("grpc_callout_service: Received a new request %+v\n", req)

		var response *eppb.ProcessingResponse
		switch {
		case req.GetRequestHeaders() != nil:
			response, err = s.HandleRequestHeaders(req.GetRequestHeaders())
		case req.GetResponseHeaders() != nil:
			response, err = s.HandleResponseHeaders(req.GetResponseHeaders())
		case req.GetRequestBody() != nil:
			response, err = s.HandleRequestBody(req.GetRequestBody())
		case req.GetResponseBody() != nil:
			response, err = s.HandleResponseBody(req.GetResponseBody())
		case req.GetRequestTrailers() != nil:
			response, err = s.HandleRequestTrailers(req.GetRequestTrailers())
		case req.GetResponseTrailers() != nil:
			response, err = s.HandleResponseTrailers(req.GetResponseTrailers())
		}

		if err != nil {
			return err
		}

		if err := stream.Send(response); err != nil {
			return err
		}
	}
}

// HandleRequestBody handles request bodies.
func (s *GRPCCalloutService) HandleRequestBody(body *eppb.HttpBody) (*eppb.ProcessingResponse, error) {
	var data map[string]any
	if err := json.Unmarshal(body.GetBody(), &data); err != nil {
		return nil, err
	}

	modelVal, ok := data["model"]
	if !ok {
		fmt.Print("The incoming request did not contain a model parameter\n")
		return &eppb.ProcessingResponse{
			Response: &eppb.ProcessingResponse_RequestBody{
				RequestBody: &eppb.BodyResponse{},
			},
		}, nil
	}

	modelStr, ok := modelVal.(string)
	if !ok {
		return &eppb.ProcessingResponse{
			Response: &eppb.ProcessingResponse_RequestBody{
				RequestBody: &eppb.BodyResponse{},
			},
		}, fmt.Errorf("the model parameter value %v is not a string", modelVal)
	}

	fmt.Print("grpc_callout_service: Returning mutated request headers\n")
	return &eppb.ProcessingResponse{
		Response: &eppb.ProcessingResponse_RequestBody{
			RequestBody: &eppb.BodyResponse{
				Response: &eppb.CommonResponse{
					// Necessary so that the new headers are used in the routing decision.
					ClearRouteCache: true,
					HeaderMutation: &eppb.HeaderMutation{
						SetHeaders: []*basepb.HeaderValueOption{
							{
								Header: &basepb.HeaderValue{
									Key:   "Model",
									Value: modelStr,
								},
							},
						},
					},
				},
			},
		},
	}, nil
}

// HandleRequestHeaders handles request headers.
func (s *GRPCCalloutService) HandleRequestHeaders(headers *eppb.HttpHeaders) (*eppb.ProcessingResponse, error) {
	return &eppb.ProcessingResponse{
		Response: &eppb.ProcessingResponse_RequestHeaders{
			RequestHeaders: &eppb.HeadersResponse{},
		},
	}, nil
}

// HandleResponseHeaders handles response headers.
func (s *GRPCCalloutService) HandleResponseHeaders(headers *eppb.HttpHeaders) (*eppb.ProcessingResponse, error) {
	return &eppb.ProcessingResponse{
		Response: &eppb.ProcessingResponse_ResponseHeaders{
			ResponseHeaders: &eppb.HeadersResponse{},
		},
	}, nil
}

// HandleResponseBody handles response bodies.
func (s *GRPCCalloutService) HandleResponseBody(body *eppb.HttpBody) (*eppb.ProcessingResponse, error) {
	return &eppb.ProcessingResponse{
		Response: &eppb.ProcessingResponse_ResponseBody{
			ResponseBody: &eppb.BodyResponse{},
		},
	}, nil
}

// HandleRequestTrailers handles request trailers.
func (s *GRPCCalloutService) HandleRequestTrailers(trailers *eppb.HttpTrailers) (*eppb.ProcessingResponse, error) {
	return &eppb.ProcessingResponse{
		Response: &eppb.ProcessingResponse_RequestTrailers{
			RequestTrailers: &eppb.TrailersResponse{},
		},
	}, nil
}

// HandleResponseTrailers handles response trailers.
func (s *GRPCCalloutService) HandleResponseTrailers(trailers *eppb.HttpTrailers) (*eppb.ProcessingResponse, error) {
	return &eppb.ProcessingResponse{
		Response: &eppb.ProcessingResponse_ResponseTrailers{
			ResponseTrailers: &eppb.TrailersResponse{},
		},
	}, nil
}
