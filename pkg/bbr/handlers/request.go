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
	"strings"

	basepb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	eppb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/metrics"
	bbrplugins "sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/plugins"
	helpers "sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/utils"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// HandleRequestBody handles request bodies.
func (s *Server) HandleRequestBody(ctx context.Context, requestBodyBytes []byte) ([]*eppb.ProcessingResponse, error) {
	logger := log.FromContext(ctx)
	var ret []*eppb.ProcessingResponse

	allHeaders, mutatedBodyBytes, err := s.requestChain.Run(requestBodyBytes, s.registry)

	if err != nil {
		//TODO: add metric in metrics.go to count "other errors"
		logger.V(logutil.DEFAULT).Info("error processing body", "error", err)
		ret, _ := buildEmptyResponsesForMissingModel(s.streaming, requestBodyBytes)
		return ret, nil
	}

	model, ok := allHeaders[bbrplugins.ModelHeader]

	if !ok {
		//TODO: add metric in metrics.go to count "other errors"
		logger.V(logutil.DEFAULT).Info("manadatory header X-Gateway-Model-Name value is undetermined")
		ret, _ := buildEmptyResponsesForMissingModel(s.streaming, requestBodyBytes)
		return ret, nil
	}

	if strings.TrimSpace(model) == "" {
		metrics.RecordModelNotInBodyCounter()
		ret, _ := buildEmptyResponsesForMissingModel(s.streaming, requestBodyBytes)
		return ret, nil
	}

	//TODO: change to DEBUG
	logger.V(logutil.DEFAULT).Info("model extracted from request body", "model", model)

	metrics.RecordSuccessCounter()

	if s.streaming {
		ret = append(ret, &eppb.ProcessingResponse{
			Response: &eppb.ProcessingResponse_RequestHeaders{
				RequestHeaders: &eppb.HeadersResponse{
					Response: &eppb.CommonResponse{
						ClearRouteCache: true,
						HeaderMutation: &eppb.HeaderMutation{
							SetHeaders: []*basepb.HeaderValueOption{
								{
									Header: &basepb.HeaderValue{
										Key:      bbrplugins.ModelHeader,
										RawValue: []byte(model),
									},
								},
							},
						},
					},
				},
			},
		})
		ret = addStreamedBodyResponse(ret, mutatedBodyBytes)

		//TODO: change to DEBUG
		logger.V(logutil.DEFAULT).Info("RESPONSE", "response", helpers.PrettyPrintResponses(ret))

		return ret, nil
	}

	return []*eppb.ProcessingResponse{
		{
			Response: &eppb.ProcessingResponse_RequestBody{
				RequestBody: &eppb.BodyResponse{
					Response: &eppb.CommonResponse{
						// Necessary so that the new headers are used in the routing decision.
						ClearRouteCache: true,
						HeaderMutation: &eppb.HeaderMutation{
							SetHeaders: []*basepb.HeaderValueOption{
								{
									Header: &basepb.HeaderValue{
										Key:      bbrplugins.ModelHeader,
										RawValue: []byte(model),
									},
								},
							},
						},
						BodyMutation: &eppb.BodyMutation{
							Mutation: &eppb.BodyMutation_Body{
								Body: mutatedBodyBytes,
							},
						},
					},
				},
			},
		},
	}, nil
}

func addStreamedBodyResponse(responses []*eppb.ProcessingResponse, mutatedBodyBytes []byte) []*eppb.ProcessingResponse {
	return append(responses, &eppb.ProcessingResponse{
		Response: &eppb.ProcessingResponse_RequestBody{
			RequestBody: &eppb.BodyResponse{
				Response: &eppb.CommonResponse{
					BodyMutation: &eppb.BodyMutation{
						Mutation: &eppb.BodyMutation_StreamedResponse{
							StreamedResponse: &eppb.StreamedBodyResponse{
								Body:        mutatedBodyBytes,
								EndOfStream: true,
							},
						},
					},
				},
			},
		},
	})
}

// HandleRequestHeaders handles request headers.
func (s *Server) HandleRequestHeaders(headers *eppb.HttpHeaders) ([]*eppb.ProcessingResponse, error) {
	return []*eppb.ProcessingResponse{
		{
			Response: &eppb.ProcessingResponse_RequestHeaders{
				RequestHeaders: &eppb.HeadersResponse{},
			},
		},
	}, nil
}

// HandleRequestTrailers handles request trailers.
func (s *Server) HandleRequestTrailers(trailers *eppb.HttpTrailers) ([]*eppb.ProcessingResponse, error) {
	return []*eppb.ProcessingResponse{
		{
			Response: &eppb.ProcessingResponse_RequestTrailers{
				RequestTrailers: &eppb.TrailersResponse{},
			},
		},
	}, nil
}

// buildEmptyResponsesForMissingModel is a local helper that returns the appropriate empty responses
// for the "model not found" branch depending on streaming mode.
// It is also used to create empty responses in case of other errors related to running plugins on the body
// This is not very clean and MUST be segregated in the future.
// Corresponding metrics should be defined to make different errors observable
func buildEmptyResponsesForMissingModel(streaming bool, requestBodyBytes []byte) ([]*eppb.ProcessingResponse, error) {
	var ret []*eppb.ProcessingResponse

	if streaming {
		// Emit empty headers response, then stream body unchanged.
		ret = append(ret, &eppb.ProcessingResponse{
			Response: &eppb.ProcessingResponse_RequestHeaders{
				RequestHeaders: &eppb.HeadersResponse{},
			},
		})
		ret = addStreamedBodyResponse(ret, requestBodyBytes)
		return ret, nil
	}

	// Non-streaming: emit empty body response.
	ret = append(ret, &eppb.ProcessingResponse{
		Response: &eppb.ProcessingResponse_RequestBody{
			RequestBody: &eppb.BodyResponse{},
		},
	})
	return ret, nil
}
