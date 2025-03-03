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
	"fmt"

	basepb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	eppb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// HandleRequestBody handles request bodies.
func (s *Server) HandleRequestBody(ctx context.Context, body *eppb.HttpBody) (*eppb.ProcessingResponse, error) {
	logger := log.FromContext(ctx)

	var data map[string]any
	if err := json.Unmarshal(body.GetBody(), &data); err != nil {
		return nil, err
	}

	modelVal, ok := data["model"]
	if !ok {
		logger.V(logutil.DEFAULT).Info("Request body does not contain model parameter")
		return &eppb.ProcessingResponse{
			Response: &eppb.ProcessingResponse_RequestBody{
				RequestBody: &eppb.BodyResponse{},
			},
		}, nil
	}

	modelStr, ok := modelVal.(string)
	if !ok {
		logger.V(logutil.DEFAULT).Info("Model parameter value is not a string")
		return &eppb.ProcessingResponse{
			Response: &eppb.ProcessingResponse_RequestBody{
				RequestBody: &eppb.BodyResponse{},
			},
		}, fmt.Errorf("the model parameter value %v is not a string", modelVal)
	}

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
									Key:      "X-Gateway-Model-Name",
									RawValue: []byte(modelStr),
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
func (s *Server) HandleRequestHeaders(headers *eppb.HttpHeaders) (*eppb.ProcessingResponse, error) {
	return &eppb.ProcessingResponse{
		Response: &eppb.ProcessingResponse_RequestHeaders{
			RequestHeaders: &eppb.HeadersResponse{},
		},
	}, nil
}

// HandleRequestTrailers handles request trailers.
func (s *Server) HandleRequestTrailers(trailers *eppb.HttpTrailers) (*eppb.ProcessingResponse, error) {
	return &eppb.ProcessingResponse{
		Response: &eppb.ProcessingResponse_RequestTrailers{
			RequestTrailers: &eppb.TrailersResponse{},
		},
	}, nil
}
