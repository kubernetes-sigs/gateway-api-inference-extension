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
	"fmt"
	"time"

	basepb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	eppb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/common"
	reqenvoy "sigs.k8s.io/gateway-api-inference-extension/pkg/common/envoy/request"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
)

const (
	modelHeader     = "X-Gateway-Model-Name"
	baseModelHeader = "X-Gateway-Base-Model-Name"

	executeExtensionPoint = "Request"
)

// HandleRequestBody parses the raw body bytes into reqCtx.Request.Body and processes the request.
func (s *Server) HandleRequestBody(ctx context.Context, reqCtx *RequestContext, requestBodyBytes []byte) ([]*eppb.ProcessingResponse, error) {
	logger := log.FromContext(ctx)
	var ret []*eppb.ProcessingResponse

	if err := json.Unmarshal(requestBodyBytes, &reqCtx.Request.Body); err != nil {
		return nil, err
	}

	targetModelAny, ok := reqCtx.Request.Body["model"]
	if !ok {
		metrics.RecordModelNotParsedCounter()
		targetModelAny = ""
	}

	targetModel, ok := targetModelAny.(string)
	if !ok {
		metrics.RecordModelNotParsedCounter()
		return nil, errors.New("model is not a string")
	}

	logger.Info("Parsed model name", "model", targetModel)

	if targetModel == "" {
		metrics.RecordModelNotInBodyCounter()
		logger.V(logutil.DEFAULT).Info("Request body does not contain model parameter")
		if s.streaming {
			ret = append(ret, &eppb.ProcessingResponse{
				Response: &eppb.ProcessingResponse_RequestHeaders{
					RequestHeaders: &eppb.HeadersResponse{},
				},
			})
			ret = addStreamedBodyResponse(ret, requestBodyBytes)
			return ret, nil
		} else {
			ret = append(ret, &eppb.ProcessingResponse{
				Response: &eppb.ProcessingResponse_RequestBody{
					RequestBody: &eppb.BodyResponse{},
				},
			})
		}
		return ret, nil
	}

	// TODO: apply updated headers/body to the request.
	if _, _, err := s.executePlugins(ctx, reqCtx.Request.Headers, reqCtx.Request.Body, s.requestPlugins); err != nil {
		return nil, fmt.Errorf("failed to execute request plugins - %w", err)
	}

	metrics.RecordSuccessCounter()
	baseModel := s.ds.GetBaseModel(targetModel)

	logger.Info("Base model from datastore", "baseModel", baseModel)

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
										Key:      modelHeader,
										RawValue: []byte(targetModel),
									},
								},
								{
									Header: &basepb.HeaderValue{
										Key:      baseModelHeader,
										RawValue: []byte(baseModel),
									},
								},
							},
						},
					},
				},
			},
		})
		ret = addStreamedBodyResponse(ret, requestBodyBytes)
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
										Key:      modelHeader,
										RawValue: []byte(targetModel),
									},
								},
								{
									Header: &basepb.HeaderValue{
										Key:      baseModelHeader,
										RawValue: []byte(baseModel),
									},
								},
							},
						},
					},
				},
			},
		},
	}, nil
}

// executePlugins executes BBR plugins in the order they were registered
// and returns the final updated headers and body.
func (s *Server) executePlugins(ctx context.Context, headers map[string]string, body map[string]any,
	plugins []framework.PayloadProcessor) (map[string]string, map[string]any, error) {
	updatedHeaders := headers
	updatedBody := body
	var err error
	for _, plugin := range plugins {
		log.FromContext(ctx).Info("Executing plugin", "plugin", plugin.TypedName())
		before := time.Now()
		updatedHeaders, updatedBody, err = plugin.Execute(ctx, updatedHeaders, updatedBody)
		metrics.RecordPluginProcessingLatency(executeExtensionPoint, plugin.TypedName().Type, plugin.TypedName().Name, time.Since(before))
		if err != nil {
			return nil, nil, fmt.Errorf("failed to execute payload processor %s - %w", plugin.TypedName(), err)
		}
	}

	return updatedHeaders, updatedBody, nil
}

func addStreamedBodyResponse(responses []*eppb.ProcessingResponse, requestBodyBytes []byte) []*eppb.ProcessingResponse {
	commonResponses := common.BuildChunkedBodyResponses(requestBodyBytes, true)
	for _, commonResp := range commonResponses {
		responses = append(responses, &eppb.ProcessingResponse{
			Response: &eppb.ProcessingResponse_RequestBody{
				RequestBody: &eppb.BodyResponse{
					Response: commonResp,
				},
			},
		})
	}
	return responses
}

// HandleRequestHeaders extracts request headers into reqCtx and returns
// the ext-proc header response.
func (s *Server) HandleRequestHeaders(reqCtx *RequestContext, headers *eppb.HttpHeaders) ([]*eppb.ProcessingResponse, error) {
	reqCtx.RequestReceivedTimestamp = time.Now()

	if headers != nil && headers.Headers != nil {
		for _, header := range headers.Headers.Headers {
			reqCtx.Request.Headers[header.Key] = reqenvoy.GetHeaderValue(header)
		}
	}

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
