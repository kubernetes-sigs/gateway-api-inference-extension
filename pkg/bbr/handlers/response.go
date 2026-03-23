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
	"strconv"
	"time"

	eppb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/metrics"
	envoy "sigs.k8s.io/gateway-api-inference-extension/pkg/common/envoy"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
)

// HandleResponseHeaders extracts response headers into reqCtx and returns
// the ext-proc header response.
func (s *Server) HandleResponseHeaders(reqCtx *RequestContext, headers *eppb.HttpHeaders) ([]*eppb.ProcessingResponse, error) {
	if headers != nil && headers.Headers != nil {
		for _, header := range headers.Headers.Headers {
			reqCtx.Response.Headers[header.Key] = envoy.GetHeaderValue(header)
		}
	}

	if !s.streaming || headers.GetEndOfStream() {
		return []*eppb.ProcessingResponse{
			{
				Response: &eppb.ProcessingResponse_ResponseHeaders{
					ResponseHeaders: &eppb.HeadersResponse{},
				},
			},
		}, nil
	}

	// In streaming mode with a body pending, defer the response —
	// HandleResponseBody will send it together with the body response,
	// mirroring the request-side pattern.
	return nil, nil
}

// HandleResponseBody handles response bodies by executing response plugins in order.
func (s *Server) HandleResponseBody(ctx context.Context, reqCtx *RequestContext, responseBodyBytes []byte) ([]*eppb.ProcessingResponse, error) {
	logger := log.FromContext(ctx)
	if len(s.responsePlugins) == 0 {
		if s.streaming {
			return s.generateEmptyResponseBodyResponse(responseBodyBytes), nil
		}
		return []*eppb.ProcessingResponse{
			{
				Response: &eppb.ProcessingResponse_ResponseBody{
					ResponseBody: &eppb.BodyResponse{},
				},
			},
		}, nil
	}

	if err := json.Unmarshal(responseBodyBytes, &reqCtx.Response.Body); err != nil {
		logger.Error(err, "Failed to parse response body as JSON, skipping response plugins")
		if s.streaming {
			return s.generateEmptyResponseBodyResponse(responseBodyBytes), nil
		}
		return []*eppb.ProcessingResponse{
			{
				Response: &eppb.ProcessingResponse_ResponseBody{
					ResponseBody: &eppb.BodyResponse{},
				},
			},
		}, nil
	}

	if err := s.runResponsePlugins(ctx, reqCtx.CycleState, reqCtx.Response); err != nil {
		return nil, fmt.Errorf("failed to execute response plugins - %w", err)
	}

	bodyMutated := reqCtx.Response.BodyMutated()
	var mutatedBytes []byte
	if bodyMutated {
		var err error
		mutatedBytes, err = json.Marshal(reqCtx.Response.Body)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal mutated response body - %w", err)
		}
		reqCtx.Response.SetHeader(contentLengthHeader, strconv.Itoa(len(mutatedBytes)))
	}

	if s.streaming {
		var ret []*eppb.ProcessingResponse
		ret = append(ret, &eppb.ProcessingResponse{
			Response: &eppb.ProcessingResponse_ResponseHeaders{
				ResponseHeaders: &eppb.HeadersResponse{
					Response: &eppb.CommonResponse{
						ClearRouteCache: true,
						HeaderMutation: &eppb.HeaderMutation{
							SetHeaders:    envoy.GenerateHeadersMutation(reqCtx.Response.MutatedHeaders()),
							RemoveHeaders: reqCtx.Response.RemovedHeaders(),
						},
					},
				},
			},
		})
		if bodyMutated {
			ret = envoy.AddStreamedResponseBody(ret, mutatedBytes)
		} else {
			ret = envoy.AddStreamedResponseBody(ret, responseBodyBytes)
		}
		return ret, nil
	}

	response := &eppb.CommonResponse{
		ClearRouteCache: true,
		HeaderMutation: &eppb.HeaderMutation{
			SetHeaders:    envoy.GenerateHeadersMutation(reqCtx.Response.MutatedHeaders()),
			RemoveHeaders: reqCtx.Response.RemovedHeaders(),
		},
	}
	if bodyMutated {
		response.BodyMutation = &eppb.BodyMutation{
			Mutation: &eppb.BodyMutation_Body{
				Body: mutatedBytes,
			},
		}
	}

	return []*eppb.ProcessingResponse{
		{
			Response: &eppb.ProcessingResponse_ResponseBody{
				ResponseBody: &eppb.BodyResponse{
					Response: response,
				},
			},
		},
	}, nil
}

// generateEmptyResponseBodyResponse builds a streaming response with an empty
// ResponseHeaders followed by chunked body responses via AddStreamedResponseBody.
func (s *Server) generateEmptyResponseBodyResponse(responseBodyBytes []byte) []*eppb.ProcessingResponse {
	responses := []*eppb.ProcessingResponse{
		{
			Response: &eppb.ProcessingResponse_ResponseHeaders{
				ResponseHeaders: &eppb.HeadersResponse{},
			},
		},
	}
	responses = envoy.AddStreamedResponseBody(responses, responseBodyBytes)
	return responses
}

// HandleResponseTrailers handles response trailers.
func (s *Server) HandleResponseTrailers(trailers *eppb.HttpTrailers) ([]*eppb.ProcessingResponse, error) {
	return []*eppb.ProcessingResponse{
		{
			Response: &eppb.ProcessingResponse_ResponseTrailers{
				ResponseTrailers: &eppb.TrailersResponse{},
			},
		},
	}, nil
}

// runResponsePlugins executes response plugins in the order they were registered.
func (s *Server) runResponsePlugins(ctx context.Context, cycleState *framework.CycleState, response *framework.InferenceResponse) error {
	var err error
	for _, plugin := range s.responsePlugins {
		log.FromContext(ctx).V(logutil.VERBOSE).Info("Executing response plugin", "plugin", plugin.TypedName())
		before := time.Now()
		err = plugin.ProcessResponse(ctx, cycleState, response)
		metrics.RecordPluginProcessingLatency(responsePluginExtensionPoint, plugin.TypedName().Type, plugin.TypedName().Name, time.Since(before))
		if err != nil {
			return fmt.Errorf("failed to execute response plugin '%s' - %w", plugin.TypedName(), err)
		}
	}

	return nil
}
