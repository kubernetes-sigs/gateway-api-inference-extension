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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"mime"
	"mime/multipart"
	"strings"
	"time"

	eppb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/metrics"
	envoy "sigs.k8s.io/gateway-api-inference-extension/pkg/common/envoy"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"
)

// HandleRequestBody parses the raw body bytes into reqCtx.Request.Body and processes the request.
func (s *Server) HandleRequestBody(ctx context.Context, reqCtx *RequestContext, requestBodyBytes []byte) ([]*eppb.ProcessingResponse, error) {
	logger := log.FromContext(ctx)
	var ret []*eppb.ProcessingResponse

	contentType := getHeader(reqCtx.Request.Headers, "content-type")
	path := getHeader(reqCtx.Request.Headers, ":path")
	if strings.Contains(strings.ToLower(contentType), "multipart/form-data") {
		if !metadata.PathAllowedForMultipartModelExtraction(path) {
			return bodyOnlyResponse(s.streaming, requestBodyBytes, ret)
		}
		model, parseErr := parseModelFromMultipart(requestBodyBytes, contentType)
		if parseErr != nil || model == "" {
			metrics.RecordBodyFieldNotFound("model")
			return bodyOnlyResponse(s.streaming, requestBodyBytes, ret)
		}
		metrics.RecordSuccessCounter()
		reqCtx.Request.SetHeader(ModelHeader, model)
		reqCtx.Request.SetHeader(BaseModelHeader, s.ds.GetBaseModel(model))
		return buildHeaderAndBodyResponse(s.streaming, reqCtx, requestBodyBytes, ret)
	}

	if err := json.Unmarshal(requestBodyBytes, &reqCtx.Request.Body); err != nil {
		return nil, err
	}

	if err := s.runRequestPlugins(ctx, reqCtx.Request); err != nil {
		logger.V(logutil.DEFAULT).Error(err, "failed to execute request plugins")
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

	// TODO temp until this is implemented as plugin
	baseModel := s.ds.GetBaseModel(reqCtx.Request.Headers[ModelHeader])
	reqCtx.Request.SetHeader(BaseModelHeader, baseModel)
	logger.Info("Base model from datastore", "baseModel", baseModel)

	metrics.RecordSuccessCounter()

	if s.streaming {
		ret = append(ret, &eppb.ProcessingResponse{
			Response: &eppb.ProcessingResponse_RequestHeaders{
				RequestHeaders: &eppb.HeadersResponse{
					Response: &eppb.CommonResponse{
						ClearRouteCache: true,
						HeaderMutation: &eppb.HeaderMutation{
							SetHeaders:    envoy.GenerateHeadersMutation(reqCtx.Request.MutatedHeaders()),
							RemoveHeaders: reqCtx.Request.RemovedHeaders(),
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
							SetHeaders:    envoy.GenerateHeadersMutation(reqCtx.Request.MutatedHeaders()),
							RemoveHeaders: reqCtx.Request.RemovedHeaders(),
						},
					},
				},
			},
		},
	}, nil
}

func getHeader(headers map[string]string, key string) string {
	for k, v := range headers {
		if strings.EqualFold(k, key) {
			return v
		}
	}
	return ""
}

func bodyOnlyResponse(streaming bool, body []byte, ret []*eppb.ProcessingResponse) ([]*eppb.ProcessingResponse, error) {
	if streaming {
		ret = append(ret, &eppb.ProcessingResponse{
			Response: &eppb.ProcessingResponse_RequestHeaders{RequestHeaders: &eppb.HeadersResponse{}},
		})
		return addStreamedBodyResponse(ret, body), nil
	}
	return []*eppb.ProcessingResponse{{Response: &eppb.ProcessingResponse_RequestBody{RequestBody: &eppb.BodyResponse{}}}}, nil
}

func buildHeaderAndBodyResponse(streaming bool, reqCtx *RequestContext, body []byte, ret []*eppb.ProcessingResponse) ([]*eppb.ProcessingResponse, error) {
	removed := reqCtx.Request.RemovedHeaders()
	headerMutation := &eppb.HeaderMutation{
		SetHeaders: reqenvoy.GenerateHeadersMutation(reqCtx.Request.MutatedHeaders()),
	}
	if len(removed) > 0 {
		headerMutation.RemoveHeaders = removed
	}
	if streaming {
		ret = append(ret, &eppb.ProcessingResponse{
			Response: &eppb.ProcessingResponse_RequestHeaders{
				RequestHeaders: &eppb.HeadersResponse{
					Response: &eppb.CommonResponse{ClearRouteCache: true, HeaderMutation: headerMutation},
				},
			},
		})
		return addStreamedBodyResponse(ret, body), nil
	}
	return []*eppb.ProcessingResponse{{
		Response: &eppb.ProcessingResponse_RequestBody{
			RequestBody: &eppb.BodyResponse{
				Response: &eppb.CommonResponse{ClearRouteCache: true, HeaderMutation: headerMutation},
			},
		},
	}}, nil
}

func parseModelFromMultipart(body []byte, contentType string) (string, error) {
	_, params, err := mime.ParseMediaType(contentType)
	if err != nil {
		return "", err
	}
	boundary := params["boundary"]
	if boundary == "" {
		return "", nil
	}
	r := multipart.NewReader(bytes.NewReader(body), boundary)
	for {
		p, err := r.NextPart()
		if err != nil {
			break
		}
		if p.FormName() == "model" {
			var buf bytes.Buffer
			if _, err := buf.ReadFrom(p); err != nil {
				return "", err
			}
			return strings.TrimSpace(buf.String()), nil
		}
	}
	return "", nil
}

// runRequestPlugins executes request plugins in the order they were registered.
func (s *Server) runRequestPlugins(ctx context.Context, request *framework.InferenceRequest) error {
	var err error
	for _, plugin := range s.requestPlugins {
		log.FromContext(ctx).V(logutil.VERBOSE).Info("Executing request plugin", "plugin", plugin.TypedName())
		before := time.Now()
		err = plugin.ProcessRequest(ctx, request)
		metrics.RecordPluginProcessingLatency(requestPluginExtensionPoint, plugin.TypedName().Type, plugin.TypedName().Name, time.Since(before))
		if err != nil {
			return fmt.Errorf("failed to execute request plugin '%s' - %w", plugin.TypedName(), err)
		}
	}

	return nil
}

func addStreamedBodyResponse(responses []*eppb.ProcessingResponse, requestBodyBytes []byte) []*eppb.ProcessingResponse {
	commonResponses := envoy.BuildChunkedBodyResponses(requestBodyBytes, true)
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
			reqCtx.Request.Headers[header.Key] = envoy.GetHeaderValue(header)
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
