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
	"strconv"
	"time"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/protobuf/types/known/structpb"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
)

func (s *StreamingServer) HandleRequestHeaders(ctx context.Context, reqCtx *RequestContext, req *extProcPb.ProcessingRequest_RequestHeaders) error {
	reqCtx.RequestReceivedTimestamp = time.Now()

	// an EoS in the request headers means this request has no body or trailers.
	if req.RequestHeaders.EndOfStream {
		// We will route this request to a random pod as this is assumed to just be a GET
		// More context: https://github.com/kubernetes-sigs/gateway-api-inference-extension/pull/526
		// The above PR will address endpoint admission, but currently any request without a body will be
		// routed to a random upstream pod.
		pod := s.director.GetRandomPod()
		if pod == nil {
			return errutil.Error{Code: errutil.Internal, Msg: "no pods available in datastore"}
		}
		pool, err := s.datastore.PoolGet()
		if err != nil {
			return err
		}
		reqCtx.TargetEndpoint = pod.Address + ":" + strconv.Itoa(int(pool.Spec.TargetPortNumber))
		reqCtx.RequestSize = 0
		reqCtx.reqHeaderResp = s.generateRequestHeaderResponse(reqCtx)
		return nil
	}

	for _, header := range req.RequestHeaders.Headers.Headers {
		if header.RawValue != nil {
			reqCtx.Request.Headers[header.Key] = string(header.RawValue)
		} else {
			reqCtx.Request.Headers[header.Key] = header.Value
		}
	}
	return nil
}

func (s *StreamingServer) generateRequestBodyResponse(requestBodyBytes []byte) *extProcPb.ProcessingResponse {
	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_RequestBody{
			RequestBody: &extProcPb.BodyResponse{
				Response: &extProcPb.CommonResponse{
					BodyMutation: &extProcPb.BodyMutation{
						Mutation: &extProcPb.BodyMutation_StreamedResponse{
							StreamedResponse: &extProcPb.StreamedBodyResponse{
								Body:        requestBodyBytes,
								EndOfStream: true,
							},
						},
					},
				},
			},
		},
	}
}

func (s *StreamingServer) generateRequestHeaderResponse(reqCtx *RequestContext) *extProcPb.ProcessingResponse {
	// The Endpoint Picker supports two approaches to communicating the target endpoint, as a request header
	// and as an unstructure ext-proc response metadata key/value pair. This enables different integration
	// options for gateway providers.
	return &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_RequestHeaders{
			RequestHeaders: &extProcPb.HeadersResponse{
				Response: &extProcPb.CommonResponse{
					ClearRouteCache: true,
					HeaderMutation: &extProcPb.HeaderMutation{
						SetHeaders: s.generateHeaders(reqCtx),
					},
				},
			},
		},
		DynamicMetadata: s.generateMetadata(reqCtx.TargetEndpoint),
	}
}

func (s *StreamingServer) generateHeaders(reqCtx *RequestContext) []*configPb.HeaderValueOption {
	// can likely refactor these two bespoke headers to be updated in PostDispatch, to centralize logic.
	headers := []*configPb.HeaderValueOption{
		{
			Header: &configPb.HeaderValue{
				Key:      s.destinationEndpointHintKey,
				RawValue: []byte(reqCtx.TargetEndpoint),
			},
		},
	}
	if reqCtx.RequestSize > 0 {
		// We need to update the content length header if the body is mutated, see Envoy doc:
		// https://www.envoyproxy.io/docs/envoy/latest/api-v3/extensions/filters/http/ext_proc/v3/processing_mode.proto
		headers = append(headers, &configPb.HeaderValueOption{
			Header: &configPb.HeaderValue{
				Key:      "Content-Length",
				RawValue: []byte(strconv.Itoa(reqCtx.RequestSize)),
			},
		})
	}

	// include all headers
	for key, value := range reqCtx.Request.Headers {
		headers = append(headers, &configPb.HeaderValueOption{
			Header: &configPb.HeaderValue{
				Key:      key,
				RawValue: []byte(value),
			},
		})
	}
	return headers
}

func (s *StreamingServer) generateMetadata(endpoint string) *structpb.Struct {
	targetEndpointValue := &structpb.Struct{
		Fields: map[string]*structpb.Value{
			s.destinationEndpointHintKey: {
				Kind: &structpb.Value_StringValue{
					StringValue: endpoint,
				},
			},
		},
	}
	dynamicMetadata := targetEndpointValue
	if s.destinationEndpointHintMetadataNamespace != "" {
		// If a namespace is defined, wrap the selected endpoint with that.
		dynamicMetadata = &structpb.Struct{
			Fields: map[string]*structpb.Value{
				s.destinationEndpointHintMetadataNamespace: {
					Kind: &structpb.Value_StructValue{
						StructValue: targetEndpointValue,
					},
				},
			},
		}
	}
	return dynamicMetadata
}
