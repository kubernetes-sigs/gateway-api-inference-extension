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

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/codec"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
)

const (
	// VllmGeneratePath is the gRPC method path for vLLM generation.
	VllmGeneratePath = "/vllm.grpc.engine.VllmEngine/Generate"
)

// handleGRPCRequestBody handles the gRPC request body processing.
func (s *StreamingServer) handleGRPCRequestBody(ctx context.Context, reqCtx *RequestContext, body []byte) error {
	logger := log.FromContext(ctx)
	logger.Info("gRPC request body", "path", reqCtx.ReqPath)

	if reqCtx.ReqPath == VllmGeneratePath {
		reqBody, isStreaming, err := codec.ConvertToLLMRequestBody(body)
		if err != nil {
			logger.Error(err, "ConvertToLLMRequestBody error")
			return err
		}
		reqCtx.SchedulingRequestBody = reqBody
		reqCtx.modelServerStreaming = isStreaming
	}
	return nil
}

// handleGRPCResponseTrailers handles the gRPC response trailers.
// Instead of using endOfStream in response frame, gRPC will always send a trailer to indicate the endOfStream.
// Thus, we also set the reqCtx.respBodyResp here to send out the response.
func (s *StreamingServer) handleGRPCResponseTrailers(reqCtx *RequestContext, body []byte) {
	// Ensure the body response is generated if there was any buffered body.
	if len(body) > 0 {
		// Unlike server sent event, gRPC does not need to send empty response with EoS to true.
		// gRPC will use response trailer to determine whether the response is complete.
		reqCtx.respBodyResp = generateResponseBodyResponses(body, true)
	}

	// Send an empty trailers response to complete the stream.
	reqCtx.respTrailerResp = &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_ResponseTrailers{
			ResponseTrailers: &extProcPb.TrailersResponse{},
		},
	}
}

func (s *StreamingServer) handleGRPCResponseBodyModelStreaming(ctx context.Context, reqCtx *RequestContext, responseBytest []byte) {
	logger := log.FromContext(ctx)
	_, err := s.director.HandleResponseBodyStreaming(ctx, reqCtx)
	if err != nil {
		logger.Error(err, "error in HandleResponseBodyStreaming")
	}

	usage, err := codec.ParseUsage(responseBytest)
	if err != nil {
		logger.Error(err, "error in codec.ParseUsage")
	}
	if usage != nil {
		reqCtx.Usage = *usage
		metrics.RecordInputTokens(reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.Usage.PromptTokens)
		metrics.RecordOutputTokens(reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.Usage.CompletionTokens)
		cachedToken := 0
		if reqCtx.Usage.PromptTokenDetails != nil {
			cachedToken = reqCtx.Usage.PromptTokenDetails.CachedTokens
		}
		metrics.RecordPromptCachedTokens(reqCtx.IncomingModelName, reqCtx.TargetModelName, cachedToken)
	}
}
