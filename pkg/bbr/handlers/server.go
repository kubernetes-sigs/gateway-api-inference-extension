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
	"errors"
	"io"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/framework"
	envoy "sigs.k8s.io/gateway-api-inference-extension/pkg/common/envoy"
	errcommon "sigs.k8s.io/gateway-api-inference-extension/pkg/common/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	reqcommon "sigs.k8s.io/gateway-api-inference-extension/pkg/common/request"
	"sigs.k8s.io/gateway-api-inference-extension/version"
)

const (
	contentLengthHeader = "Content-Length"

	requestPluginExtensionPoint  = "request"
	responsePluginExtensionPoint = "response"
)

func NewServer(streaming bool, requestPlugins []framework.RequestProcessor, responsePlugins []framework.ResponseProcessor) *Server {
	return &Server{
		streaming:       streaming,
		requestPlugins:  requestPlugins,
		responsePlugins: responsePlugins,
	}
}

// Server implements the Envoy external processing server.
// https://www.envoyproxy.io/docs/envoy/latest/api-v3/service/ext_proc/v3/external_processor.proto
type Server struct {
	streaming       bool
	requestPlugins  []framework.RequestProcessor
	responsePlugins []framework.ResponseProcessor
}

// RequestContext stores context information during the lifetime of an HTTP request.
type RequestContext struct {
	RequestReceivedTimestamp  time.Time
	ResponseCompleteTimestamp time.Time
	CycleState                *framework.CycleState
	Request                   *framework.InferenceRequest
	Response                  *framework.InferenceResponse
}

func (s *Server) Process(srv extProcPb.ExternalProcessor_ProcessServer) error {
	ctx := srv.Context()

	// Start tracing span for the request
	tracer := otel.Tracer(
		"gateway-api-inference-extension/bbr/extproc",
		trace.WithInstrumentationVersion(version.BuildRef),
		trace.WithInstrumentationAttributes(
			attribute.String("commit-sha", version.CommitSHA),
		),
	)
	ctx, span := tracer.Start(ctx, "gateway.request", trace.WithSpanKind(trace.SpanKindServer))
	defer span.End()

	logger := log.FromContext(ctx)
	loggerVerbose := logger.V(logutil.VERBOSE)
	loggerVerbose.Info("Processing")

	reqCtx := &RequestContext{
		CycleState: framework.NewCycleState(),
		Request:    framework.NewInferenceRequest(),
		Response:   framework.NewInferenceResponse(),
	}

	var requestBody []byte
	var responseBody []byte

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		req, recvErr := srv.Recv()
		if recvErr == io.EOF || errors.Is(recvErr, context.Canceled) {
			return nil
		}
		if recvErr != nil {
			return status.Errorf(codes.Unknown, "cannot receive stream request: %v", recvErr)
		}

		var responses []*extProcPb.ProcessingResponse
		var err error
		switch v := req.Request.(type) {
		case *extProcPb.ProcessingRequest_RequestHeaders:
			if requestId := envoy.ExtractHeaderValue(v, reqcommon.RequestIdHeaderKey); len(requestId) > 0 {
				logger = logger.WithValues(reqcommon.RequestIdHeaderKey, requestId)
				loggerVerbose = logger.V(logutil.VERBOSE)
				ctx = log.IntoContext(ctx, logger)
			}
			if s.streaming && !v.RequestHeaders.GetEndOfStream() {
				_ = s.HandleRequestHeaders(reqCtx, v.RequestHeaders)
				loggerVerbose.Info("captured request headers, deferring response until body arrives...")
			} else {
				responses = s.HandleRequestHeaders(reqCtx, v.RequestHeaders)
				loggerVerbose.Info("processing request headers complete")
			}
		case *extProcPb.ProcessingRequest_RequestBody:
			loggerVerbose.Info("Incoming request body chunk", "EoS", v.RequestBody.EndOfStream)
			requestBody = append(requestBody, v.RequestBody.Body...)
			if s.streaming && !v.RequestBody.EndOfStream {
				continue
			}
			responses, err = s.HandleRequestBody(ctx, reqCtx, requestBody)
			loggerVerbose.Info("processing request body complete")
		case *extProcPb.ProcessingRequest_RequestTrailers:
			responses, err = s.HandleRequestTrailers(v.RequestTrailers)
		case *extProcPb.ProcessingRequest_ResponseHeaders:
			if s.streaming && !v.ResponseHeaders.GetEndOfStream() {
				_ = s.HandleResponseHeaders(reqCtx, v.ResponseHeaders)
				loggerVerbose.Info("captured response headers, deferring response until body arrives...")
			} else {
				responses = s.HandleResponseHeaders(reqCtx, v.ResponseHeaders)
				loggerVerbose.Info("processing response headers complete")
			}
		case *extProcPb.ProcessingRequest_ResponseBody:
			loggerVerbose.Info("Incoming response body chunk", "EoS", v.ResponseBody.EndOfStream)
			responseBody = append(responseBody, v.ResponseBody.Body...)
			if s.streaming && !v.ResponseBody.EndOfStream {
				continue
			}
			responses, err = s.HandleResponseBody(ctx, reqCtx, responseBody)
			loggerVerbose.Info("processing response body complete")
		case *extProcPb.ProcessingRequest_ResponseTrailers:
			responses, err = s.HandleResponseTrailers(v.ResponseTrailers)
		default:
			logger.V(logutil.DEFAULT).Error(nil, "unknown Request type", "request", v)
			return status.Error(codes.Unknown, "unknown request type")
		}

		// Handle the err and fire an immediate response.
		if err != nil {
			if logger.V(logutil.DEBUG).Enabled() {
				logger.V(logutil.DEBUG).Error(err, "failed to process request", "request", req)
			} else {
				logger.V(logutil.DEFAULT).Error(err, "failed to process request")
			}
			resp, err := errcommon.BuildErrResponse(err)
			if err != nil {
				return err
			}
			if sendErr := srv.Send(resp); sendErr != nil {
				logger.V(logutil.DEFAULT).Error(sendErr, "Send failed")
				return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", sendErr)
			}
			return nil
		}

		for _, resp := range responses {
			if err := srv.Send(resp); err != nil {
				logger.V(logutil.DEFAULT).Error(err, "send failed")
				return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
			}
		}
	}
}
