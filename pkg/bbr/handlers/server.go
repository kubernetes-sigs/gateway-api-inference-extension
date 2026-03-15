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
	reqenvoy "sigs.k8s.io/gateway-api-inference-extension/pkg/common/envoy/request"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	reqcommon "sigs.k8s.io/gateway-api-inference-extension/pkg/common/request"
	"sigs.k8s.io/gateway-api-inference-extension/version"
)

const (
	ModelField      = "model"
	ModelHeader     = "X-Gateway-Model-Name"
	BaseModelHeader = "X-Gateway-Base-Model-Name"

	contentLengthHeader = "Content-Length"

	requestPluginExtensionPoint  = "request"
	responsePluginExtensionPoint = "response"
)

type Datastore interface {
	GetBaseModel(modelName string) string
}

func NewServer(streaming bool, ds Datastore, requestPlugins []framework.RequestProcessor, responsePlugins []framework.ResponseProcessor) *Server {
	return &Server{
		streaming:       streaming,
		ds:              ds,
		requestPlugins:  requestPlugins,
		responsePlugins: responsePlugins,
	}
}

// Server implements the Envoy external processing server.
// https://www.envoyproxy.io/docs/envoy/latest/api-v3/service/ext_proc/v3/external_processor.proto
type Server struct {
	streaming       bool
	ds              Datastore
	requestPlugins  []framework.RequestProcessor
	responsePlugins []framework.ResponseProcessor
}

// RequestContext stores context information during the lifetime of an HTTP request.
type RequestContext struct {
	RequestReceivedTimestamp  time.Time
	ResponseCompleteTimestamp time.Time
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
		Request:  framework.NewInferenceRequest(),
		Response: framework.NewInferenceResponse(),
	}
	var body []byte
	respStreamedBody := &streamedBody{}

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
			if s.streaming && !v.RequestHeaders.GetEndOfStream() {
				// If streaming and the body is not empty, then headers are handled when processing request body.
				loggerVerbose.Info("Received headers, passing off header processing until body arrives...")
			} else {
				if requestId := reqenvoy.ExtractHeaderValue(v, reqcommon.RequestIdHeaderKey); len(requestId) > 0 {
					logger = logger.WithValues(reqcommon.RequestIdHeaderKey, requestId)
					loggerVerbose = logger.V(logutil.VERBOSE)
					ctx = log.IntoContext(ctx, logger)
				}
				responses, err = s.HandleRequestHeaders(reqCtx, v.RequestHeaders)
			}
		case *extProcPb.ProcessingRequest_RequestBody:
			loggerVerbose.Info("Incoming body chunk", "EoS", v.RequestBody.EndOfStream)
			body = append(body, v.RequestBody.Body...)
			if s.streaming && !v.RequestBody.EndOfStream {
				continue
			}
			responses, err = s.HandleRequestBody(ctx, reqCtx, body)
			loggerVerbose.Info("Processing complete body")
		case *extProcPb.ProcessingRequest_RequestTrailers:
			responses, err = s.HandleRequestTrailers(req.GetRequestTrailers())
		case *extProcPb.ProcessingRequest_ResponseHeaders:
			responses, err = s.HandleResponseHeaders(reqCtx, req.GetResponseHeaders())
		case *extProcPb.ProcessingRequest_ResponseBody:
			loggerVerbose.Info("Incoming response body chunk", "EoS", v.ResponseBody.EndOfStream)
			responses, err = s.processResponseBody(ctx, reqCtx, req.GetResponseBody(), respStreamedBody)
		default:
			logger.V(logutil.DEFAULT).Error(nil, "Unknown Request type", "request", v)
			return status.Error(codes.Unknown, "unknown request type")
		}

		if err != nil {
			if logger.V(logutil.DEBUG).Enabled() {
				logger.V(logutil.DEBUG).Error(err, "Failed to process request", "request", req)
			} else {
				logger.V(logutil.DEFAULT).Error(err, "Failed to process request")
			}
			return status.Errorf(status.Code(err), "failed to handle request: %v", err)
		}

		for _, resp := range responses {
			if logger.V(logutil.DEBUG).Enabled() {
				logger.V(logutil.DEBUG).Info("Response generated", "response", resp)
			} else {
				loggerVerbose.Info("Response generated")
			}
			if err := srv.Send(resp); err != nil {
				logger.V(logutil.DEFAULT).Error(err, "Send failed")
				return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
			}
		}
	}
}

type streamedBody struct {
	body []byte
}

func (s *Server) processResponseBody(ctx context.Context, reqCtx *RequestContext, body *extProcPb.HttpBody, streamedRespBody *streamedBody) ([]*extProcPb.ProcessingResponse, error) {
	loggerVerbose := log.FromContext(ctx).V(logutil.VERBOSE)

	var responseBodyBytes []byte
	if s.streaming {
		streamedRespBody.body = append(streamedRespBody.body, body.Body...)
		if body.EndOfStream {
			loggerVerbose.Info("Flushing response stream buffer")
			responseBodyBytes = streamedRespBody.body
		} else {
			return nil, nil
		}
	} else {
		responseBodyBytes = body.GetBody()
	}

	return s.HandleResponseBody(ctx, reqCtx, responseBodyBytes)
}
