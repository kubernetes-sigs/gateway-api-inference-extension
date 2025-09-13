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

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
)

func NewServer(streaming bool) *Server {
	return &Server{streaming: streaming}
}

// Server implements the Envoy external processing server.
// https://www.envoyproxy.io/docs/envoy/latest/api-v3/service/ext_proc/v3/external_processor.proto
type Server struct {
	streaming bool
}

func (s *Server) Process(srv extProcPb.ExternalProcessor_ProcessServer) error {
	ctx := srv.Context()
	logger := log.FromContext(ctx)
	loggerVerbose := logger.V(logutil.VERBOSE)
	loggerVerbose.Info("Processing")

	streamedBody := &streamedBody{}

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
			if s.streaming && !req.GetRequestHeaders().GetEndOfStream() {
				// If streaming and the body is not empty, then headers are handled when processing request body.
				loggerVerbose.Info("Received headers, passing off header processing until body arrives...")
			} else {
				if requestId := requtil.ExtractHeaderValue(v, requtil.RequestIdHeaderKey); len(requestId) > 0 {
					logger = logger.WithValues(requtil.RequestIdHeaderKey, requestId)
					loggerVerbose = logger.V(logutil.VERBOSE)
					ctx = log.IntoContext(ctx, logger)
				}
				responses, err = s.HandleRequestHeaders(req.GetRequestHeaders())
			}
		case *extProcPb.ProcessingRequest_RequestBody:
			if logger.V(logutil.DEBUG).Enabled() {
				logger.V(logutil.DEBUG).Info("Incoming body chunk", "body", string(v.RequestBody.Body), "EoS", v.RequestBody.EndOfStream)
			} else {
				loggerVerbose.Info("Incoming body chunk", "EoS", v.RequestBody.EndOfStream)
			}
			responses, err = s.processRequestBody(ctx, req.GetRequestBody(), streamedBody, logger)
		case *extProcPb.ProcessingRequest_RequestTrailers:
			responses, err = s.HandleRequestTrailers(req.GetRequestTrailers())
		case *extProcPb.ProcessingRequest_ResponseHeaders:
			responses, err = s.HandleResponseHeaders(req.GetResponseHeaders())
		case *extProcPb.ProcessingRequest_ResponseBody:
			responses, err = s.HandleResponseBody(req.GetResponseBody())
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

func (s *Server) processRequestBody(ctx context.Context, body *extProcPb.HttpBody, streamedBody *streamedBody, logger logr.Logger) ([]*extProcPb.ProcessingResponse, error) {
	loggerVerbose := logger.V(logutil.VERBOSE)

	var requestBodyBytes []byte
	if s.streaming {
		streamedBody.body = append(streamedBody.body, body.Body...)
		// In the stream case, we can receive multiple request bodies.
		if body.EndOfStream {
			loggerVerbose.Info("Flushing stream buffer")
			requestBodyBytes = streamedBody.body
		} else {
			return nil, nil
		}
	} else {
		requestBodyBytes = body.GetBody()
	}

	return s.HandleRequestBody(ctx, requestBodyBytes)
}
