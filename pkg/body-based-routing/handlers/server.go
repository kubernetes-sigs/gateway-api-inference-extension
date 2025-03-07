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
	"io"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
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

	reader, writer := io.Pipe()

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
			// This error occurs very frequently, though it doesn't seem to have any impact.
			// TODO Figure out if we can remove this noise.
			loggerVerbose.Error(recvErr, "Cannot receive stream request")
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
				responses, err = s.HandleRequestHeaders(req.GetRequestHeaders())
			}
		case *extProcPb.ProcessingRequest_RequestBody:
			loggerVerbose.Info("Incoming body chunk", "body", string(v.RequestBody.Body), "EoS", v.RequestBody.EndOfStream)
			responses, err = s.processRequestBody(ctx, req.GetRequestBody(), writer, reader, logger)
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
			logger.V(logutil.DEFAULT).Error(err, "Failed to process request", "request", req)
			return status.Errorf(status.Code(err), "failed to handle request: %v", err)
		}

		for _, resp := range responses {
			loggerVerbose.Info("Response generated", "response", resp)
			if err := srv.Send(resp); err != nil {
				logger.V(logutil.DEFAULT).Error(err, "Send failed")
				return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
			}
		}
	}
}

func (s *Server) processRequestBody(ctx context.Context, body *extProcPb.HttpBody, bufferWriter *io.PipeWriter, bufferReader *io.PipeReader, logger logr.Logger) ([]*extProcPb.ProcessingResponse, error) {
	loggerVerbose := logger.V(logutil.VERBOSE)

	var requestBody map[string]interface{}
	if s.streaming {
		// In the stream case, we can receive multiple request bodies.
		// To buffer the full message, we create a goroutine with a writer.Write()
		// call, which will block until the corresponding reader reads from it.
		// We do not read until we receive the EndofStream signal, and then
		// decode the entire JSON body.
		if !body.EndOfStream {
			go func() {
				loggerVerbose.Info("Writing to stream buffer")
				_, err := bufferWriter.Write(body.Body)
				if err != nil {
					logger.V(logutil.DEFAULT).Error(err, "Error populating writer")
				}
			}()

			return nil, nil
		}

		if body.EndOfStream {
			loggerVerbose.Info("Flushing stream buffer")
			decoder := json.NewDecoder(bufferReader)
			if err := decoder.Decode(&requestBody); err != nil {
				logger.V(logutil.DEFAULT).Error(err, "Error unmarshaling request body")
			}
			bufferReader.Close()
		}
	} else {
		if err := json.Unmarshal(body.GetBody(), &requestBody); err != nil {
			return nil, err
		}
	}

	requestBodyResp, err := s.HandleRequestBody(ctx, requestBody)
	if err != nil {
		return nil, err
	}

	return requestBodyResp, nil
}
