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
	"io"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"sigs.k8s.io/controller-runtime/pkg/log"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

func NewServer(scheduler Scheduler, destinationEndpointHintMetadataNamespace, destinationEndpointHintKey string, datastore datastore.Datastore) *Server {
	return &Server{
		scheduler:                                scheduler,
		destinationEndpointHintMetadataNamespace: destinationEndpointHintMetadataNamespace,
		destinationEndpointHintKey:               destinationEndpointHintKey,
		datastore:                                datastore,
	}
}

// Server implements the Envoy external processing server.
// https://www.envoyproxy.io/docs/envoy/latest/api-v3/service/ext_proc/v3/external_processor.proto
type Server struct {
	scheduler Scheduler
	// The key of the header to specify the target pod address. This value needs to match Envoy
	// configuration.
	destinationEndpointHintKey string
	// The key acting as the outer namespace struct in the metadata extproc response to communicate
	// back the picked endpoints.
	destinationEndpointHintMetadataNamespace string
	datastore                                datastore.Datastore
}

type Scheduler interface {
	Schedule(ctx context.Context, b *scheduling.LLMRequest) (targetPod backendmetrics.PodMetrics, err error)
}

func (s *Server) Process(srv extProcPb.ExternalProcessor_ProcessServer) error {
	ctx := srv.Context()
	logger := log.FromContext(ctx)
	loggerVerbose := logger.V(logutil.VERBOSE)
	loggerVerbose.Info("Processing")

	// Create request context to share states during life time of an HTTP request.
	// See https://github.com/envoyproxy/envoy/issues/17540.
	reqCtx := &RequestContext{}

	// Create variable for error handling as each request should only report once for
	// error metric. This doesn't cover the error "Cannot receive stream request" because
	// such error might happen even the response is processed.
	var err error
	defer func(error) {
		if reqCtx.ResponseStatusCode != "" {
			metrics.RecordRequestErrCounter(reqCtx.Model, reqCtx.ResolvedTargetModel, reqCtx.ResponseStatusCode)
		} else if err != nil {
			metrics.RecordRequestErrCounter(reqCtx.Model, reqCtx.ResolvedTargetModel, errutil.CanonicalCode(err))
		}
	}(err)

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		req, recvErr := srv.Recv()
		if recvErr == io.EOF || status.Code(recvErr) == codes.Canceled {
			return nil
		}
		if recvErr != nil {
			// This error occurs very frequently, though it doesn't seem to have any impact.
			// TODO Figure out if we can remove this noise.
			loggerVerbose.Error(err, "Cannot receive stream request")
			return status.Errorf(codes.Unknown, "cannot receive stream request: %v", err)
		}

		var resp *extProcPb.ProcessingResponse
		switch v := req.Request.(type) {
		case *extProcPb.ProcessingRequest_RequestHeaders:
			reqCtx.RequestReceivedTimestamp = time.Now()
			resp = HandleRequestHeaders(ctx, reqCtx, req)
			loggerVerbose.Info("Request context after HandleRequestHeaders", "context", reqCtx)
		case *extProcPb.ProcessingRequest_RequestBody:
			resp, err = s.HandleRequestBody(ctx, reqCtx, req)
			if err == nil {
				metrics.RecordRequestCounter(reqCtx.Model, reqCtx.ResolvedTargetModel)
				metrics.RecordRequestSizes(reqCtx.Model, reqCtx.ResolvedTargetModel, reqCtx.RequestSize)
			}
			loggerVerbose.Info("Request context after HandleRequestBody", "context", reqCtx)
		case *extProcPb.ProcessingRequest_ResponseHeaders:
			resp, err = s.HandleResponseHeaders(ctx, reqCtx, req)
			loggerVerbose.Info("Request context after HandleResponseHeaders", "context", reqCtx)
		case *extProcPb.ProcessingRequest_ResponseBody:
			resp, err = s.HandleResponseBody(ctx, reqCtx, req)
			if err == nil && reqCtx.ResponseComplete {
				reqCtx.ResponseCompleteTimestamp = time.Now()
				metrics.RecordRequestLatencies(ctx, reqCtx.Model, reqCtx.ResolvedTargetModel, reqCtx.RequestReceivedTimestamp, reqCtx.ResponseCompleteTimestamp)
				metrics.RecordResponseSizes(reqCtx.Model, reqCtx.ResolvedTargetModel, reqCtx.ResponseSize)
				metrics.RecordInputTokens(reqCtx.Model, reqCtx.ResolvedTargetModel, reqCtx.Response.Usage.PromptTokens)
				metrics.RecordOutputTokens(reqCtx.Model, reqCtx.ResolvedTargetModel, reqCtx.Response.Usage.CompletionTokens)
			}
			loggerVerbose.Info("Request context after HandleResponseBody", "context", reqCtx)
		default:
			logger.V(logutil.DEFAULT).Error(nil, "Unknown Request type", "request", v)
			return status.Error(codes.Unknown, "unknown request type")
		}

		if err != nil {
			logger.V(logutil.DEFAULT).Error(err, "Failed to process request", "request", req)
			resp, err = BuildErrResponse(err)
			if err != nil {
				return err
			}
		}

		loggerVerbose.Info("Response generated", "response", resp)
		if err := srv.Send(resp); err != nil {
			logger.V(logutil.DEFAULT).Error(err, "Send failed")
			return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
		}
	}
}

func BuildErrResponse(err error) (*extProcPb.ProcessingResponse, error) {
	var resp *extProcPb.ProcessingResponse

	switch errutil.CanonicalCode(err) {
	// This code can be returned by scheduler when there is no capacity for sheddable
	// requests.
	case errutil.InferencePoolResourceExhausted:
		resp = &extProcPb.ProcessingResponse{
			Response: &extProcPb.ProcessingResponse_ImmediateResponse{
				ImmediateResponse: &extProcPb.ImmediateResponse{
					Status: &envoyTypePb.HttpStatus{
						Code: envoyTypePb.StatusCode_TooManyRequests,
					},
				},
			},
		}
	// This code can be returned by when EPP processes the request and run into server-side errors.
	case errutil.Internal:
		resp = &extProcPb.ProcessingResponse{
			Response: &extProcPb.ProcessingResponse_ImmediateResponse{
				ImmediateResponse: &extProcPb.ImmediateResponse{
					Status: &envoyTypePb.HttpStatus{
						Code: envoyTypePb.StatusCode_InternalServerError,
					},
				},
			},
		}
	// This code can be returned when users provide invalid json request.
	case errutil.BadRequest:
		resp = &extProcPb.ProcessingResponse{
			Response: &extProcPb.ProcessingResponse_ImmediateResponse{
				ImmediateResponse: &extProcPb.ImmediateResponse{
					Status: &envoyTypePb.HttpStatus{
						Code: envoyTypePb.StatusCode_BadRequest,
					},
				},
			},
		}
	case errutil.BadConfiguration:
		resp = &extProcPb.ProcessingResponse{
			Response: &extProcPb.ProcessingResponse_ImmediateResponse{
				ImmediateResponse: &extProcPb.ImmediateResponse{
					Status: &envoyTypePb.HttpStatus{
						Code: envoyTypePb.StatusCode_NotFound,
					},
				},
			},
		}
	default:
		return nil, status.Errorf(status.Code(err), "failed to handle request: %v", err)
	}
	return resp, nil
}

// RequestContext stores context information during the life time of an HTTP request.
type RequestContext struct {
	TargetPod                 string
	TargetEndpoint            string
	Model                     string
	ResolvedTargetModel       string
	RequestReceivedTimestamp  time.Time
	ResponseCompleteTimestamp time.Time
	RequestSize               int
	Response                  Response
	ResponseSize              int
	ResponseComplete          bool
	ResponseStatusCode        string
}
