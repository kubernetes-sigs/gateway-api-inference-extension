package handlers

import (
	"context"
	"errors"
	"io"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/scheduling"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/logging"
)

func NewServer(scheduler Scheduler, targetEndpointKey string, datastore datastore.Datastore) *Server {
	return &Server{
		scheduler:         scheduler,
		targetEndpointKey: targetEndpointKey,
		datastore:         datastore,
	}
}

// Server implements the Envoy external processing server.
// https://www.envoyproxy.io/docs/envoy/latest/api-v3/service/ext_proc/v3/external_processor.proto
type Server struct {
	scheduler Scheduler
	// The key of the header to specify the target pod address. This value needs to match Envoy
	// configuration.
	targetEndpointKey string
	datastore         datastore.Datastore
}

type Scheduler interface {
	Schedule(ctx context.Context, b *scheduling.LLMRequest) (targetPod datastore.PodMetrics, err error)
}

func (s *Server) Process(srv extProcPb.ExternalProcessor_ProcessServer) error {
	ctx := srv.Context()
	logger := log.FromContext(ctx)
	loggerVerbose := logger.V(logutil.VERBOSE)
	loggerVerbose.Info("Processing")

	// Create request context to share states during life time of an HTTP request.
	// See https://github.com/envoyproxy/envoy/issues/17540.
	reqCtx := &RequestContext{}

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		req, err := srv.Recv()
		if err == io.EOF || errors.Is(err, context.Canceled) {
			return nil
		}
		if err != nil {
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
			switch status.Code(err) {
			// This code can be returned by scheduler when there is no capacity for sheddable
			// requests.
			case codes.ResourceExhausted:
				resp = &extProcPb.ProcessingResponse{
					Response: &extProcPb.ProcessingResponse_ImmediateResponse{
						ImmediateResponse: &extProcPb.ImmediateResponse{
							Status: &envoyTypePb.HttpStatus{
								Code: envoyTypePb.StatusCode_TooManyRequests,
							},
						},
					},
				}
			default:
				return status.Errorf(status.Code(err), "failed to handle request: %v", err)
			}
		}

		loggerVerbose.Info("Response generated", "response", resp)
		if err := srv.Send(resp); err != nil {
			logger.V(logutil.DEFAULT).Error(err, "Send failed")
			return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
		}
	}
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
}
