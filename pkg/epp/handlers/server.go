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
	"strings"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/go-logr/logr"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/structpb"
	"sigs.k8s.io/controller-runtime/pkg/log"

	envoyhandlers "sigs.k8s.io/gateway-api-inference-extension/pkg/common/envoy/handlers"
	reqenvoy "sigs.k8s.io/gateway-api-inference-extension/pkg/common/envoy/request"
	errcommon "sigs.k8s.io/gateway-api-inference-extension/pkg/common/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	reqcommon "sigs.k8s.io/gateway-api-inference-extension/pkg/common/request"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkrq "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	fwkrh "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/version"
)

func NewServer(datastore Datastore, director Director, parser fwkrh.Parser) *Server {
	return &Server{
		director:  director,
		datastore: datastore,
		parser:    parser,
	}
}

type Director interface {
	HandleRequest(ctx context.Context, reqCtx *RequestContext) (*RequestContext, error)
	HandleResponseReceived(ctx context.Context, reqCtx *RequestContext) (*RequestContext, error)
	HandleResponseBodyStreaming(ctx context.Context, reqCtx *RequestContext) (*RequestContext, error)
	HandleResponseBodyComplete(ctx context.Context, reqCtx *RequestContext) (*RequestContext, error)
	GetRandomEndpoint() *fwkdl.EndpointMetadata
}

type Datastore interface {
	PoolGet() (*datalayer.EndpointPool, error)
}

// Server implements the Envoy external processing server.
// https://www.envoyproxy.io/docs/envoy/latest/api-v3/service/ext_proc/v3/external_processor.proto
type Server struct {
	datastore Datastore
	director  Director
	parser    fwkrh.Parser
}

// RequestContext stores context information during the life time of an HTTP request.
//
// TODO(https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/2082):
// Refactor this monolithic struct. Fields related to the Envoy ext-proc protocol should be decoupled from the internal
// request lifecycle state.
type RequestContext struct {
	TargetPod                 *fwkdl.EndpointMetadata
	TargetEndpoint            string
	IncomingModelName         string
	TargetModelName           string
	FairnessID                string
	ObjectiveKey              string
	RequestReceivedTimestamp  time.Time
	ResponseCompleteTimestamp time.Time
	RequestSize               int
	Usage                     fwkrq.Usage
	ResponseSize              int
	ResponseComplete          bool
	ResponseStatusCode        string
	RequestRunning            bool
	Request                   *Request

	SchedulingRequest *schedulingtypes.LLMRequest

	RequestState         StreamRequestState
	modelServerStreaming bool

	Response *Response

	reqHeaderResp  *extProcPb.ProcessingResponse
	reqBodyResp    []*extProcPb.ProcessingResponse
	reqTrailerResp *extProcPb.ProcessingResponse

	respHeaderResp  *extProcPb.ProcessingResponse
	respBodyResp    []*extProcPb.ProcessingResponse
	respTrailerResp *extProcPb.ProcessingResponse
}

type Request struct {
	Headers  map[string]string
	RawBody  []byte // This field will be updated when request body is modified (e.g. model mutation in requestBody)
	Metadata map[string]any
}
type Response struct {
	Headers         map[string]string
	DynamicMetadata *structpb.Struct
}
type StreamRequestState int

const (
	RequestReceived                  StreamRequestState = 0
	HeaderRequestResponseComplete    StreamRequestState = 1
	BodyRequestResponsesComplete     StreamRequestState = 2
	TrailerRequestResponsesComplete  StreamRequestState = 3
	ResponseReceived                 StreamRequestState = 4
	HeaderResponseResponseComplete   StreamRequestState = 5
	BodyResponseResponsesComplete    StreamRequestState = 6
	TrailerResponseResponsesComplete StreamRequestState = 7
)

func (s *Server) Process(srv extProcPb.ExternalProcessor_ProcessServer) error {
	ctx := srv.Context()

	// Start tracing span for the request
	tracer := otel.Tracer(
		"gateway-api-inference-extension/epp/extproc",
		trace.WithInstrumentationVersion(version.BuildRef),
		trace.WithInstrumentationAttributes(
			attribute.String("commit-sha", version.CommitSHA),
		),
	)
	ctx, span := tracer.Start(ctx, "gateway.request", trace.WithSpanKind(trace.SpanKindServer))
	defer span.End()

	logger := log.FromContext(ctx)
	loggerTrace := logger.V(logutil.TRACE)
	loggerTrace.Info("Processing")

	// Create request context to share states during life time of an HTTP request.
	// See https://github.com/envoyproxy/envoy/issues/17540.
	reqCtx := &RequestContext{
		RequestState: RequestReceived,
		Request: &Request{
			Headers:  make(map[string]string),
			Metadata: make(map[string]any),
		},
		Response: &Response{
			Headers: make(map[string]string),
		},
	}

	var body []byte

	// Create error handling var as each request should only report once for
	// error metrics. This doesn't cover the error "Cannot receive stream request" because
	// such errors might happen even though response is processed.
	var err error
	defer func(error, *RequestContext) {
		if reqCtx.ResponseStatusCode != "" {
			metrics.RecordRequestErrCounter(reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.ResponseStatusCode)
		} else if err != nil {
			metrics.RecordRequestErrCounter(reqCtx.IncomingModelName, reqCtx.TargetModelName, errcommon.CanonicalCode(err))
		}
		if reqCtx.RequestRunning {
			metrics.DecRunningRequests(reqCtx.IncomingModelName)
		}

		// If we scheduled a pod (TargetPod != nil) but never marked the response  as complete (e.g. error, disconnect,
		// panic), force the completion hooks to run.
		if reqCtx.TargetPod != nil && !reqCtx.ResponseComplete {
			// Use a fresh context as the request context might be canceled (Client Disconnect).
			// We only need logging from the original context.
			cleanupCtx := log.IntoContext(context.Background(), logger)
			if _, err := s.director.HandleResponseBodyComplete(cleanupCtx, reqCtx); err != nil {
				logger.Error(err, "error in HandleResponseBodyComplete")
			}
		}
	}(err, reqCtx)

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
			return status.Errorf(codes.Unknown, "cannot receive stream request: %v", err)
		}

		reqCtx.Request.Metadata = reqenvoy.ExtractMetadataValues(req)

		switch v := req.Request.(type) {
		case *extProcPb.ProcessingRequest_RequestHeaders:
			requestID := reqenvoy.ExtractHeaderValue(v, reqcommon.RequestIdHeaderKey)
			// request ID is a must for maintaining a state per request in plugins that hold internal state and use PluginState.
			// if request id was not supplied as a header, we generate it ourselves.
			if len(requestID) == 0 {
				requestID = uuid.NewString()
				loggerTrace.Info("RequestID header is not found in the request, generated a request id")
				reqCtx.Request.Headers[reqcommon.RequestIdHeaderKey] = requestID // update in headers so director can consume it
			}
			logger = logger.WithValues(reqcommon.RequestIdHeaderKey, requestID)
			logger.V(1).Info("EPP received request") // Request ID will be logged too as part of logger context values.
			loggerTrace = logger.V(logutil.TRACE)
			ctx = log.IntoContext(ctx, logger)

			err = s.HandleRequestHeaders(ctx, reqCtx, v)
		case *extProcPb.ProcessingRequest_RequestBody:
			loggerTrace.Info("Incoming body chunk", "EoS", v.RequestBody.EndOfStream)
			// In the stream case, we can receive multiple request bodies.
			body = append(body, v.RequestBody.Body...)

			// Message is buffered, we can read and decode.
			if v.RequestBody.EndOfStream {
				loggerTrace.Info("decoding")
				reqCtx.Request.RawBody = body

				// Body stream complete. Capture raw size for flow control.
				reqCtx.RequestSize = len(body)
				body = []byte{}

				reqCtx, err = s.director.HandleRequest(ctx, reqCtx)
				if err != nil {
					logger.V(1).Error(err, "Error handling request")
					break
				}

				reqCtx.reqHeaderResp = s.generateRequestHeaderResponse(ctx, reqCtx)
				reqCtx.reqBodyResp = envoyhandlers.GenerateRequestBodyResponses(reqCtx.Request.RawBody)

				metrics.RecordRequestCounter(reqCtx.IncomingModelName, reqCtx.TargetModelName)
				metrics.RecordRequestSizes(reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.RequestSize)
			}
		case *extProcPb.ProcessingRequest_RequestTrailers:
			// This is currently unused.
		case *extProcPb.ProcessingRequest_ResponseHeaders:
			for _, header := range v.ResponseHeaders.Headers.GetHeaders() {
				value := string(header.RawValue)
				loggerTrace.Info("header", "key", header.Key, "value", value)
				if header.Key == "status" && value != "200" {
					reqCtx.ResponseStatusCode = errcommon.ModelServerError
				} else if header.Key == "content-type" && strings.Contains(value, "text/event-stream") {
					reqCtx.modelServerStreaming = true
					loggerTrace.Info("model server is streaming response")
				}
			}
			reqCtx.RequestState = ResponseReceived

			var responseErr error
			reqCtx, responseErr = s.HandleResponseHeaders(ctx, reqCtx, v)
			if responseErr != nil {
				if logger.V(logutil.DEBUG).Enabled() {
					logger.V(logutil.DEBUG).Error(responseErr, "Failed to process response headers", "request", req)
				} else {
					logger.V(1).Error(responseErr, "Failed to process response headers")
				}
			}
			reqCtx.respHeaderResp = s.generateResponseHeaderResponse(reqCtx)

		case *extProcPb.ProcessingRequest_ResponseBody:
			endOfStream := v.ResponseBody.EndOfStream
			chunk := v.ResponseBody.Body

			if reqCtx.modelServerStreaming {
				s.HandleResponseBodyModelStreaming(ctx, reqCtx, chunk, endOfStream)
				reqCtx.respBodyResp = envoyhandlers.GenerateResponseBodyResponses(chunk, endOfStream)
			} else {
				body = append(body, chunk...)
			}

			if endOfStream {
				err = s.finishResponse(ctx, reqCtx, body)
			}
		case *extProcPb.ProcessingRequest_ResponseTrailers:
			// For HTTP, the response trailer is not sent. Thus, this case will not be triggered.
			// For gRPC(over HTTP2), the protocol relies on responseTrialers to determine whether a response is complete.
			// More info: https://chromium.googlesource.com/external/github.com/grpc/grpc/+/HEAD/doc/PROTOCOL-HTTP2.md#responses
			err = s.finishResponse(ctx, reqCtx, body)
			if err == nil {
				reqCtx.respTrailerResp = &extProcPb.ProcessingResponse{
					Response: &extProcPb.ProcessingResponse_ResponseTrailers{
						ResponseTrailers: &extProcPb.TrailersResponse{},
					},
				}
			}
		}

		// Handle the err and fire an immediate response.
		if err != nil {
			if logger.V(logutil.DEBUG).Enabled() {
				logger.V(logutil.DEBUG).Error(err, "Failed to process request", "request", req)
			} else {
				logger.V(1).Error(err, "Failed to process request")
			}
			resp, err := buildErrResponse(err)
			if err != nil {
				return err
			}
			if err := srv.Send(resp); err != nil {
				logger.V(1).Error(err, "Send failed")
				return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
			}
			return nil
		}
		loggerTrace.Info("checking", "request state", reqCtx.RequestState)
		if err := reqCtx.updateStateAndSendIfNeeded(srv, logger); err != nil {
			return err
		}
	}
}

// finishResponse ensures all post-response logic, such as metric recording
// and state updates, is executed exactly once for the request lifecycle.
func (s *Server) finishResponse(ctx context.Context, reqCtx *RequestContext, body []byte) error {
	// Return early if the response has already been finished to prevent
	// duplicate execution of side effects and metrics.
	if reqCtx.ResponseComplete {
		return nil
	}

	reqCtx.ResponseComplete = true
	reqCtx.ResponseCompleteTimestamp = time.Now()
	reqCtx.ResponseSize = len(body)

	if reqCtx.modelServerStreaming {
		if _, err := s.director.HandleResponseBodyComplete(ctx, reqCtx); err != nil {
			log.FromContext(ctx).Error(err, "error in HandleResponseBodyComplete")
		}
		metrics.RecordRequestLatencies(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.RequestReceivedTimestamp, reqCtx.ResponseCompleteTimestamp)
		metrics.RecordResponseSizes(reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.ResponseSize)
		metrics.RecordNormalizedTimePerOutputToken(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.RequestReceivedTimestamp, reqCtx.ResponseCompleteTimestamp, reqCtx.Usage.CompletionTokens)
	} else {
		reqCtx.respBodyResp = envoyhandlers.GenerateResponseBodyResponses(body, true)
		if _, err := s.HandleResponseBody(ctx, reqCtx, body); err != nil {
			return err
		}
		metrics.RecordRequestLatencies(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.RequestReceivedTimestamp, reqCtx.ResponseCompleteTimestamp)
		metrics.RecordResponseSizes(reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.ResponseSize)
		metrics.RecordInputTokens(reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.Usage.PromptTokens)
		metrics.RecordOutputTokens(reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.Usage.CompletionTokens)
		if reqCtx.Usage.PromptTokenDetails != nil {
			metrics.RecordPromptCachedTokens(reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.Usage.PromptTokenDetails.CachedTokens)
		}
	}
	return nil
}

// updateStateAndSendIfNeeded checks state and can send mutiple responses in a single pass, but only if ordered properly.
// Order of requests matter in FULL_DUPLEX_STREAMING. For both request and response, the order of response sent back MUST be: Header->Body->Trailer, with trailer being optional.
func (r *RequestContext) updateStateAndSendIfNeeded(srv extProcPb.ExternalProcessor_ProcessServer, logger logr.Logger) error {
	loggerTrace := logger.V(logutil.TRACE)
	// No switch statement as we could send multiple responses in one pass.
	if r.RequestState == RequestReceived && r.reqHeaderResp != nil {
		loggerTrace.Info("Sending request header response", "obj", r.reqHeaderResp)
		if err := srv.Send(r.reqHeaderResp); err != nil {
			logger.V(1).Error(err, "error sending response")
			return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
		}
		r.RequestState = HeaderRequestResponseComplete
	}
	if r.RequestState == HeaderRequestResponseComplete && r.reqBodyResp != nil && len(r.reqBodyResp) > 0 {
		loggerTrace.Info("Sending request body response(s)")

		for _, response := range r.reqBodyResp {
			if err := srv.Send(response); err != nil {
				return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
			}
		}
		logger.V(1).Info("EPP sent request body response(s) to proxy", "modelName", r.IncomingModelName, "targetModelName", r.TargetModelName)
		r.RequestState = BodyRequestResponsesComplete
		metrics.IncRunningRequests(r.IncomingModelName)
		r.RequestRunning = true
		// Dump the response so a new stream message can begin
		r.reqBodyResp = nil
	}
	if r.RequestState == BodyRequestResponsesComplete && r.reqTrailerResp != nil {
		// Trailers in requests are not guaranteed
		if err := srv.Send(r.reqTrailerResp); err != nil {
			return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
		}
	}
	if r.RequestState == ResponseReceived && r.respHeaderResp != nil {
		loggerTrace.Info("Sending response header response", "obj", r.respHeaderResp)
		if err := srv.Send(r.respHeaderResp); err != nil {
			return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
		}
		r.RequestState = HeaderResponseResponseComplete
	}
	if r.RequestState == HeaderResponseResponseComplete {
		loggerTrace.Info("Sending response body response(s)")
		for _, response := range r.respBodyResp {
			if err := srv.Send(response); err != nil {
				return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
			}
		}
		if r.ResponseComplete {
			logger.V(1).Info("EPP sent response body back to proxy")
			r.RequestState = BodyResponseResponsesComplete
		}
		// Dump the response so a new stream message can begin
		r.respBodyResp = nil
	}
	if r.RequestState == BodyResponseResponsesComplete && r.respTrailerResp != nil {
		// Trailers in requests are not guaranteed
		if err := srv.Send(r.respTrailerResp); err != nil {
			return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
		}
	}
	return nil
}

func buildErrResponse(err error) (*extProcPb.ProcessingResponse, error) {
	var resp *extProcPb.ProcessingResponse

	switch errcommon.CanonicalCode(err) {
	// This code can be returned by scheduler when there is no capacity for sheddable
	// requests.
	case errcommon.ResourceExhausted:
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
	case errcommon.Internal:
		resp = &extProcPb.ProcessingResponse{
			Response: &extProcPb.ProcessingResponse_ImmediateResponse{
				ImmediateResponse: &extProcPb.ImmediateResponse{
					Status: &envoyTypePb.HttpStatus{
						Code: envoyTypePb.StatusCode_InternalServerError,
					},
				},
			},
		}
	// This code can be returned by the director when there are no candidate pods for the request scheduling.
	case errcommon.ServiceUnavailable:
		resp = &extProcPb.ProcessingResponse{
			Response: &extProcPb.ProcessingResponse_ImmediateResponse{
				ImmediateResponse: &extProcPb.ImmediateResponse{
					Status: &envoyTypePb.HttpStatus{
						Code: envoyTypePb.StatusCode_ServiceUnavailable,
					},
				},
			},
		}
	// This code can be returned when users provide invalid json request.
	case errcommon.BadRequest:
		resp = &extProcPb.ProcessingResponse{
			Response: &extProcPb.ProcessingResponse_ImmediateResponse{
				ImmediateResponse: &extProcPb.ImmediateResponse{
					Status: &envoyTypePb.HttpStatus{
						Code: envoyTypePb.StatusCode_BadRequest,
					},
				},
			},
		}
	default:
		return nil, status.Errorf(status.Code(err), "failed to handle request: %v", err)
	}

	if err.Error() != "" {
		resp.Response.(*extProcPb.ProcessingResponse_ImmediateResponse).ImmediateResponse.Body = []byte(err.Error())
	}

	return resp, nil
}
