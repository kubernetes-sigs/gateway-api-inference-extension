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

	envoy "sigs.k8s.io/gateway-api-inference-extension/pkg/common/envoy"
	errcommon "sigs.k8s.io/gateway-api-inference-extension/pkg/common/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	reqcommon "sigs.k8s.io/gateway-api-inference-extension/pkg/common/request"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkrh "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/version"
)

// EvictChannelLookup is an optional interface for looking up eviction channels by request ID.
// When set on the StreamingServer, the Process() loop will select on the eviction channel
// to support eviction of in-flight requests via ext_proc ImmediateResponse.
type EvictChannelLookup interface {
	Get(requestID string) chan struct{}
	Deregister(requestID string)
}

func NewStreamingServer(datastore Datastore, director Director, parser fwkrh.Parser) *StreamingServer {
	return &StreamingServer{
		director:  director,
		datastore: datastore,
		parser:    parser,
	}
}

// SetEvictChannelLookup sets the eviction channel lookup for eviction support.
func (s *StreamingServer) SetEvictChannelLookup(lookup EvictChannelLookup) {
	s.evictionLookup = lookup
}

type Director interface {
	HandleRequest(ctx context.Context, reqCtx *RequestContext, inferenceRequestBody *fwkrh.InferenceRequestBody) (*RequestContext, error)
	HandleResponseHeader(ctx context.Context, reqCtx *RequestContext) *RequestContext
	HandleResponseBody(ctx context.Context, reqCtx *RequestContext, endOfStream bool) *RequestContext
	GetRandomEndpoint() *fwkdl.EndpointMetadata
}

type Datastore interface {
	PoolGet() (*datalayer.EndpointPool, error)
}

// Server implements the Envoy external processing server.
// https://www.envoyproxy.io/docs/envoy/latest/api-v3/service/ext_proc/v3/external_processor.proto
type StreamingServer struct {
	datastore      Datastore
	director       Director
	parser         fwkrh.Parser
	evictionLookup EvictChannelLookup // optional, set for eviction support
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
	Priority                  int
	RequestReceivedTimestamp  time.Time
	ResponseCompleteTimestamp time.Time
	RequestSize               int
	Usage                     fwkrh.Usage
	ResponseSize              int
	ResponseBodyStarted       bool
	ResponseComplete          bool
	ResponseStatusCode        string
	RequestRunning            bool
	Request                   *Request

	SchedulingRequest *schedulingtypes.InferenceRequest

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
	// RequestEvicted indicates the request was evicted by flow control.
	// The state machine sends an ImmediateResponse(429) to Envoy.
	RequestEvicted StreamRequestState = 8
)

// recvResult holds the result of a srv.Recv() call from the reader goroutine.
type recvResult struct {
	req *extProcPb.ProcessingRequest
	err error
}

func (s *StreamingServer) Process(srv extProcPb.ExternalProcessor_ProcessServer) error {
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
	var evictionRequestID string

	// Start a single reader goroutine for the lifetime of the stream.
	// This avoids spawning a new goroutine per message and allows the main loop to
	// select on both incoming messages and the eviction channel.
	recvCh := make(chan recvResult, 1)
	// Capture the stream context's Done channel before ctx is reassigned in the main loop.
	// This avoids a data race between the reader goroutine reading ctx and the main loop writing it.
	streamDone := srv.Context().Done()
	go func() {
		for {
			req, err := srv.Recv()
			select {
			case recvCh <- recvResult{req: req, err: err}:
			case <-streamDone:
				return
			}
			if err != nil {
				return
			}
		}
	}()

	// evictCh starts nil — selecting on a nil channel blocks forever.
	// After scheduling, it is set to the eviction channel, dynamically
	// enabling eviction listening.
	var evictCh chan struct{}

	// Create error handling var as each request should only report once for
	// error metrics. This doesn't cover the error "Cannot receive stream request" because
	// such errors might happen even though response is processed.
	var err error
	defer func() {
		// Clean up eviction channel registration on exit.
		if s.evictionLookup != nil && evictionRequestID != "" {
			s.evictionLookup.Deregister(evictionRequestID)
		}
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
			s.director.HandleResponseBody(cleanupCtx, reqCtx, true)
		}
	}()

	for {
		var req *extProcPb.ProcessingRequest
		var recvErr error

		// Main select: listen for incoming messages, eviction signals, and context cancellation.
		// evictCh is nil until scheduling completes, so the eviction case blocks forever until then.
		select {
		case result := <-recvCh:
			req = result.req
			recvErr = result.err
		case <-evictCh:
			// Skip if the response already completed — sending ImmediateResponse
			// after the final body chunk would be a protocol violation.
			if reqCtx.ResponseComplete {
				logger.V(logutil.DEBUG).Info("Eviction signal received but response already complete, ignoring",
					"requestID", evictionRequestID)
				evictCh = nil // prevent closed channel from firing repeatedly
				continue
			}
			// Eviction triggered — transition to evicted state and let the state machine send the response.
			logger.Info("Request evicted by flow control", "requestID", evictionRequestID)
			reqCtx.RequestState = RequestEvicted
			if sendErr := reqCtx.updateStateAndSendIfNeeded(srv, logger); sendErr != nil {
				return sendErr
			}
			return nil
		case <-ctx.Done():
			return ctx.Err()
		}

		if recvErr == io.EOF || status.Code(recvErr) == codes.Canceled {
			return nil
		}
		if recvErr != nil {
			return status.Errorf(codes.Unknown, "cannot receive stream request: %v", recvErr)
		}

		reqCtx.Request.Metadata = envoy.ExtractMetadataValues(req)

		switch v := req.Request.(type) {
		case *extProcPb.ProcessingRequest_RequestHeaders:
			requestID := envoy.ExtractHeaderValue(v, reqcommon.RequestIdHeaderKey)
			// request ID is a must for maintaining a state per request in plugins that hold internal state and use PluginState.
			// if request id was not supplied as a header, we generate it ourselves.
			if len(requestID) == 0 {
				requestID = uuid.NewString()
				loggerTrace.Info("RequestID header is not found in the request, generated a request id")
				reqCtx.Request.Headers[reqcommon.RequestIdHeaderKey] = requestID // update in headers so director can consume it
			}
			logger = logger.WithValues(reqcommon.RequestIdHeaderKey, requestID)
			logger.V(logutil.DEFAULT).Info("EPP received request") // Request ID will be logged too as part of logger context values.
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

				inferenceRequestBody, parseErr := s.parser.ParseRequest(ctx, reqCtx.Request.RawBody, reqCtx.Request.Headers)
				if parseErr != nil {
					err = errcommon.Error{Code: errcommon.BadRequest, Msg: parseErr.Error()}
					logger.Error(err, "Error parsing request")
					break
				}

				reqCtx, err = s.director.HandleRequest(ctx, reqCtx, inferenceRequestBody)
				if err != nil {
					logger.Error(err, "Error handling request")
					break
				}

				// After scheduling, look up the eviction channel for eviction support.
				// Setting evictCh from nil to a real channel dynamically enables the
				// eviction case in the main select.
				if s.evictionLookup != nil {
					evictionRequestID = reqCtx.Request.Headers[reqcommon.RequestIdHeaderKey]
					evictCh = s.evictionLookup.Get(evictionRequestID)
				}

				if reqCtx.SchedulingRequest != nil && reqCtx.SchedulingRequest.Body != nil {
					reqCtx.modelServerStreaming = reqCtx.SchedulingRequest.Body.Stream
				}

				reqCtx.reqHeaderResp = s.generateRequestHeaderResponse(ctx, reqCtx)
				reqCtx.reqBodyResp = envoy.GenerateRequestBodyResponses(reqCtx.Request.RawBody)
				metrics.RecordRequestCounter(reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.Priority)
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
			reqCtx = s.HandleResponseHeaders(ctx, reqCtx, v)
			reqCtx.respHeaderResp = s.generateResponseHeaderResponse(reqCtx)

		case *extProcPb.ProcessingRequest_ResponseBody:
			endOfStream := v.ResponseBody.EndOfStream
			chunk := v.ResponseBody.Body

			if reqCtx.modelServerStreaming {
				if endOfStream {
					reqCtx.ResponseComplete = true
					reqCtx.ResponseCompleteTimestamp = time.Now()
				}
				s.HandleResponseBody(ctx, reqCtx, chunk, endOfStream)
				// Rewrite the model name in response body back to the original client-facing name.
				chunk = rewriteModelName(chunk, reqCtx.TargetModelName, reqCtx.IncomingModelName)
				// For streaming response, we send response chunk back to envoy every time we received it.
				reqCtx.respBodyResp = generateResponseBodyResponses(chunk, endOfStream, reqCtx.Response.DynamicMetadata)
			} else {
				body = append(body, chunk...)
				if endOfStream {
					s.finishResponse(ctx, reqCtx, body, reqCtx.modelServerStreaming, true)
				}
			}
		case *extProcPb.ProcessingRequest_ResponseTrailers:
			// For HTTP, the response trailer is not sent. Thus, this case will not be triggered.
			// For gRPC(over HTTP2), the protocol relies on responseTrialers to determine whether a response is complete.
			// More info: https://chromium.googlesource.com/external/github.com/grpc/grpc/+/HEAD/doc/PROTOCOL-HTTP2.md#responses
			s.finishResponse(ctx, reqCtx, body, reqCtx.modelServerStreaming, false)
			reqCtx.respTrailerResp = &extProcPb.ProcessingResponse{
				Response: &extProcPb.ProcessingResponse_ResponseTrailers{
					ResponseTrailers: &extProcPb.TrailersResponse{},
				},
			}
		}

		// Handle the err and fire an immediate response.
		if err != nil {
			if logger.V(logutil.DEBUG).Enabled() {
				logger.V(logutil.DEBUG).Error(err, "Failed to process request", "request", req)
			} else {
				logger.Error(err, "Failed to process request")
			}
			resp, err := errcommon.BuildErrResponse(err)
			if err != nil {
				return err
			}
			if err := srv.Send(resp); err != nil {
				logger.Error(err, "Send failed")
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
func (s *StreamingServer) finishResponse(ctx context.Context, reqCtx *RequestContext, body []byte, modelStreaming bool, setEos bool) {
	// Return early if the response has already been finished to prevent
	// duplicate execution of side effects and metrics.
	if reqCtx.ResponseComplete {
		return
	}

	reqCtx.ResponseComplete = true
	reqCtx.ResponseCompleteTimestamp = time.Now()
	reqCtx = s.HandleResponseBody(ctx, reqCtx, body, true)
	if !modelStreaming {
		// Rewrite the model name in response body back to the original client-facing name.
		body = rewriteModelName(body, reqCtx.TargetModelName, reqCtx.IncomingModelName)
		// For non-streaming response, we send response back to envoy after receiving all the response body.
		reqCtx.respBodyResp = generateResponseBodyResponses(body, setEos, reqCtx.Response.DynamicMetadata)
	}
}

// rewriteModelName replaces occurrences of the target (internal) model name with the
// incoming (client-facing) model name in the response body bytes. This ensures clients
// see the model name they originally requested, not the internal backend model name.
// It is a no-op when the names are identical or either is empty.
func rewriteModelName(body []byte, targetModel, incomingModel string) []byte {
	if targetModel == "" || incomingModel == "" || targetModel == incomingModel {
		return body
	}
	old := []byte(`"model":"` + targetModel + `"`)
	new := []byte(`"model":"` + incomingModel + `"`)
	result := bytes.ReplaceAll(body, old, new)
	if !bytes.Equal(result, body) {
		return result
	}
	// Also handle the case where JSON has spaces after the colon: "model": "..."
	old = []byte(`"model": "` + targetModel + `"`)
	new = []byte(`"model": "` + incomingModel + `"`)
	return bytes.ReplaceAll(body, old, new)
}

// updateStateAndSendIfNeeded checks state and can send multiple responses in a single pass, but only if ordered properly.
// Order of requests matter in FULL_DUPLEX_STREAMING. For both request and response, the order of response sent back MUST be: Header->Body->Trailer, with trailer being optional.
func (r *RequestContext) updateStateAndSendIfNeeded(srv extProcPb.ExternalProcessor_ProcessServer, logger logr.Logger) error {
	loggerTrace := logger.V(logutil.TRACE)

	// Handle eviction — send ImmediateResponse(429) to Envoy to reset the upstream connection.
	if r.RequestState == RequestEvicted {
		loggerTrace.Info("Sending ImmediateResponse for evicted request")
		return srv.Send(&extProcPb.ProcessingResponse{
			Response: &extProcPb.ProcessingResponse_ImmediateResponse{
				ImmediateResponse: &extProcPb.ImmediateResponse{
					Status: &envoyTypePb.HttpStatus{
						Code: envoyTypePb.StatusCode_TooManyRequests,
					},
					Body: []byte("request evicted by flow control"),
				},
			},
		})
	}

	// No switch statement as we could send multiple responses in one pass.
	if r.RequestState == RequestReceived && r.reqHeaderResp != nil {
		loggerTrace.Info("Sending request header response", "obj", r.reqHeaderResp)
		if err := srv.Send(r.reqHeaderResp); err != nil {
			logger.Error(err, "error sending response")
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
		logger.V(logutil.DEFAULT).Info("EPP sent request body response(s) to proxy", "modelName", r.IncomingModelName, "targetModelName", r.TargetModelName)
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
			logger.V(logutil.DEFAULT).Info("EPP sent response body back to proxy")
			r.RequestState = BodyResponseResponsesComplete
		}
		// Dump the response so a new stream message can begin
		r.respBodyResp = nil
	}
	if r.RequestState == BodyResponseResponsesComplete && r.respTrailerResp != nil {
		// Trailers in requests are not guaranteed
		if err := srv.Send(r.respTrailerResp); err != nil {
			return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
		} else {
			logger.V(logutil.DEBUG).Info("EPP sent trailer back to proxy")
		}
	}
	return nil
}
