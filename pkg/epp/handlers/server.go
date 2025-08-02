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
	"io"
	"strings"
	"sync"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"github.com/go-logr/logr"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"sigs.k8s.io/controller-runtime/pkg/log"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
)

const (
	// Certain envoy implementations set a max limit of 64Kb per streamed chunk, intentionally setting this lower for a safe margin.
	bodyByteLimit = 62000
)

func NewStreamingServer(destinationEndpointHintMetadataNamespace, destinationEndpointHintKey string, datastore Datastore, director Director) *StreamingServer {
	return &StreamingServer{
		destinationEndpointHintMetadataNamespace: destinationEndpointHintMetadataNamespace,
		destinationEndpointHintKey:               destinationEndpointHintKey,
		director:                                 director,
		datastore:                                datastore,
		workerInstanceCache:                      make(map[string]string),
		workerInstanceMutex:                      sync.RWMutex{},
	}
}

type Director interface {
	HandleRequest(ctx context.Context, reqCtx *RequestContext) (*RequestContext, error)
	HandleResponse(ctx context.Context, reqCtx *RequestContext) (*RequestContext, error)
	GetRandomPod() *backend.Pod
}

type Datastore interface {
	PoolGet() (*v1.InferencePool, error)
}

// Server implements the Envoy external processing server.
// https://www.envoyproxy.io/docs/envoy/latest/api-v3/service/ext_proc/v3/external_processor.proto
type StreamingServer struct {
	// The key of the header to specify the target pod address. This value needs to match Envoy
	// configuration.
	destinationEndpointHintKey string
	// The key acting as the outer namespace struct in the metadata extproc response to communicate
	// back the picked endpoints.
	destinationEndpointHintMetadataNamespace string
	datastore                                Datastore
	director                                 Director

	// Worker instance ID cache to store worker_instance_id for each session
	workerInstanceCache map[string]string
	workerInstanceMutex sync.RWMutex
}

// RequestContext stores context information during the life time of an HTTP request.
// TODO: The requestContext is gathering a ton of fields. A future refactor needs to tease these fields apart.
// Specifically, there are fields related to the ext-proc protocol, and then fields related to the lifecycle of the request.
// We should split these apart as this monolithic object exposes too much data to too many layers.
type RequestContext struct {
	TargetPod                 *backend.Pod
	TargetEndpoint            string
	Model                     string
	ResolvedTargetModel       string
	RequestReceivedTimestamp  time.Time
	ResponseCompleteTimestamp time.Time
	RequestSize               int
	Usage                     Usage
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
	Headers     map[string]string
	Body        map[string]any
	Metadata    map[string]any
	Annotations []string
}
type Response struct {
	Headers map[string]string
}
type StreamRequestState int

const (
	RequestReceived                  StreamRequestState = 0
	HeaderRequestResponseComplete    StreamRequestState = 1
	BodyRequestResponsesComplete     StreamRequestState = 2
	TrailerRequestResponsesComplete  StreamRequestState = 3
	ResponseRecieved                 StreamRequestState = 4
	HeaderResponseResponseComplete   StreamRequestState = 5
	BodyResponseResponsesComplete    StreamRequestState = 6
	TrailerResponseResponsesComplete StreamRequestState = 7
)

func (s *StreamingServer) Process(srv extProcPb.ExternalProcessor_ProcessServer) error {
	ctx := srv.Context()
	logger := log.FromContext(ctx)
	loggerTrace := logger.V(logutil.TRACE)
	loggerTrace.Info("Processing")

	// Create request context to share states during life time of an HTTP request.
	// See https://github.com/envoyproxy/envoy/issues/17540.
	reqCtx := &RequestContext{
		RequestState: RequestReceived,
		Request: &Request{
			Headers:     make(map[string]string),
			Body:        make(map[string]any),
			Metadata:    make(map[string]any),
			Annotations: []string{},
		},
		Response: &Response{
			Headers: make(map[string]string),
		},
	}

	var body []byte
	var responseBody map[string]any

	// Create error handling var as each request should only report once for
	// error metrics. This doesn't cover the error "Cannot receive stream request" because
	// such errors might happen even though response is processed.
	var err error
	defer func(error, *RequestContext) {
		if reqCtx.ResponseStatusCode != "" {
			metrics.RecordRequestErrCounter(reqCtx.Model, reqCtx.ResolvedTargetModel, reqCtx.ResponseStatusCode)
		} else if err != nil {
			metrics.RecordRequestErrCounter(reqCtx.Model, reqCtx.ResolvedTargetModel, errutil.CanonicalCode(err))
		}
		if reqCtx.RequestRunning {
			metrics.DecRunningRequests(reqCtx.Model)
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
			// This error occurs very frequently, though it doesn't seem to have any impact.
			// TODO Figure out if we can remove this noise.
			logger.V(logutil.DEFAULT).Error(err, "Cannot receive stream request")
			return status.Errorf(codes.Unknown, "cannot receive stream request: %v", err)
		}

		reqCtx.Request.Metadata = requtil.ExtractMetadataValues(req)

		switch v := req.Request.(type) {
		case *extProcPb.ProcessingRequest_RequestHeaders:
			if requestId := requtil.ExtractHeaderValue(v, requtil.RequestIdHeaderKey); len(requestId) > 0 {
				logger = logger.WithValues(requtil.RequestIdHeaderKey, requestId)
				loggerTrace = logger.V(logutil.TRACE)
				ctx = log.IntoContext(ctx, logger)
			}
			err = s.HandleRequestHeaders(ctx, reqCtx, v)
		case *extProcPb.ProcessingRequest_RequestBody:
			loggerTrace.Info("Incoming body chunk", "EoS", v.RequestBody.EndOfStream)
			// In the stream case, we can receive multiple request bodies.
			body = append(body, v.RequestBody.Body...)

			// Message is buffered, we can read and decode.
			if v.RequestBody.EndOfStream {
				loggerTrace.Info("decoding")
				err = json.Unmarshal(body, &reqCtx.Request.Body)
				if err != nil {
					if logger.V(logutil.DEBUG).Enabled() {
						err = errutil.Error{Code: errutil.BadRequest, Msg: "Error unmarshaling request body: " + string(body)}
					} else {
						err = errutil.Error{Code: errutil.BadRequest, Msg: "Error unmarshaling request body"}
					}
					break
				}

				// Body stream complete. Allocate empty slice for response to use.
				body = []byte{}

				reqCtx, err = s.director.HandleRequest(ctx, reqCtx)
				if err != nil {
					logger.V(logutil.DEFAULT).Error(err, "Error handling request")
					break
				}

				// Add query_instance_id annotation to the request metadata before sending to FrontEnd
				if reqCtx.Request != nil {
					// Ensure Annotations slice is initialized
					if reqCtx.Request.Annotations == nil {
						reqCtx.Request.Annotations = []string{}
					}

					// Add the annotation (if not already present)
					found := false
					for _, a := range reqCtx.Request.Annotations {
						if a == "query_instance_id" {
							found = true
							break
						}
					}

					if !found {
						reqCtx.Request.Annotations = append(reqCtx.Request.Annotations, "query_instance_id")
						logger.V(logutil.VERBOSE).Info("Added query_instance_id annotation to request")
					}
				}

				// Populate the ExtProc protocol responses for the request body.
				requestBodyBytes, err := json.Marshal(reqCtx.Request.Body)
				if err != nil {
					logger.V(logutil.DEFAULT).Error(err, "Error marshalling request body")
					break
				}

				// Log the complete request body being sent to FrontEnd for debugging
				logger.V(logutil.VERBOSE).Info("Sending request body to FrontEnd", "request_body", string(requestBodyBytes))
				reqCtx.RequestSize = len(requestBodyBytes)
				reqCtx.reqHeaderResp = s.generateRequestHeaderResponse(reqCtx)
				reqCtx.reqBodyResp = s.generateRequestBodyResponses(requestBodyBytes)

				metrics.RecordRequestCounter(reqCtx.Model, reqCtx.ResolvedTargetModel)
				metrics.RecordRequestSizes(reqCtx.Model, reqCtx.ResolvedTargetModel, reqCtx.RequestSize)
			}
		case *extProcPb.ProcessingRequest_RequestTrailers:
			// This is currently unused.
		case *extProcPb.ProcessingRequest_ResponseHeaders:
			for _, header := range v.ResponseHeaders.Headers.GetHeaders() {
				value := string(header.RawValue)

				loggerTrace.Info("header", "key", header.Key, "value", value)
				if header.Key == "status" && value != "200" {
					reqCtx.ResponseStatusCode = errutil.ModelServerError
				} else if header.Key == "content-type" && strings.Contains(value, "text/event-stream") {
					reqCtx.modelServerStreaming = true
					loggerTrace.Info("model server is streaming response")
				}
			}
			reqCtx.RequestState = ResponseRecieved

			var responseErr error
			reqCtx, responseErr = s.HandleResponseHeaders(ctx, reqCtx, v)
			if responseErr != nil {
				if logger.V(logutil.DEBUG).Enabled() {
					logger.V(logutil.DEBUG).Error(responseErr, "Failed to process response headers", "request", req)
				} else {
					logger.V(logutil.DEFAULT).Error(responseErr, "Failed to process response headers")
				}
			}
			reqCtx.respHeaderResp = s.generateResponseHeaderResponse(reqCtx)

		case *extProcPb.ProcessingRequest_ResponseBody:
			if reqCtx.modelServerStreaming {
				// Currently we punt on response parsing if the modelServer is streaming, and we just passthrough.

				responseText := string(v.ResponseBody.Body)
				s.HandleResponseBodyModelStreaming(ctx, reqCtx, responseText)
				if v.ResponseBody.EndOfStream {
					loggerTrace.Info("stream completed")

					reqCtx.ResponseCompleteTimestamp = time.Now()
					metrics.RecordRequestLatencies(ctx, reqCtx.Model, reqCtx.ResolvedTargetModel, reqCtx.RequestReceivedTimestamp, reqCtx.ResponseCompleteTimestamp)
					metrics.RecordResponseSizes(reqCtx.Model, reqCtx.ResolvedTargetModel, reqCtx.ResponseSize)
				}

				reqCtx.respBodyResp = generateResponseBodyResponses(v.ResponseBody.Body, v.ResponseBody.EndOfStream)
			} else {
				body = append(body, v.ResponseBody.Body...)

				// Message is buffered, we can read and decode.
				if v.ResponseBody.EndOfStream {
					loggerTrace.Info("stream completed")
					// Don't send a 500 on a response error. Just let the message passthrough and log our error for debugging purposes.
					// We assume the body is valid JSON, err messages are not guaranteed to be json, and so capturing and sending a 500 obfuscates the response message.
					// Using the standard 'err' var will send an immediate error response back to the caller.
					var responseErr error
					responseErr = json.Unmarshal(body, &responseBody)
					if responseErr != nil {
						if logger.V(logutil.DEBUG).Enabled() {
							logger.V(logutil.DEBUG).Error(responseErr, "Error unmarshalling request body", "body", string(body))
						} else {
							logger.V(logutil.DEFAULT).Error(responseErr, "Error unmarshalling request body", "body", string(body))
						}
						reqCtx.respBodyResp = generateResponseBodyResponses(body, true)
						break
					}

					reqCtx, responseErr = s.HandleResponseBody(ctx, reqCtx, responseBody)
					if responseErr != nil {
						if logger.V(logutil.DEBUG).Enabled() {
							logger.V(logutil.DEBUG).Error(responseErr, "Failed to process response body", "request", req)
						} else {
							logger.V(logutil.DEFAULT).Error(responseErr, "Failed to process response body")
						}
					} else if reqCtx.ResponseComplete {
						reqCtx.ResponseCompleteTimestamp = time.Now()
						metrics.RecordRequestLatencies(ctx, reqCtx.Model, reqCtx.ResolvedTargetModel, reqCtx.RequestReceivedTimestamp, reqCtx.ResponseCompleteTimestamp)
						metrics.RecordResponseSizes(reqCtx.Model, reqCtx.ResolvedTargetModel, reqCtx.ResponseSize)
						metrics.RecordInputTokens(reqCtx.Model, reqCtx.ResolvedTargetModel, reqCtx.Usage.PromptTokens)
						metrics.RecordOutputTokens(reqCtx.Model, reqCtx.ResolvedTargetModel, reqCtx.Usage.CompletionTokens)
					}
				}
			}
		case *extProcPb.ProcessingRequest_ResponseTrailers:
			// This is currently unused.
		}

		// Handle the err and fire an immediate response.
		if err != nil {
			if logger.V(logutil.DEBUG).Enabled() {
				logger.V(logutil.DEBUG).Error(err, "Failed to process request", "request", req)
			} else {
				logger.V(logutil.DEFAULT).Error(err, "Failed to process request")
			}
			resp, err := buildErrResponse(err)
			if err != nil {
				return err
			}
			if err := srv.Send(resp); err != nil {
				logger.V(logutil.DEFAULT).Error(err, "Send failed")
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

// updateStateAndSendIfNeeded checks state and can send mutiple responses in a single pass, but only if ordered properly.
// Order of requests matter in FULL_DUPLEX_STREAMING. For both request and response, the order of response sent back MUST be: Header->Body->Trailer, with trailer being optional.
func (r *RequestContext) updateStateAndSendIfNeeded(srv extProcPb.ExternalProcessor_ProcessServer, logger logr.Logger) error {
	loggerTrace := logger.V(logutil.TRACE)
	// No switch statement as we could send multiple responses in one pass.
	if r.RequestState == RequestReceived && r.reqHeaderResp != nil {
		loggerTrace.Info("Sending request header response", "obj", r.reqHeaderResp)
		if err := srv.Send(r.reqHeaderResp); err != nil {
			logger.V(logutil.DEFAULT).Error(err, "error sending response")
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
		r.RequestState = BodyRequestResponsesComplete
		metrics.IncRunningRequests(r.Model)
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
	if r.RequestState == ResponseRecieved && r.respHeaderResp != nil {
		loggerTrace.Info("Sending response header response", "obj", r.respHeaderResp)
		if err := srv.Send(r.respHeaderResp); err != nil {
			return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
		}
		r.RequestState = HeaderResponseResponseComplete
	}
	if r.RequestState == HeaderResponseResponseComplete && r.respBodyResp != nil && len(r.respBodyResp) > 0 {
		loggerTrace.Info("Sending response body response(s)")
		for _, response := range r.respBodyResp {
			if err := srv.Send(response); err != nil {
				return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
			}

			body := response.Response.(*extProcPb.ProcessingResponse_ResponseBody)
			if body.ResponseBody.Response.GetBodyMutation().GetStreamedResponse().GetEndOfStream() {
				r.RequestState = BodyResponseResponsesComplete
			}
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
	// This code can be returned by the director when there are no candidate pods for the request scheduling.
	case errutil.ServiceUnavailable:
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

	if err.Error() != "" {
		resp.Response.(*extProcPb.ProcessingResponse_ImmediateResponse).ImmediateResponse.Body = []byte(err.Error())
	}

	return resp, nil
}

func buildCommonResponses(bodyBytes []byte, byteLimit int, setEos bool) []*extProcPb.CommonResponse {
	responses := []*extProcPb.CommonResponse{}
	startingIndex := 0
	bodyLen := len(bodyBytes)

	if bodyLen == 0 {
		return []*extProcPb.CommonResponse{
			{
				BodyMutation: &extProcPb.BodyMutation{
					Mutation: &extProcPb.BodyMutation_StreamedResponse{
						StreamedResponse: &extProcPb.StreamedBodyResponse{
							Body:        bodyBytes,
							EndOfStream: setEos,
						},
					},
				},
			},
		}
	}

	for startingIndex < bodyLen {
		eos := false
		len := min(bodyLen-startingIndex, byteLimit)
		chunk := bodyBytes[startingIndex : len+startingIndex]
		if setEos && len+startingIndex >= bodyLen {
			eos = true
		}

		commonResp := &extProcPb.CommonResponse{
			BodyMutation: &extProcPb.BodyMutation{
				Mutation: &extProcPb.BodyMutation_StreamedResponse{
					StreamedResponse: &extProcPb.StreamedBodyResponse{
						Body:        chunk,
						EndOfStream: eos,
					},
				},
			},
		}
		responses = append(responses, commonResp)
		startingIndex += len
	}

	return responses
}

// getWorkerInstanceID retrieves the stored worker instance ID for a given session identifier
func (s *StreamingServer) getWorkerInstanceID(sessionID string) (string, bool) {
	s.workerInstanceMutex.RLock()
	defer s.workerInstanceMutex.RUnlock()
	workerID, exists := s.workerInstanceCache[sessionID]
	return workerID, exists
}

// setWorkerInstanceID stores the worker instance ID for a given session identifier
func (s *StreamingServer) setWorkerInstanceID(sessionID, workerInstanceID string) {
	s.workerInstanceMutex.Lock()
	defer s.workerInstanceMutex.Unlock()
	s.workerInstanceCache[sessionID] = workerInstanceID
}

// getSessionIdentifier extracts a session identifier from the request context
// This uses a combination of headers to identify the session/client
func (s *StreamingServer) getSessionIdentifier(reqCtx *RequestContext) string {
	// Try to use x-request-id if it represents a session
	if requestID, exists := reqCtx.Request.Headers["x-request-id"]; exists && requestID != "" {
		return requestID
	}

	// Fallback to using authorization header or other stable identifiers
	if auth, exists := reqCtx.Request.Headers["authorization"]; exists && auth != "" {
		return auth
	}

	// Fallback to user-agent + some other header combination
	userAgent := reqCtx.Request.Headers["user-agent"]
	xForwardedFor := reqCtx.Request.Headers["x-forwarded-for"]
	return userAgent + "|" + xForwardedFor
}
