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
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"strconv"
	"strings"
	"time"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/structpb"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

func NewStreamingServer(scheduler Scheduler, destinationEndpointHintMetadataNamespace, destinationEndpointHintKey string, datastore datastore.Datastore) *StreamingServer {
	return &StreamingServer{
		scheduler:                                scheduler,
		destinationEndpointHintMetadataNamespace: destinationEndpointHintMetadataNamespace,
		destinationEndpointHintKey:               destinationEndpointHintKey,
		datastore:                                datastore,

		// Default FrontEnd configuration (same sidecar)
		frontEndPort: "8000",
		httpClient: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

type StreamingServer struct {
	scheduler Scheduler
	// The key of the header to specify the target pod address. This value needs to match Envoy
	// configuration.
	destinationEndpointHintKey string
	// The key acting as the outer namespace struct in the metadata extproc response to communicate
	// back the picked endpoints.
	destinationEndpointHintMetadataNamespace string
	datastore                                datastore.Datastore

	// FrontEnd service configuration
	frontEndPort string
	httpClient   *http.Client
}

func (s *StreamingServer) Process(srv extProcPb.ExternalProcessor_ProcessServer) error {
	ctx := srv.Context()
	logger := log.FromContext(ctx)
	loggerTrace := logger.V(logutil.TRACE)
	loggerTrace.Info("Processing")

	// Create request context to share states during life time of an HTTP request.
	// See https://github.com/envoyproxy/envoy/issues/17540.
	reqCtx := &RequestContext{
		RequestState: RequestReceived,
	}

	var body []byte
	var requestBody, responseBody map[string]interface{}

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

		switch v := req.Request.(type) {
		case *extProcPb.ProcessingRequest_RequestHeaders:
			err = s.HandleRequestHeaders(ctx, reqCtx, v)
		case *extProcPb.ProcessingRequest_RequestBody:
			loggerTrace.Info("Incoming body chunk", "EoS", v.RequestBody.EndOfStream)
			// In the stream case, we can receive multiple request bodies.
			body = append(body, v.RequestBody.Body...)

			// Message is buffered, we can read and decode.
			if v.RequestBody.EndOfStream {
				loggerTrace.Info("decoding")
				err = json.Unmarshal(body, &requestBody)
				if err != nil {
					logger.V(logutil.DEFAULT).Error(err, "Error unmarshaling request body")
				}

				// Body stream complete. Allocate empty slice for response to use.
				body = []byte{}

				reqCtx, err = s.HandleRequestBody(ctx, reqCtx, req, requestBody)
				if err != nil {
					logger.V(logutil.DEFAULT).Error(err, "Error handling body")
				} else {
					metrics.RecordRequestCounter(reqCtx.Model, reqCtx.ResolvedTargetModel)
					metrics.RecordRequestSizes(reqCtx.Model, reqCtx.ResolvedTargetModel, reqCtx.RequestSize)
				}
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
			reqCtx.respHeaderResp = &extProcPb.ProcessingResponse{
				Response: &extProcPb.ProcessingResponse_ResponseHeaders{
					ResponseHeaders: &extProcPb.HeadersResponse{
						Response: &extProcPb.CommonResponse{
							HeaderMutation: &extProcPb.HeaderMutation{
								SetHeaders: []*configPb.HeaderValueOption{
									{
										Header: &configPb.HeaderValue{
											// This is for debugging purpose only.
											Key:      "x-went-into-resp-headers",
											RawValue: []byte("true"),
										},
									},
								},
							},
						},
					},
				},
			}

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

				reqCtx.respBodyResp = &extProcPb.ProcessingResponse{
					Response: &extProcPb.ProcessingResponse_ResponseBody{
						ResponseBody: &extProcPb.BodyResponse{
							Response: &extProcPb.CommonResponse{
								BodyMutation: &extProcPb.BodyMutation{
									Mutation: &extProcPb.BodyMutation_StreamedResponse{
										StreamedResponse: &extProcPb.StreamedBodyResponse{
											Body:        v.ResponseBody.Body,
											EndOfStream: v.ResponseBody.EndOfStream,
										},
									},
								},
							},
						},
					},
				}
			} else {
				body = append(body, v.ResponseBody.Body...)

				// Message is buffered, we can read and decode.
				if v.ResponseBody.EndOfStream {
					loggerTrace.Info("stream completed")
					// Don't send a 500 on a response error. Just let the message passthrough and log our error for debugging purposes.
					// We assume the body is valid JSON, err messages are not guaranteed to be json, and so capturing and sending a 500 obfuscates the response message.
					// using the standard 'err' var will send an immediate error response back to the caller.
					var responseErr error
					responseErr = json.Unmarshal(body, &responseBody)
					if responseErr != nil {
						logger.V(logutil.DEFAULT).Error(responseErr, "Error unmarshaling request body")
					}

					reqCtx, responseErr = s.HandleResponseBody(ctx, reqCtx, responseBody)
					if responseErr != nil {
						logger.V(logutil.DEFAULT).Error(responseErr, "Failed to process response body", "request", req)
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
			logger.V(logutil.DEFAULT).Error(err, "Failed to process request", "request", req)
			resp, err := BuildErrResponse(err)
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
	if r.RequestState == HeaderRequestResponseComplete && r.reqBodyResp != nil {
		loggerTrace.Info("Sending request body response")
		if err := srv.Send(r.reqBodyResp); err != nil {
			return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
		}
		r.RequestState = BodyRequestResponsesComplete
		metrics.IncRunningRequests(r.Model)
		r.RequestRunning = true
		// Dump the response so a new stream message can begin
		r.reqBodyResp = nil
	}
	if r.RequestState == BodyRequestResponsesComplete && r.reqTrailerResp != nil {
		// Trailers in requests are not guaranteed
		if err := srv.Send(r.reqHeaderResp); err != nil {
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
	if r.RequestState == HeaderResponseResponseComplete && r.respBodyResp != nil {
		loggerTrace.Info("Sending response body response")
		if err := srv.Send(r.respBodyResp); err != nil {
			return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
		}

		body := r.respBodyResp.Response.(*extProcPb.ProcessingResponse_ResponseBody)
		if body.ResponseBody.Response.GetBodyMutation().GetStreamedResponse().GetEndOfStream() {
			r.RequestState = BodyResponseResponsesComplete
		}
		// Dump the response so a new stream message can begin
		r.respBodyResp = nil
	}
	if r.RequestState == BodyResponseResponsesComplete && r.respTrailerResp != nil {
		// Trailers in requests are not guaranteed
		if err := srv.Send(r.reqHeaderResp); err != nil {
			return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
		}
	}
	return nil
}

// HandleRequestBody always returns the requestContext even in the error case, as the request context is used in error handling.
func (s *StreamingServer) HandleRequestBody(
	ctx context.Context,
	reqCtx *RequestContext,
	req *extProcPb.ProcessingRequest,
	requestBodyMap map[string]interface{},
) (*RequestContext, error) {
	var requestBodyBytes []byte
	logger := log.FromContext(ctx)

	// Resolve target models.
	model, ok := requestBodyMap["model"].(string)
	if !ok {
		return reqCtx, errutil.Error{Code: errutil.BadRequest, Msg: "model not found in request"}
	}

	modelName := model

	// NOTE: The nil checking for the modelObject means that we DO allow passthrough currently.
	// This might be a security risk in the future where adapters not registered in the InferenceModel
	// are able to be requested by using their distinct name.
	modelObj := s.datastore.ModelGet(model)
	if modelObj == nil {
		return reqCtx, errutil.Error{Code: errutil.BadConfiguration, Msg: fmt.Sprintf("error finding a model object in InferenceModel for input %v", model)}
	}
	if len(modelObj.Spec.TargetModels) > 0 {
		modelName = RandomWeightedDraw(logger, modelObj, 0)
		if modelName == "" {
			return reqCtx, errutil.Error{Code: errutil.BadConfiguration, Msg: fmt.Sprintf("error getting target model name for model %v", modelObj.Name)}
		}
	}
	llmReq := &scheduling.LLMRequest{
		Model:               model,
		ResolvedTargetModel: modelName,
		Critical:            datastore.IsCritical(modelObj),
	}
	logger.V(logutil.DEBUG).Info("LLM request assembled", "model", llmReq.Model, "targetModel", llmReq.ResolvedTargetModel, "critical", llmReq.Critical)

	var err error
	// Update target models in the body.
	if llmReq.Model != llmReq.ResolvedTargetModel {
		requestBodyMap["model"] = llmReq.ResolvedTargetModel
	}

	requestBodyBytes, err = json.Marshal(requestBodyMap)
	if err != nil {
		logger.V(logutil.DEFAULT).Error(err, "Error marshaling request body")
		return reqCtx, errutil.Error{Code: errutil.Internal, Msg: fmt.Sprintf("error marshaling request body: %v", err)}
	}

	target, err := s.scheduler.Schedule(ctx, llmReq)
	if err != nil {
		return reqCtx, errutil.Error{Code: errutil.InferencePoolResourceExhausted, Msg: fmt.Errorf("failed to find target pod: %w", err).Error()}
	}
	targetPod := target.GetPod()

	// Insert target endpoint to instruct Envoy to route requests to the specified target pod.
	// Attach the port number
	pool, err := s.datastore.PoolGet()
	if err != nil {
		return reqCtx, err
	}
	endpoint := targetPod.Address + ":" + strconv.Itoa(int(pool.Spec.TargetPortNumber))

	logger.V(logutil.DEFAULT).Info("Request handled",
		"model", llmReq.Model, "targetModel", llmReq.ResolvedTargetModel, "endpoint", targetPod, "endpoint metrics",
		fmt.Sprintf("%+v", target))

	// Call FrontEnd service to get worker instance ID
	if err = s.fetchWorkerIDFromFrontEnd(ctx, reqCtx); err != nil {
		logger.V(logutil.DEFAULT).Error(err, "Failed to fetch worker ID from FrontEnd service")
		// Continue processing even if FrontEnd call fails
	}

	reqCtx.Model = llmReq.Model
	reqCtx.ResolvedTargetModel = llmReq.ResolvedTargetModel
	reqCtx.RequestSize = len(requestBodyBytes)
	reqCtx.TargetPod = targetPod.NamespacedName.String()
	reqCtx.TargetEndpoint = endpoint

	s.populateRequestHeaderResponse(reqCtx, endpoint, len(requestBodyBytes))

	reqCtx.reqBodyResp = &extProcPb.ProcessingResponse{
		// The Endpoint Picker supports two approaches to communicating the target endpoint, as a request header
		// and as an unstructure ext-proc response metadata key/value pair. This enables different integration
		// options for gateway providers.
		Response: &extProcPb.ProcessingResponse_RequestBody{
			RequestBody: &extProcPb.BodyResponse{
				Response: &extProcPb.CommonResponse{
					BodyMutation: &extProcPb.BodyMutation{
						Mutation: &extProcPb.BodyMutation_StreamedResponse{
							StreamedResponse: &extProcPb.StreamedBodyResponse{
								Body:        requestBodyBytes,
								EndOfStream: true,
							},
						},
					},
				},
			},
		},
	}
	return reqCtx, nil
}

// HandleResponseBody always returns the requestContext even in the error case, as the request context is used in error handling.
func (s *StreamingServer) HandleResponseBody(
	ctx context.Context,
	reqCtx *RequestContext,
	response map[string]interface{},
) (*RequestContext, error) {
	logger := log.FromContext(ctx)
	responseBytes, err := json.Marshal(response)
	if err != nil {
		logger.V(logutil.DEFAULT).Error(err, "error marshalling responseBody")
		return reqCtx, err
	}
	if response["usage"] != nil {
		usg := response["usage"].(map[string]interface{})
		usage := Usage{
			PromptTokens:     int(usg["prompt_tokens"].(float64)),
			CompletionTokens: int(usg["completion_tokens"].(float64)),
			TotalTokens:      int(usg["total_tokens"].(float64)),
		}
		reqCtx.Usage = usage
		logger.V(logutil.VERBOSE).Info("Response generated", "usage", reqCtx.Usage)
	}
	reqCtx.ResponseSize = len(responseBytes)
	// ResponseComplete is to indicate the response is complete. In non-streaming
	// case, it will be set to be true once the response is processed; in
	// streaming case, it will be set to be true once the last chunk is processed.
	// TODO(https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/178)
	// will add the processing for streaming case.
	reqCtx.ResponseComplete = true

	reqCtx.respBodyResp = &extProcPb.ProcessingResponse{
		// The Endpoint Picker supports two approaches to communicating the target endpoint, as a request header
		// and as an unstructure ext-proc response metadata key/value pair. This enables different integration
		// options for gateway providers.
		Response: &extProcPb.ProcessingResponse_ResponseBody{
			ResponseBody: &extProcPb.BodyResponse{
				Response: &extProcPb.CommonResponse{
					BodyMutation: &extProcPb.BodyMutation{
						Mutation: &extProcPb.BodyMutation_StreamedResponse{
							StreamedResponse: &extProcPb.StreamedBodyResponse{
								Body:        responseBytes,
								EndOfStream: true,
							},
						},
					},
				},
			},
		},
	}
	return reqCtx, nil
}

// The function is to handle streaming response if the modelServer is streaming.
func (s *StreamingServer) HandleResponseBodyModelStreaming(
	ctx context.Context,
	reqCtx *RequestContext,
	responseText string,
) {
	if strings.Contains(responseText, streamingEndMsg) {
		resp := ParseRespForUsage(ctx, responseText)
		metrics.RecordInputTokens(reqCtx.Model, reqCtx.ResolvedTargetModel, resp.Usage.PromptTokens)
		metrics.RecordOutputTokens(reqCtx.Model, reqCtx.ResolvedTargetModel, resp.Usage.CompletionTokens)
	}
}

func (s *StreamingServer) HandleRequestHeaders(ctx context.Context, reqCtx *RequestContext, req *extProcPb.ProcessingRequest_RequestHeaders) error {
	reqCtx.RequestReceivedTimestamp = time.Now()

	// an EoS in the request headers means this request has no body or trailers.
	if req.RequestHeaders.EndOfStream {
		// We will route this request to a random pod as this is assumed to just be a GET
		// More context: https://github.com/kubernetes-sigs/gateway-api-inference-extension/pull/526
		// The above PR will address endpoint admission, but currently any request without a body will be
		// routed to a random upstream pod.
		pod := GetRandomPod(s.datastore)
		pool, err := s.datastore.PoolGet()
		if err != nil {
			return err
		}
		endpoint := pod.Address + ":" + strconv.Itoa(int(pool.Spec.TargetPortNumber))
		s.populateRequestHeaderResponse(reqCtx, endpoint, 0)
	}
	return nil
}

// SetFrontEndConfig allows configuring the FrontEnd service endpoint
func (s *StreamingServer) SetFrontEndConfig(port string) {
	s.frontEndPort = port
}

// fetchWorkerIDFromFrontEnd makes a blocking HTTP request to the FrontEnd service
// to get the worker_instance_id for this request
func (s *StreamingServer) fetchWorkerIDFromFrontEnd(ctx context.Context, reqCtx *RequestContext) error {
	logger := log.FromContext(ctx)

	// Construct FrontEnd URL
	frontEndURL := fmt.Sprintf("http://localhost:%s/v1/chat/completions", s.frontEndPort)

	// Create request body with nvext annotations
	requestBody := map[string]interface{}{
		"nvext": map[string]interface{}{
			"annotations": []string{"query_instance_id"},
		},
	}

	requestJSON, err := json.Marshal(requestBody)
	if err != nil {
		logger.V(logutil.DEFAULT).Error(err, "Failed to marshal request body")
		return fmt.Errorf("failed to marshal request body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, frontEndURL, bytes.NewBuffer(requestJSON))
	if err != nil {
		logger.V(logutil.DEFAULT).Error(err, "Failed to create FrontEnd request")
		return fmt.Errorf("failed to create FrontEnd request: %w", err)
	}

	// Set appropriate headers
	req.Header.Set("Content-Type", "application/json")

	// Make the blocking HTTP request
	logger.V(logutil.VERBOSE).Info("Making blocking HTTP request to FrontEnd",
		"url", frontEndURL,
		"body", string(requestJSON))
	resp, err := s.httpClient.Do(req)
	if err != nil {
		logger.V(logutil.DEFAULT).Error(err, "Failed to make FrontEnd request")
		return fmt.Errorf("failed to make FrontEnd request: %w", err)
	}
	defer resp.Body.Close()

	// Read response body
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		logger.V(logutil.DEFAULT).Error(err, "Failed to read FrontEnd response body")
		return fmt.Errorf("failed to read FrontEnd response body: %w", err)
	}

	// Parse JSON response
	var responseData map[string]interface{}
	if err := json.Unmarshal(body, &responseData); err != nil {
		logger.V(logutil.DEFAULT).Error(err, "Failed to unmarshal FrontEnd response")
		return fmt.Errorf("failed to unmarshal FrontEnd response: %w", err)
	}

	// Extract worker_instance_id
	if workerID, exists := responseData["worker_instance_id"]; exists {
		if workerIDStr, ok := workerID.(string); ok && workerIDStr != "" {
			reqCtx.WorkerInstanceID = workerIDStr
			logger.V(logutil.VERBOSE).Info("Extracted worker instance ID from FrontEnd", "worker_instance_id", workerIDStr)
			return nil
		}
	}

	logger.V(logutil.DEFAULT).Info("worker_instance_id not found or empty in FrontEnd response")
	return fmt.Errorf("worker_instance_id not found in FrontEnd response")
}

func (s *StreamingServer) populateRequestHeaderResponse(reqCtx *RequestContext, endpoint string, requestBodyLength int) {
	headers := []*configPb.HeaderValueOption{
		{
			Header: &configPb.HeaderValue{
				Key:      s.destinationEndpointHintKey,
				RawValue: []byte(endpoint),
			},
		},
	}

	// Add worker ID header if obtained from FrontEnd service
	if reqCtx.WorkerInstanceID != "" {
		headers = append(headers, &configPb.HeaderValueOption{
			Header: &configPb.HeaderValue{
				Key:      "x-gateway-worker-id",
				RawValue: []byte(reqCtx.WorkerInstanceID),
			},
		})
		headers = append(headers, &configPb.HeaderValueOption{
			Header: &configPb.HeaderValue{
				Key:      "host_rewrite_header",
				RawValue: []byte("x-gateway-destination-endpoint"),
			},
		})
	}

	if requestBodyLength > 0 {
		// We need to update the content length header if the body is mutated, see Envoy doc:
		// https://www.envoyproxy.io/docs/envoy/latest/api-v3/extensions/filters/http/ext_proc/v3/processing_mode.proto
		headers = append(headers, &configPb.HeaderValueOption{
			Header: &configPb.HeaderValue{
				Key:      "Content-Length",
				RawValue: []byte(strconv.Itoa(requestBodyLength)),
			},
		})
	}

	targetEndpointValue := &structpb.Struct{
		Fields: map[string]*structpb.Value{
			s.destinationEndpointHintKey: {
				Kind: &structpb.Value_StringValue{
					StringValue: endpoint,
				},
			},
		},
	}
	dynamicMetadata := targetEndpointValue
	if s.destinationEndpointHintMetadataNamespace != "" {
		// If a namespace is defined, wrap the selected endpoint with that.
		dynamicMetadata = &structpb.Struct{
			Fields: map[string]*structpb.Value{
				s.destinationEndpointHintMetadataNamespace: {
					Kind: &structpb.Value_StructValue{
						StructValue: targetEndpointValue,
					},
				},
			},
		}
	}

	reqCtx.reqHeaderResp = &extProcPb.ProcessingResponse{
		Response: &extProcPb.ProcessingResponse_RequestHeaders{
			RequestHeaders: &extProcPb.HeadersResponse{
				Response: &extProcPb.CommonResponse{
					ClearRouteCache: true,
					HeaderMutation: &extProcPb.HeaderMutation{
						SetHeaders: headers,
					},
				},
			},
		},
		DynamicMetadata: dynamicMetadata,
	}
}

func RandomWeightedDraw(logger logr.Logger, model *v1alpha2.InferenceModel, seed int64) string {
	// TODO: after we are down to 1 server implementation, make these methods a part of the struct
	// and handle random seeding on the struct.
	source := rand.NewSource(rand.Int63())
	if seed > 0 {
		source = rand.NewSource(seed)
	}
	r := rand.New(source)

	// all the weight values are nil, then we should return random model name
	if model.Spec.TargetModels[0].Weight == nil {
		index := r.Int31n(int32(len(model.Spec.TargetModels)))
		return model.Spec.TargetModels[index].Name
	}

	var weights int32
	for _, model := range model.Spec.TargetModels {
		weights += *model.Weight
	}
	logger.V(logutil.TRACE).Info("Weights for model computed", "model", model.Name, "weights", weights)
	randomVal := r.Int31n(weights)
	// TODO: optimize this without using loop
	for _, model := range model.Spec.TargetModels {
		if randomVal < *model.Weight {
			return model.Name
		}
		randomVal -= *model.Weight
	}
	return ""
}

func GetRandomPod(ds datastore.Datastore) *backendmetrics.Pod {
	pods := ds.PodGetAll()
	number := rand.Intn(len(pods))
	pod := pods[number]
	return pod.GetPod()
}
