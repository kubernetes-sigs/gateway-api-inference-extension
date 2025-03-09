package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"time"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/structpb"
	"sigs.k8s.io/controller-runtime/pkg/log"
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
}

func (s *StreamingServer) Process(srv extProcPb.ExternalProcessor_ProcessServer) error {
	ctx := srv.Context()
	logger := log.FromContext(ctx)
	loggerVerbose := logger.V(logutil.VERBOSE)
	loggerVerbose.Info("Processing")

	// Create request context to share states during life time of an HTTP request.
	// See https://github.com/envoyproxy/envoy/issues/17540.
	reqCtx := &StreamingRequestContext{
		RequestState: RequestReceived,
	}

	reader, writer := io.Pipe()
	decoder := json.NewDecoder(reader)

	var requestBody, responseBody map[string]interface{}
	// Create error handling var as each request should only report once for
	// error metrics. This doesn't cover the error "Cannot receive stream request" because
	// such errors might happen even though response is processed.
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

		switch v := req.Request.(type) {
		case *extProcPb.ProcessingRequest_RequestHeaders:
			// Do nothing. Header info is handled in the HandleRequestBody func
		case *extProcPb.ProcessingRequest_RequestBody:
			loggerVerbose.Info("Incoming body chunk", "body", string(v.RequestBody.Body), "EoS", v.RequestBody.EndOfStream)
			// In the stream case, we can receive multiple request bodies.
			// To buffer the full message, we create a goroutine with a writer.Write()
			// call, which will block until the corresponding reader reads from it.
			// We do not read until we receive the EndofStream signal, and then
			// decode the entire JSON body.
			go func() {
				_, err := writer.Write(v.RequestBody.Body)
				if err != nil {
					logger.V(logutil.DEFAULT).Error(err, "Error populating writer")
				}
			}()

			// Message is buffered, we can read and decode.
			if v.RequestBody.EndOfStream {
				loggerVerbose.Info("decoding")
				err = decoder.Decode(&requestBody)
				if err != nil {
					logger.V(logutil.DEFAULT).Error(err, "Error unmarshaling request body")
				}
				// Body stream complete. Close the reader pipe,  and start anew for response.
				reader.Close()
				reader, writer = io.Pipe()
				decoder = json.NewDecoder(reader)

				reqCtx, err = s.HandleRequestBody(ctx, reqCtx, req, requestBody)
				if err != nil {
					logger.V(logutil.DEFAULT).Error(err, "Error handling body")
				} else {
					metrics.RecordRequestCounter(reqCtx.Model, reqCtx.ResolvedTargetModel)
					metrics.RecordRequestSizes(reqCtx.Model, reqCtx.ResolvedTargetModel, reqCtx.RequestSize)
				}
				loggerVerbose.Info("Request context after HandleRequestBody", "context", reqCtx)
			}
		case *extProcPb.ProcessingRequest_RequestTrailers:
			// This is currently unused.
		case *extProcPb.ProcessingRequest_ResponseHeaders:
			loggerVerbose.Info("got response headers", "headers", v.ResponseHeaders.Headers.GetHeaders())
			for _, header := range v.ResponseHeaders.Headers.GetHeaders() {
				code := header.RawValue[0]
				if header.Key == "status" && string(code) != "200" {
					reqCtx.ResponseStatusCode = errutil.ModelServerError
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
			go func() {
				_, err := writer.Write(v.ResponseBody.Body)
				if err != nil {
					logger.V(logutil.DEFAULT).Error(err, "Error populating writer")
				}
			}()

			// Message is buffered, we can read and decode.
			if v.ResponseBody.EndOfStream {
				err = decoder.Decode(&responseBody)
				if err != nil {
					logger.V(logutil.DEFAULT).Error(err, "Error unmarshaling request body")
				}
				// Body stream complete. Close the reader pipe.
				reader.Close()

				reqCtx, err = s.HandleResponseBody(ctx, reqCtx, responseBody)
				if err == nil && reqCtx.ResponseComplete {
					reqCtx.ResponseCompleteTimestamp = time.Now()
					metrics.RecordRequestLatencies(ctx, reqCtx.Model, reqCtx.ResolvedTargetModel, reqCtx.RequestReceivedTimestamp, reqCtx.ResponseCompleteTimestamp)
					metrics.RecordResponseSizes(reqCtx.Model, reqCtx.ResolvedTargetModel, reqCtx.ResponseSize)
					metrics.RecordInputTokens(reqCtx.Model, reqCtx.ResolvedTargetModel, reqCtx.Usage.PromptTokens)
					metrics.RecordOutputTokens(reqCtx.Model, reqCtx.ResolvedTargetModel, reqCtx.Usage.CompletionTokens)
				}
				loggerVerbose.Info("Request context after HandleResponseBody", "context", reqCtx)
			}
		case *extProcPb.ProcessingRequest_ResponseTrailers:
			// This is currently unused.
		}

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
		loggerVerbose.Info("checking", "request state", reqCtx.RequestState)
		if err := reqCtx.updateStateAndSendIfNeeded(srv, loggerVerbose); err != nil {
			return err
		}
	}
}

// updateStateAndSendIfNeeded checks state and can send mutiple responses in a single pass, but only if ordered properly.
// Order of requests matter in FULL_DUPLEX_STREAMING. For both request and response, the order of response sent back MUST be: Header->Body->Trailer, with trailer being optional.
func (r *StreamingRequestContext) updateStateAndSendIfNeeded(srv extProcPb.ExternalProcessor_ProcessServer, loggerVerbose logr.Logger) error {
	// No switch statement as we could send multiple responses in one pass.
	if r.RequestState == RequestReceived && r.reqHeaderResp != nil {
		loggerVerbose.Info("Request header response", "obj", r.reqHeaderResp)
		if err := srv.Send(r.reqHeaderResp); err != nil {
			loggerVerbose.Error(err, "error sending response")
			return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
		}
		r.RequestState = HeaderRequestResponseComplete
	}
	if r.RequestState == HeaderRequestResponseComplete && r.reqBodyResp != nil {
		loggerVerbose.Info("Request body response", "obj", r.reqBodyResp)
		if err := srv.Send(r.reqBodyResp); err != nil {
			return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
		}
		r.RequestState = BodyRequestResponsesComplete
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
		loggerVerbose.Info("Response header response", "obj", r.respHeaderResp)
		if err := srv.Send(r.respHeaderResp); err != nil {
			return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
		}
		r.RequestState = HeaderResponseResponseComplete
	}
	if r.RequestState == HeaderResponseResponseComplete && r.respBodyResp != nil {
		loggerVerbose.Info("Response body response", "obj", r.respBodyResp)
		if err := srv.Send(r.respBodyResp); err != nil {
			return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
		}
		r.RequestState = BodyResponseResponsesComplete
		// Dump the response so a new stream message can begin
		r.reqBodyResp = nil
	}
	if r.RequestState == BodyResponseResponsesComplete && r.respTrailerResp != nil {
		// Trailers in requests are not guaranteed
		if err := srv.Send(r.reqHeaderResp); err != nil {
			return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
		}
	}
	return nil
}

type StreamingRequestContext struct {
	TargetPod                 string
	TargetEndpoint            string
	Model                     string
	ResolvedTargetModel       string
	RequestState              StreamRequestState
	RequestReceivedTimestamp  time.Time
	ResponseCompleteTimestamp time.Time
	RequestSize               int
	Usage                     Usage
	ResponseSize              int
	ResponseComplete          bool
	ResponseStatusCode        string

	reqHeaderResp  *extProcPb.ProcessingResponse
	reqBodyResp    *extProcPb.ProcessingResponse
	reqTrailerResp *extProcPb.ProcessingResponse

	respHeaderResp  *extProcPb.ProcessingResponse
	respBodyResp    *extProcPb.ProcessingResponse
	respTrailerResp *extProcPb.ProcessingResponse
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

// HandleRequestBody always returns the requestContext even in the error case, as the request context is used in error handling.
func (s *StreamingServer) HandleRequestBody(
	ctx context.Context,
	reqCtx *StreamingRequestContext,
	req *extProcPb.ProcessingRequest,
	requestBodyMap map[string]interface{},
) (*StreamingRequestContext, error) {
	var requestBodyBytes []byte
	logger := log.FromContext(ctx)
	loggerVerbose := logger.V(logutil.VERBOSE)
	loggerVerbose.Info("Handling request body")

	// Resolve target models.
	model, ok := requestBodyMap["model"].(string)
	if !ok {
		return reqCtx, errutil.Error{Code: errutil.BadRequest, Msg: "model not found in request"}
	}
	loggerVerbose.Info("Model requested", "model", model)
	modelName := model

	// NOTE: The nil checking for the modelObject means that we DO allow passthrough currently.
	// This might be a security risk in the future where adapters not registered in the InferenceModel
	// are able to be requested by using their distinct name.
	modelObj := s.datastore.ModelGet(model)
	if modelObj == nil {
		return reqCtx, errutil.Error{Code: errutil.BadConfiguration, Msg: fmt.Sprintf("error finding a model object in InferenceModel for input %v", model)}
	}
	if len(modelObj.Spec.TargetModels) > 0 {
		modelName = datastore.RandomWeightedDraw(logger, modelObj, 0)
		if modelName == "" {
			return reqCtx, errutil.Error{Code: errutil.BadConfiguration, Msg: fmt.Sprintf("error getting target model name for model %v", modelObj.Name)}
		}
	}
	llmReq := &scheduling.LLMRequest{
		Model:               model,
		ResolvedTargetModel: modelName,
		Critical:            datastore.IsCritical(modelObj),
	}
	loggerVerbose.Info("LLM request assembled", "request", llmReq)

	var err error
	// Update target models in the body.
	if llmReq.Model != llmReq.ResolvedTargetModel {
		requestBodyMap["model"] = llmReq.ResolvedTargetModel
		requestBodyBytes, err = json.Marshal(requestBodyMap)
		if err != nil {
			logger.V(logutil.DEFAULT).Error(err, "Error marshaling request body")
			return reqCtx, errutil.Error{Code: errutil.Internal, Msg: fmt.Sprintf("error marshaling request body: %v", err)}
		}
		loggerVerbose.Info("Updated request body marshalled", "body", string(requestBodyBytes))
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
		"model", llmReq.Model, "targetModel", llmReq.ResolvedTargetModel, "endpoint", targetPod)

	reqCtx.Model = llmReq.Model
	reqCtx.ResolvedTargetModel = llmReq.ResolvedTargetModel
	reqCtx.RequestSize = len(requestBodyBytes)
	reqCtx.TargetPod = targetPod.NamespacedName.String()
	reqCtx.TargetEndpoint = endpoint

	headers := []*configPb.HeaderValueOption{
		{
			Header: &configPb.HeaderValue{
				Key:      s.destinationEndpointHintKey,
				RawValue: []byte(endpoint),
			},
		},
		// We need to update the content length header if the body is mutated, see Envoy doc:
		// https://www.envoyproxy.io/docs/envoy/latest/api-v3/extensions/filters/http/ext_proc/v3/processing_mode.proto
		{
			Header: &configPb.HeaderValue{
				Key:      "Content-Length",
				RawValue: []byte(strconv.Itoa(len(requestBodyBytes))),
			},
		},
	}
	// Print headers for debugging
	for _, header := range headers {
		logger.V(logutil.DEBUG).Info("Request body header", "key", header.Header.Key, "value", header.Header.RawValue)
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
	reqCtx *StreamingRequestContext,
	response map[string]interface{},
) (*StreamingRequestContext, error) {
	logger := log.FromContext(ctx)
	loggerVerbose := logger.V(logutil.VERBOSE)
	loggerVerbose.Info("Processing HandleResponseBody")
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
		loggerVerbose.Info("Response generated", "usage", reqCtx.Usage)
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
