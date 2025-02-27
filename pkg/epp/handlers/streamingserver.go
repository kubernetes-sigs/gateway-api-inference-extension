package handlers

import (
	"io"
	"strconv"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/structpb"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
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
	beeps, boops, bops := 0, 0, 0
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

		// var resp *extProcPb.ProcessingResponse
		// switch v := req.Request.(type) {
		// case *extProcPb.ProcessingRequest_RequestHeaders:
		// 	reqCtx.RequestReceivedTimestamp = time.Now()
		// 	resp = HandleRequestHeaders(ctx, reqCtx, req)
		// 	loggerVerbose.Info("Request context after HandleRequestHeaders", "context", reqCtx)
		// case *extProcPb.ProcessingRequest_RequestBody:
		// 	loggerVerbose.Info("[TESTING] Request body before entering func", "body", req.GetRequestBody())

		// 	resp, err = s.HandleRequestBody(ctx, reqCtx, req)
		// 	if !reqCtx.EndofStream {
		// 		break
		// 	}
		// 	if err == nil {
		// 		metrics.RecordRequestCounter(reqCtx.Model, reqCtx.ResolvedTargetModel)
		// 		metrics.RecordRequestSizes(reqCtx.Model, reqCtx.ResolvedTargetModel, reqCtx.RequestSize)
		// 	}
		// 	loggerVerbose.Info("Request context after HandleRequestBody", "context", reqCtx)
		// case *extProcPb.ProcessingRequest_ResponseHeaders:
		// 	resp, err = s.HandleResponseHeaders(ctx, reqCtx, req)
		// 	loggerVerbose.Info("Request context after HandleResponseHeaders", "context", reqCtx)
		// case *extProcPb.ProcessingRequest_ResponseBody:
		// 	resp, err = s.HandleResponseBody(ctx, reqCtx, req)
		// 	if err == nil && reqCtx.ResponseComplete {
		// 		reqCtx.ResponseCompleteTimestamp = time.Now()
		// 		metrics.RecordRequestLatencies(ctx, reqCtx.Model, reqCtx.ResolvedTargetModel, reqCtx.RequestReceivedTimestamp, reqCtx.ResponseCompleteTimestamp)
		// 		metrics.RecordResponseSizes(reqCtx.Model, reqCtx.ResolvedTargetModel, reqCtx.ResponseSize)
		// 		metrics.RecordInputTokens(reqCtx.Model, reqCtx.ResolvedTargetModel, reqCtx.Response.Usage.PromptTokens)
		// 		metrics.RecordOutputTokens(reqCtx.Model, reqCtx.ResolvedTargetModel, reqCtx.Response.Usage.CompletionTokens)
		// 	}
		// 	loggerVerbose.Info("Request context after HandleResponseBody", "context", reqCtx)
		// case *extProcPb.ProcessingRequest_RequestTrailers:
		// case *extProcPb.ProcessingRequest_ResponseTrailers:
		// default:
		// 	logger.V(logutil.DEFAULT).Error(nil, "Unknown Request type", "request", v)
		// 	return status.Error(codes.Unknown, "unknown request type")
		// }

		// resp, err := s.parseError(err)
		// if err != nil {
		// 	// Everything is awful, run it up the stack.
		// 	logger.V(logutil.DEFAULT).Error(err, "Failed to process request", "request", req)
		// 	return err
		// } else if resp != nil {
		// 	// We have an immediate response we can send. Let the gateway know.
		// 	if err := srv.Send(resp); err != nil {
		// 		logger.V(logutil.DEFAULT).Error(err, "Send failed")
		// 		return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
		// 	}
		// }

		// loggerVerbose.Info("Response generated", "response", resp)
		// if err := srv.Send(resp); err != nil {
		// 	logger.V(logutil.DEFAULT).Error(err, "Send failed")
		// 	return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
		// }

		pool, err := s.datastore.PoolGet()
		if err != nil {
			return err
		}
		endpoint := "10.108.10.4" + ":" + strconv.Itoa(int(pool.Spec.TargetPortNumber))

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
		switch v := req.Request.(type) {
		case *extProcPb.ProcessingRequest_RequestHeaders:
			beeps++
			loggerVerbose.Info("BEEP", "beeps", beeps, "boops", boops, "bops", bops)
			headerResp := &extProcPb.ProcessingResponse{
				Response: &extProcPb.ProcessingResponse_RequestHeaders{
					RequestHeaders: &extProcPb.HeadersResponse{
						Response: &extProcPb.CommonResponse{
							HeaderMutation: &extProcPb.HeaderMutation{
								SetHeaders: []*configPb.HeaderValueOption{
									{
										Header: &configPb.HeaderValue{
											Key:      s.destinationEndpointHintKey,
											RawValue: []byte(endpoint),
										},
									},
								},
							},
						},
					},
				},
			}
			if err := srv.Send(headerResp); err != nil {
				logger.V(logutil.DEFAULT).Error(err, "Send failed")
				return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
			}
		case *extProcPb.ProcessingRequest_RequestBody:
			boops++
			loggerVerbose.Info("BOOP", "beeps", beeps, "boops", boops, "bops", bops)
			bodyResp := &extProcPb.ProcessingResponse{
				// The Endpoint Picker supports two approaches to communicating the target endpoint, as a request header
				// and as an unstructure ext-proc response metadata key/value pair. This enables different integration
				// options for gateway providers.
				Response: &extProcPb.ProcessingResponse_RequestBody{
					RequestBody: &extProcPb.BodyResponse{
						Response: &extProcPb.CommonResponse{
							BodyMutation: &extProcPb.BodyMutation{
								Mutation: &extProcPb.BodyMutation_StreamedResponse{
									StreamedResponse: &extProcPb.StreamedBodyResponse{
										Body:        v.RequestBody.Body,
										EndOfStream: v.RequestBody.EndOfStream,
									},
								},
							},
						},
					},
				},
				DynamicMetadata: dynamicMetadata,
			}
			if err := srv.Send(bodyResp); err != nil {
				logger.V(logutil.DEFAULT).Error(err, "Send failed")
				return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
			}
		case *extProcPb.ProcessingRequest_RequestTrailers:
			bops++
			loggerVerbose.Info("BOP", "beeps", beeps, "boops", boops, "bops", bops)
			trailerResp := &extProcPb.ProcessingResponse{
				Response: &extProcPb.ProcessingResponse_RequestTrailers{
					RequestTrailers: &extProcPb.TrailersResponse{},
				},
			}
			if err := srv.Send(trailerResp); err != nil {
				logger.V(logutil.DEFAULT).Error(err, "Send failed")
				return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
			}
		}
	}
}

// func (s *StreamingServer) parseError(err error) (*extProcPb.ProcessingResponse, error) {
// 	if err != nil {
// 		return nil, nil
// 	}

// 	resp := &extProcPb.ProcessingResponse{}
// 	switch errutil.CanonicalCode(err) {
// 	// This code can be returned by scheduler when there is no capacity for sheddable
// 	// requests.
// 	case errutil.InferencePoolResourceExhausted:
// 		resp = &extProcPb.ProcessingResponse{
// 			Response: &extProcPb.ProcessingResponse_ImmediateResponse{
// 				ImmediateResponse: &extProcPb.ImmediateResponse{
// 					Status: &envoyTypePb.HttpStatus{
// 						Code: envoyTypePb.StatusCode_TooManyRequests,
// 					},
// 				},
// 			},
// 		}
// 	// This code can be returned by when EPP processes the request and run into server-side errors.
// 	case errutil.Internal:
// 		resp = &extProcPb.ProcessingResponse{
// 			Response: &extProcPb.ProcessingResponse_ImmediateResponse{
// 				ImmediateResponse: &extProcPb.ImmediateResponse{
// 					Status: &envoyTypePb.HttpStatus{
// 						Code: envoyTypePb.StatusCode_InternalServerError,
// 					},
// 				},
// 			},
// 		}
// 	// This code can be returned when users provide invalid json request.
// 	case errutil.BadRequest:
// 		resp = &extProcPb.ProcessingResponse{
// 			Response: &extProcPb.ProcessingResponse_ImmediateResponse{
// 				ImmediateResponse: &extProcPb.ImmediateResponse{
// 					Status: &envoyTypePb.HttpStatus{
// 						Code: envoyTypePb.StatusCode_BadRequest,
// 					},
// 				},
// 			},
// 		}
// 	case errutil.BadConfiguration:
// 		resp = &extProcPb.ProcessingResponse{
// 			Response: &extProcPb.ProcessingResponse_ImmediateResponse{
// 				ImmediateResponse: &extProcPb.ImmediateResponse{
// 					Status: &envoyTypePb.HttpStatus{
// 						Code: envoyTypePb.StatusCode_NotFound,
// 					},
// 				},
// 			},
// 		}
// 	default:
// 		return nil, status.Errorf(status.Code(err), "failed to handle request: %v", err)
// 	}
// 	return resp, nil
// }

// type requestMutations struct {
// 	requestHeaderResponse extProcPb.ProcessingResponse_RequestHeaders
// 	requestBodyResponse   extProcPb.ProcessingResponse_RequestBody
// 	requestTrailerReponse extProcPb.ProcessingResponse_RequestTrailers
// 	responseToSend        extProcPb.ProcessingResponse
// }
