package handlers

import (
	"io"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/api/v1alpha1"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/backend"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/metrics"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/scheduling"
	klog "k8s.io/klog/v2"
)

func NewServer(pp PodProvider, scheduler Scheduler, targetPodHeader string, datastore ModelDataStore) *Server {
	return &Server{
		scheduler:       scheduler,
		podProvider:     pp,
		targetPodHeader: targetPodHeader,
		datastore:       datastore,
	}
}

// Server implements the Envoy external processing server.
// https://www.envoyproxy.io/docs/envoy/latest/api-v3/service/ext_proc/v3/external_processor.proto
type Server struct {
	scheduler   Scheduler
	podProvider PodProvider
	// The key of the header to specify the target pod address. This value needs to match Envoy
	// configuration.
	targetPodHeader string
	datastore       ModelDataStore
}

type Scheduler interface {
	Schedule(b *scheduling.LLMRequest) (targetPod backend.Pod, err error)
}

// PodProvider is an interface to provide set of pods in the backend and information such as metrics.
type PodProvider interface {
	GetPodMetrics(pod backend.Pod) (*backend.PodMetrics, bool)
	UpdatePodMetrics(pod backend.Pod, pm *backend.PodMetrics)
}

type ModelDataStore interface {
	FetchModelData(modelName string) (returnModel *v1alpha1.InferenceModel)
}

func (s *Server) Process(srv extProcPb.ExternalProcessor_ProcessServer) error {
	klog.V(3).Info("Processing")
	ctx := srv.Context()
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
		if err == io.EOF {
			return nil
		}
		if err != nil {
			// This error occurs very frequently, though it doesn't seem to have any impact.
			// TODO Figure out if we can remove this noise.
			klog.V(3).Infof("cannot receive stream request: %v", err)
			return status.Errorf(codes.Unknown, "cannot receive stream request: %v", err)
		}

		var resp *extProcPb.ProcessingResponse
		switch v := req.Request.(type) {
		case *extProcPb.ProcessingRequest_RequestHeaders:
			reqCtx.RequestReceivedTimestamp = time.Now()
			resp = HandleRequestHeaders(reqCtx, req)
			klog.V(3).Infof("Request context after HandleRequestHeaders: %+v", reqCtx)
		case *extProcPb.ProcessingRequest_RequestBody:
			resp, err = s.HandleRequestBody(reqCtx, req)
			if err == nil {
				metrics.RecordRequestCounter(reqCtx.Model, reqCtx.ResolvedTargetModel)
				metrics.RecordRequestSizes(reqCtx.Model, reqCtx.ResolvedTargetModel, reqCtx.RequestSize)
			}
			klog.V(3).Infof("Request context after HandleRequestBody: %+v", reqCtx)
		case *extProcPb.ProcessingRequest_ResponseHeaders:
			resp, err = s.HandleResponseHeaders(reqCtx, req)
			klog.V(3).Infof("Request context after HandleResponseHeaders: %+v", reqCtx)
		case *extProcPb.ProcessingRequest_ResponseBody:
			resp, err = s.HandleResponseBody(reqCtx, req)
			if err == nil && reqCtx.ResponseComplete {
				reqCtx.ResponseCompleteTimestamp = time.Now()
				metrics.RecordRequestLatencies(reqCtx.Model, reqCtx.ResolvedTargetModel, reqCtx.RequestReceivedTimestamp, reqCtx.ResponseCompleteTimestamp)
			}
			klog.V(3).Infof("Request context after HandleResponseBody: %+v", reqCtx)
		default:
			klog.Errorf("Unknown Request type %+v", v)
			return status.Error(codes.Unknown, "unknown request type")
		}
		if err != nil {
			klog.Errorf("failed to process request: %v", err)
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

		klog.V(3).Infof("response: %v", resp)
		if err := srv.Send(resp); err != nil {
			klog.Errorf("send error %v", err)
			return status.Errorf(codes.Unknown, "failed to send response back to Envoy: %v", err)
		}
	}
}

// RequestContext stores context information during the life time of an HTTP request.
type RequestContext struct {
	TargetPod                 backend.Pod
	Model                     string
	ResolvedTargetModel       string
	RequestReceivedTimestamp  time.Time
	ResponseCompleteTimestamp time.Time
	RequestSize               int
	Response                  Response
	ResponseComplete          bool
}
