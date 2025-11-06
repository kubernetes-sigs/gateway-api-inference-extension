package slo_aware_router

import (
	"context"
	"fmt"
	"time"

	"github.com/go-logr/logr"
	"github.com/google/uuid"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
)

var _ requestcontrol.PreRequest = &SLOAwareRouter{}
var _ requestcontrol.ResponseReceived = &SLOAwareRouter{}
var _ requestcontrol.ResponseStreaming = &SLOAwareRouter{}
var _ requestcontrol.ResponseComplete = &SLOAwareRouter{}

type SLORequestContext struct {
	SchedulingRequest         schedulingtypes.LLMRequest
	TargetPod                 *backend.Pod
	SchedulingResult          *schedulingtypes.SchedulingResult
	LastSeenMetrics           map[string]*backendmetrics.MetricsState
	LastTokenTimestamp        time.Time
	RequestReceivedTimestamp  time.Time
	GeneratedTokenCount       int
	IncomingModelName         string
	TTFT                      float64
	PredictedTTFT             float64
	AvgTPOT                   float64
	AvgPredictedTPOT          float64
	TokenSampler              *TokenSampler
	TPOTObservations          []float64
	PredictedTPOTObservations []float64

	PrefixCacheScoresForPods map[string]float64

	// TTFTSLO is the target time to first token SLO for the request.
	TTFTSLO float64
	// TPOTSLO is the target time per output token SLO for the request.
	AvgTPOTSLO float64

	// PredictorBasedScheduling indicates whether to use predictor based scheduling.
	PredictorBasedScheduling bool
	//PredictedTTFTForScheduling is the map of pod names to predicted TTFT values for scheduling.
	PredictedTTFTForScheduling map[string]float64
	// PredictedTPOTForScheduling is the map of pod names to predicted TPOT values for scheduling.
	PredictedTPOTForScheduling map[string]float64

	// boolean set if request has valid pod based on predictions
	HasValidPod bool
}

func NewSLORequestContext(request *schedulingtypes.LLMRequest) *SLORequestContext {
	return &SLORequestContext{
		SchedulingRequest:          *request,
		LastSeenMetrics:            make(map[string]*backendmetrics.MetricsState),
		PrefixCacheScoresForPods:   make(map[string]float64),
		PredictedTTFTForScheduling: make(map[string]float64),
		PredictedTPOTForScheduling: make(map[string]float64),
	}
}

func (s *SLOAwareRouter) getSLOContextForRequest(request *schedulingtypes.LLMRequest) (*SLORequestContext, error) {
	id := request.Headers[requtil.RequestIdHeaderKey]
	if ctx, exists := s.sloContextStore.Load(id); exists {
		return ctx.(*SLORequestContext), nil
	}
	return nil, fmt.Errorf("SLO context not found for request ID: %s", id)
}

func (s *SLOAwareRouter) setSLOContextForRequest(request *schedulingtypes.LLMRequest, ctx *SLORequestContext) {
	id := request.Headers[requtil.RequestIdHeaderKey]
	s.sloContextStore.Store(id, ctx)
}

func (s *SLOAwareRouter) deleteSLOContextForRequest(request *schedulingtypes.LLMRequest) {
	id := request.Headers[requtil.RequestIdHeaderKey]
	s.sloContextStore.Delete(id)
}

// --- RequestControl Hooks ---

func (t *SLOAwareRouter) PreRequest(ctx context.Context, request *schedulingtypes.LLMRequest, schedulingResult *schedulingtypes.SchedulingResult) {
	logger := log.FromContext(ctx)

	if schedulingResult == nil || len(schedulingResult.ProfileResults) == 0 {
		logger.V(logutil.DEBUG).Info("SLOAwareRouter: Skipping PreRequest because no scheduling result was provided.")
		return
	}

	targetPod := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName].TargetPods[0].GetPod()

	podName := types.NamespacedName{
		Name:      targetPod.NamespacedName.Name,
		Namespace: targetPod.NamespacedName.Namespace,
	}

	logger.V(logutil.DEBUG).Info("request ID for SLO tracking", "requestID", request.Headers[requtil.RequestIdHeaderKey], "podName", podName)
	if request.Headers[requtil.RequestIdHeaderKey] == "" {
		request.Headers[requtil.RequestIdHeaderKey] = uuid.New().String()
		logger.V(logutil.DEBUG).Info("Generated new request ID for SLO tracking", "requestID", request.Headers[requtil.RequestIdHeaderKey])
		logger.V(logutil.DEBUG).Info("request headers for SLO tracking", "requestHeaders", request.Headers)
	}

	id := request.Headers[requtil.RequestIdHeaderKey]
	podRequestList, ok := t.runningRequestLists[podName]
	if !ok {
		podRequestList = NewRequestPriorityQueue()
		t.runningRequestLists[podName] = podRequestList
	}

	sloCtx, err := t.getSLOContextForRequest(request)
	if err != nil {
		id := request.Headers[requtil.RequestIdHeaderKey]
		logger.V(logutil.DEBUG).Error(err, "SLOAwareRouter.PreRequest: Failed to get SLO context for request", "requestID", id)
		return
	}

	added := podRequestList.Add(id, sloCtx.AvgTPOTSLO)
	if !added {
		logger.V(logutil.DEBUG).Info("SLOAwareRouter: Item already exists in queue", "podName", podName, "requestID", id)
	}

	// Set up SLO request context
	sloCtx.TargetPod = targetPod
	sloCtx.SchedulingResult = schedulingResult
	sloCtx.RequestReceivedTimestamp = time.Now()
	RefreshLastSeenMetrics(ctx, sloCtx)
	t.setSLOContextForRequest(request, sloCtx)
}

func (t *SLOAwareRouter) ResponseReceived(ctx context.Context, request *schedulingtypes.LLMRequest, response *requestcontrol.Response, targetPod *backend.Pod) {
	logger := log.FromContext(ctx)
	id := request.Headers[requtil.RequestIdHeaderKey]

	sloCtx, err := t.getSLOContextForRequest(request)
	if err != nil {
		logger.V(logutil.DEBUG).Error(err, "SLOAwareRouter: Failed to get SLO context for request", "requestID", id)
		return
	}

	if !t.CheckPredictor(logger, targetPod) {
		return
	}

	if err := ProcessHeaderForLatencyPrediction(ctx, t.latencypredictor, sloCtx); err != nil {
		logger.V(logutil.DEBUG).Error(err, "ProcessHeader in latencypredictor failed")
	}

}

func (t *SLOAwareRouter) ResponseStreaming(ctx context.Context, request *schedulingtypes.LLMRequest, response *requestcontrol.Response, pod *backend.Pod) {
	logger := log.FromContext(ctx)
	if !t.CheckPredictor(logger, pod) {
		return
	}

	now := time.Now()
	sloCtx, err := t.getSLOContextForRequest(request)
	if err != nil {
		id := request.Headers[requtil.RequestIdHeaderKey]
		logger.V(logutil.DEBUG).Error(err, "SLOAwareRouter.ResponseStreaming: Failed to get SLO context for request", "requestID", id)
		return
	}

	if sloCtx.TTFT == 0 {
		ProcessFirstTokenForLatencyPrediction(ctx, t.latencypredictor, sloCtx, now)
	} else {
		ProcessTokenForLatencyPrediction(ctx, t.latencypredictor, sloCtx, now)
	}

}

func (t *SLOAwareRouter) ResponseComplete(ctx context.Context, request *schedulingtypes.LLMRequest, response *requestcontrol.Response, pod *backend.Pod) {
	logger := log.FromContext(ctx)
	targetPod := pod
	if !t.CheckPredictor(logger, targetPod) {
		return
	}

	sloCtx, err := t.getSLOContextForRequest(request)
	if err != nil {
		id := request.Headers[requtil.RequestIdHeaderKey]
		logger.V(logutil.DEBUG).Error(err, "SLOAwareRouter.ResponseComplete: Failed to get SLO context for request", "requestID", id)
		return
	}

	if sloCtx.TTFT > 0 {
		logger.V(logutil.DEBUG).Info("Averages calculated", "avgActualTTFT", sloCtx.TTFT, "avgPredictedTTFT", sloCtx.PredictedTTFT)
		metrics.RecordRequestTTFT(ctx, sloCtx.IncomingModelName, request.TargetModel, sloCtx.TTFT/1000)
		metrics.RecordRequestPredictedTTFT(ctx, sloCtx.IncomingModelName, request.TargetModel, sloCtx.PredictedTTFT/1000)
		if sloCtx.TTFTSLO > 0 {
			metrics.RecordRequestTTFTWithSLO(ctx, sloCtx.IncomingModelName, request.TargetModel, sloCtx.TTFT, sloCtx.TTFTSLO)
		}
	}

	if sloCtx.AvgTPOT > 0 {
		logger.V(logutil.DEBUG).Info("Averages calculated", "avgActualTPOT", sloCtx.AvgTPOT, "avgPredictedTPOT", sloCtx.AvgPredictedTPOT)
		metrics.RecordRequestTPOT(ctx, sloCtx.IncomingModelName, request.TargetModel, sloCtx.AvgTPOT/1000)
		metrics.RecordRequestPredictedTPOT(ctx, sloCtx.IncomingModelName, request.TargetModel, sloCtx.AvgPredictedTPOT/1000)
		if sloCtx.AvgTPOTSLO > 0 {
			metrics.RecordRequestTPOTWithSLO(ctx, sloCtx.IncomingModelName, request.TargetModel, sloCtx.AvgTPOT, sloCtx.AvgTPOTSLO)
		}
	}

	logger.V(logutil.DEBUG).Info("SLO Aware Routing Mode", "PredictorBasedScheduling", sloCtx.PredictorBasedScheduling)

	podName := types.NamespacedName{
		Name:      targetPod.NamespacedName.Name,
		Namespace: targetPod.NamespacedName.Namespace,
	}

	id := request.Headers[requtil.RequestIdHeaderKey]
	podRequestList, ok := t.runningRequestLists[podName]
	if !ok {
		err := fmt.Errorf("no running request list found for pod %s", podName.String())
		logger.V(logutil.DEBUG).Error(err, "SLOAwareRouter: Failed to remove request from queue", "requestID", id)
	}

	_, removed := podRequestList.Remove(id)
	if !removed {
		logger.V(logutil.DEBUG).Info("SLOAwareRouter: Item not found in queue", "podName", podName, "requestID", id)
	}
	t.deleteSLOContextForRequest(request)
}

func (t *SLOAwareRouter) CheckPredictor(logger logr.Logger, targetPod *backend.Pod) bool {
	if targetPod == nil {
		logger.V(logutil.DEBUG).Info("SLOAwareRouter: Skipping PostResponse because no target pod was provided.")
		return false
	}
	if t.latencypredictor == nil {
		logger.V(logutil.DEBUG).Info("SLOAwareRouter: Skipping PostResponse because predictor missing")
		return false
	}
	return true
}
