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

package slo_aware_router

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/go-logr/logr"
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

type sloRequestContext struct {
	schedulingRequest         schedulingtypes.LLMRequest
	targetPod                 *backend.Pod
	schedulingResult          *schedulingtypes.SchedulingResult
	lastSeenMetrics           map[string]*backendmetrics.MetricsState
	lastTokenTimestamp        time.Time
	requestReceivedTimestamp  time.Time
	generatedTokenCount       int
	incomingModelName         string
	ttft                      float64
	predictedTTFT             float64
	avgTPOT                   float64
	avgPredictedTPOT          float64
	tokenSampler              *tokenSampler
	tpotObservations          []float64
	predictedTPOTObservations []float64

	prefixCacheScoresForPods map[string]float64

	// ttftSLO is the target time to first token SLO for the request.
	ttftSLO float64
	// TPOTSLO is the target time per output token SLO for the request.
	avgTPOTSLO float64

	// predictorBasedScheduling indicates whether to use predictor based scheduling.
	predictorBasedScheduling bool
	// predictedTTFTForScheduling is the map of pod names to predicted TTFT values for scheduling.
	predictedTTFTForScheduling map[string]float64
	// predictedTPOTForScheduling is the map of pod names to predicted TPOT values for scheduling.
	predictedTPOTForScheduling map[string]float64

	// boolean set if request has valid pod based on predictions
	hasValidPod bool
}

func newSLORequestContext(request *schedulingtypes.LLMRequest) *sloRequestContext {
	return &sloRequestContext{
		schedulingRequest:          *request,
		lastSeenMetrics:            make(map[string]*backendmetrics.MetricsState),
		prefixCacheScoresForPods:   make(map[string]float64),
		predictedTTFTForScheduling: make(map[string]float64),
		predictedTPOTForScheduling: make(map[string]float64),
	}
}

func (s *SLOAwareRouter) getSLOContextForRequest(request *schedulingtypes.LLMRequest) (*sloRequestContext, error) {
	id := request.Headers[requtil.RequestIdHeaderKey]
	if ctx, exists := s.sloContextStore.Load(id); exists {
		return ctx.(*sloRequestContext), nil
	}
	return nil, fmt.Errorf("SLO context not found for request ID: %s", id)
}

func (s *SLOAwareRouter) setSLOContextForRequest(request *schedulingtypes.LLMRequest, ctx *sloRequestContext) {
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
	if request == nil {
		logger.V(logutil.DEBUG).Info("SLOAwareRouter.PreRequest: request is nil, skipping")
		return
	}

	if schedulingResult == nil || len(schedulingResult.ProfileResults) == 0 {
		logger.V(logutil.TRACE).Info("SLOAwareRouter: Skipping PreRequest because no scheduling result was provided.")
		return
	}

	targetPod := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName].TargetPods[0].GetPod()
	if !t.checkPredictor(logger, targetPod) {
		return
	}

	podName := types.NamespacedName{
		Name:      targetPod.NamespacedName.Name,
		Namespace: targetPod.NamespacedName.Namespace,
	}

	logger.V(logutil.TRACE).Info("request ID for SLO tracking", "requestID", request.Headers[requtil.RequestIdHeaderKey], "podName", podName)
	if request.Headers[requtil.RequestIdHeaderKey] == "" {
		logger.V(logutil.DEBUG).Error(errors.New("missing request ID"), "SLOAwareRouter.PreRequest: Request is missing request ID header")
	}

	id := request.Headers[requtil.RequestIdHeaderKey]
	podRequestList, ok := t.runningRequestLists[podName]
	if !ok {
		podRequestList = newRequestPriorityQueue()
		t.runningRequestLists[podName] = podRequestList
	}

	sloCtx, err := t.getSLOContextForRequest(request)
	if err != nil {
		id := request.Headers[requtil.RequestIdHeaderKey]
		logger.V(logutil.DEBUG).Error(err, "SLOAwareRouter.PreRequest: Failed to get SLO context for request", "requestID", id)
		return
	}

	added := podRequestList.Add(id, sloCtx.avgTPOTSLO)
	if !added {
		logger.V(logutil.TRACE).Info("SLOAwareRouter: Item already exists in queue", "podName", podName, "requestID", id)
	}

	// Set up SLO request context
	sloCtx.targetPod = targetPod
	sloCtx.schedulingResult = schedulingResult
	sloCtx.requestReceivedTimestamp = time.Now()
	refreshLastSeenMetrics(ctx, sloCtx)
	t.setSLOContextForRequest(request, sloCtx)
}

func (t *SLOAwareRouter) ResponseReceived(ctx context.Context, request *schedulingtypes.LLMRequest, response *requestcontrol.Response, targetPod *backend.Pod) {
	logger := log.FromContext(ctx)
	if request == nil {
		logger.V(logutil.DEBUG).Info("SLOAwareRouter.ResponseReceived: request is nil, skipping")
		return
	}
	if !t.checkPredictor(logger, targetPod) {
		return
	}

	id := request.Headers[requtil.RequestIdHeaderKey]

	sloCtx, err := t.getSLOContextForRequest(request)
	if err != nil {
		logger.V(logutil.DEBUG).Error(err, "SLOAwareRouter: Failed to get SLO context for request", "requestID", id)
		return
	}

	if err := processHeaderForLatencyPrediction(ctx, t.latencypredictor, sloCtx); err != nil {
		logger.V(logutil.DEBUG).Error(err, "ProcessHeader in latencypredictor failed")
	}

}

func (t *SLOAwareRouter) ResponseStreaming(ctx context.Context, request *schedulingtypes.LLMRequest, response *requestcontrol.Response, pod *backend.Pod) {
	logger := log.FromContext(ctx)
	if request == nil {
		logger.V(logutil.DEBUG).Info("SLOAwareRouter.ResponseStreaming: request is nil, skipping")
		return
	}
	if !t.checkPredictor(logger, pod) || response.EndOfStream {
		return
	}

	now := time.Now()
	sloCtx, err := t.getSLOContextForRequest(request)
	if err != nil {
		id := request.Headers[requtil.RequestIdHeaderKey]
		logger.V(logutil.TRACE).Error(err, "SLOAwareRouter.ResponseStreaming: Failed to get SLO context for request", "requestID", id)
		return
	}

	if sloCtx.ttft == 0 {
		processFirstTokenForLatencyPrediction(ctx, t.latencypredictor, sloCtx, now)
	} else {
		processTokenForLatencyPrediction(ctx, t.latencypredictor, sloCtx, now)
	}

}

func (t *SLOAwareRouter) ResponseComplete(ctx context.Context, request *schedulingtypes.LLMRequest, response *requestcontrol.Response, pod *backend.Pod) {
	logger := log.FromContext(ctx)
	if request == nil {
		logger.V(logutil.DEBUG).Info("SLOAwareRouter.ResponseComplete: request is nil, skipping")
		return
	}
	targetPod := pod
	if !t.checkPredictor(logger, targetPod) {
		return
	}

	sloCtx, err := t.getSLOContextForRequest(request)
	if err != nil {
		id := request.Headers[requtil.RequestIdHeaderKey]
		logger.V(logutil.DEBUG).Error(err, "SLOAwareRouter.ResponseComplete: Failed to get SLO context for request", "requestID", id)
		return
	}

	if sloCtx.ttft > 0 {
		logger.V(logutil.TRACE).Info("Averages calculated", "avgActualTTFT", sloCtx.ttft, "avgPredictedTTFT", sloCtx.predictedTTFT)
		metrics.RecordRequestTTFT(ctx, sloCtx.incomingModelName, request.TargetModel, sloCtx.ttft/1000)
		metrics.RecordRequestPredictedTTFT(ctx, sloCtx.incomingModelName, request.TargetModel, sloCtx.predictedTTFT/1000)
		if sloCtx.ttftSLO > 0 {
			metrics.RecordRequestTTFTWithSLO(ctx, sloCtx.incomingModelName, request.TargetModel, sloCtx.ttft, sloCtx.ttftSLO)
		}
	}

	if sloCtx.avgTPOT > 0 {
		logger.V(logutil.TRACE).Info("Averages calculated", "avgActualTPOT", sloCtx.avgTPOT, "avgPredictedTPOT", sloCtx.avgPredictedTPOT)
		metrics.RecordRequestTPOT(ctx, sloCtx.incomingModelName, request.TargetModel, sloCtx.avgTPOT/1000)
		metrics.RecordRequestPredictedTPOT(ctx, sloCtx.incomingModelName, request.TargetModel, sloCtx.avgPredictedTPOT/1000)
		if sloCtx.avgTPOTSLO > 0 {
			metrics.RecordRequestTPOTWithSLO(ctx, sloCtx.incomingModelName, request.TargetModel, sloCtx.avgTPOT, sloCtx.avgTPOTSLO)
		}
	}

	logger.V(logutil.TRACE).Info("SLO Aware Routing Mode", "PredictorBasedScheduling", sloCtx.predictorBasedScheduling)

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
		logger.V(logutil.TRACE).Info("SLOAwareRouter: Item not found in queue", "podName", podName, "requestID", id)
	}
	t.deleteSLOContextForRequest(request)
}

func (t *SLOAwareRouter) checkPredictor(logger logr.Logger, targetPod *backend.Pod) bool {
	if targetPod == nil {
		logger.V(logutil.TRACE).Info("SLOAwareRouter: Skipping hook because no target pod was provided.")
		return false
	}
	if t.latencypredictor == nil {
		logger.V(logutil.TRACE).Info("SLOAwareRouter: Skipping hook because predictor missing")
		return false
	}
	return true
}
