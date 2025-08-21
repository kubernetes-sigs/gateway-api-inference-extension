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

package slorequest

import (
	"context"
	"math"
	"time"

	"github.com/google/uuid"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/handlers"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/latencypredictorasync"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	scheduling_types "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
)

const (
	SLORequestTrackerPluginType = "slo-request-tracker"
)

type SLORequestTracker struct {
	tn               plugins.TypedName
	latencypredictor latencypredictorasync.PredictorInterface
	datastore        datastore.Datastore
}

var _ requestcontrol.PreRequest = &SLORequestTracker{}
var _ requestcontrol.PostResponse = &SLORequestTracker{}
var _ requestcontrol.PostResponseChunk = &SLORequestTracker{}
var _ requestcontrol.PostResponseComplete = &SLORequestTracker{}

func New(latencypredictor latencypredictorasync.PredictorInterface, datastore datastore.Datastore) *SLORequestTracker {
	return &SLORequestTracker{
		tn:               plugins.TypedName{Type: SLORequestTrackerPluginType, Name: SLORequestTrackerPluginType},
		latencypredictor: latencypredictor,
		datastore:        datastore,
	}
}

func (t *SLORequestTracker) TypedName() plugins.TypedName {
	return t.tn
}

func (s *SLORequestTracker) WithName(name string) *SLORequestTracker {
	s.tn.Name = name
	return s
}

func (t *SLORequestTracker) PreRequest(ctx context.Context, request *scheduling_types.LLMRequest, schedulingResult *scheduling_types.SchedulingResult, targetPort int) {
	logger := log.FromContext(ctx)
	if request.TTFTSLO == 0 || request.AvgTPOTSLO == 0 {
		logger.V(logutil.DEBUG).Info("SLORequestTracker: Skipping PreRequest because no SLOs were provided.")
		return
	}

	if schedulingResult == nil || len(schedulingResult.ProfileResults) == 0 {
		logger.V(logutil.DEBUG).Info("SLORequestTracker: Skipping PreRequest because no scheduling result was provided.")
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

	err := t.datastore.PodAddRequest(podName, request.Headers[requtil.RequestIdHeaderKey], request.AvgTPOTSLO)
	if err != nil {
		logger.V(logutil.DEBUG).Error(err, "SLORequestTracker: Failed to add request to pod running queue", "podName", podName, "requestID", request.Headers[requtil.RequestIdHeaderKey])
	}
}

func (t *SLORequestTracker) PostResponse(ctx context.Context, reqCtx *handlers.RequestContext) {
	logger := log.FromContext(ctx)
	request := reqCtx.SchedulingRequest
	targetPod := reqCtx.TargetPod

	if request.TTFTSLO == 0 || request.AvgTPOTSLO == 0 {
		logger.V(logutil.DEBUG).Info("SLORequestTracker: Skipping PostResponse because no SLOs were provided.")
		return
	}

	if targetPod == nil {
		logger.V(logutil.DEBUG).Info("SLORequestTracker: Skipping PostResponse because no target pod was provided.")
		return
	}

	if t.latencypredictor == nil {
		logger.V(logutil.DEBUG).Info("SLORequestTracker: Skipping PostResponse because no latency predictor in director.")
		return
	}
	if err := requestcontrol.ProcessHeaderForLatencyPrediction(ctx, t.latencypredictor, reqCtx); err != nil {
		logger.V(logutil.DEBUG).Error(err, "ProcessHeader in latencypredictor failed")
	}

}

func (t *SLORequestTracker) PostResponseChunk(ctx context.Context, reqCtx *handlers.RequestContext) {
	logger := log.FromContext(ctx)
	request := reqCtx.SchedulingRequest
	targetPod := reqCtx.TargetPod

	if request.TTFTSLO == 0 || request.AvgTPOTSLO == 0 {
		logger.V(logutil.DEBUG).Info("SLORequestTracker: Skipping PostResponse because no SLOs were provided.")
		return
	}

	if targetPod == nil {
		logger.V(logutil.DEBUG).Info("SLORequestTracker: Skipping PostResponse because no target pod was provided.")
		return
	}

	if t.latencypredictor == nil || reqCtx.SchedulingResult == nil {
		logger.V(logutil.DEBUG).Info("Skipping header prediction; predictor or scheduling missing")
		return
	}

	now := time.Now()

	if reqCtx.TTFT == 0 {
		requestcontrol.ProcessFirstTokenForLatencyPrediction(ctx, t.latencypredictor, reqCtx, now)
	} else {
		requestcontrol.ProcessTokenForLatencyPrediction(ctx, t.latencypredictor, reqCtx, now)
	}

}

func (t *SLORequestTracker) PostResponseComplete(ctx context.Context, reqCtx *handlers.RequestContext) {
	logger := log.FromContext(ctx)
	request := reqCtx.SchedulingRequest
	targetPod := reqCtx.TargetPod
	if request.TTFTSLO == 0 || request.AvgTPOTSLO == 0 {
		logger.V(logutil.DEBUG).Info("SLORequestTracker: Skipping PostResponse because no SLOs were provided.")
		return
	}

	if targetPod == nil {
		logger.V(logutil.DEBUG).Info("SLORequestTracker: Skipping PostResponse because no target pod was provided.")
		return
	}
	if t.latencypredictor == nil {
		logger.V(logutil.DEBUG).Info("Skipping header prediction; predictor or scheduling missing")
		return
	}

	mapeTTFT := 0.0
	if reqCtx.TTFT > 0 {
		mapeTTFT = math.Abs((reqCtx.TTFT-reqCtx.PredictedTTFT)/reqCtx.TTFT) * 100
		logger.V(logutil.DEBUG).Info("Averages calculated", "avgActualTTFT", reqCtx.TTFT, "avgPredictedTTFT", reqCtx.PredictedTTFT)
		logger.V(logutil.DEBUG).Info("MAPE TTFT computed", "mapeTTFT%", mapeTTFT)
		metrics.RecordRequestTTFT(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.TTFT/1000)
		metrics.RecordRequestPredictedTTFT(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.PredictedTTFT/1000)
		metrics.RecordRequestTTFTPredictionMape(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, mapeTTFT)
	}

	mapeTPOT := 0.0
	if reqCtx.AvgTPOT > 0 {
		mapeTPOT = math.Abs((reqCtx.AvgTPOT-reqCtx.AvgPredictedTPOT)/reqCtx.AvgTPOT) * 100
		logger.V(logutil.DEBUG).Info("Averages calculated", "avgActualTPOT", reqCtx.AvgTPOT, "avgPredictedTPOT", reqCtx.AvgPredictedTPOT)
		logger.V(logutil.DEBUG).Info("MAPE TPOT computed", "mapeTPOT%", mapeTPOT)
		metrics.RecordRequestTPOT(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.AvgTPOT/1000)
		metrics.RecordRequestPredictedTPOT(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.AvgPredictedTPOT/1000)
		metrics.RecordRequestTPOTPredictionMape(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, mapeTPOT)
	}

	podName := types.NamespacedName{
		Name:      targetPod.NamespacedName.Name,
		Namespace: targetPod.NamespacedName.Namespace,
	}

	if err := t.datastore.PodRemoveRequest(podName, request.Headers[requtil.RequestIdHeaderKey]); err != nil {
		logger.V(logutil.DEBUG).Error(err, "SLORequestTracker: Failed to remove request from queue", "requestID", request.Headers[requtil.RequestIdHeaderKey])
	}
}

func (t *SLORequestTracker) IsPredictorAvailable() bool {
	return t.latencypredictor != nil
}
