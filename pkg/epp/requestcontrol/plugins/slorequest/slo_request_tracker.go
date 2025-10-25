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
	"time"

	"github.com/go-logr/logr"
	"github.com/google/uuid"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
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
	targetPod := reqCtx.TargetPod
	if !t.CheckPredictor(logger, targetPod) {
		return
	}

	if err := requestcontrol.ProcessHeaderForLatencyPrediction(ctx, t.latencypredictor, reqCtx); err != nil {
		logger.V(logutil.DEBUG).Error(err, "ProcessHeader in latencypredictor failed")
	}

}

func (t *SLORequestTracker) PostResponseChunk(ctx context.Context, reqCtx *handlers.RequestContext) {
	logger := log.FromContext(ctx)
	targetPod := reqCtx.TargetPod
	if !t.CheckPredictor(logger, targetPod) {
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
	if !t.CheckPredictor(logger, targetPod) {
		return
	}

	if reqCtx.TTFT > 0 {
		logger.V(logutil.DEBUG).Info("Averages calculated", "avgActualTTFT", reqCtx.TTFT, "avgPredictedTTFT", reqCtx.PredictedTTFT)
		metrics.RecordRequestTTFT(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.TTFT/1000)
		metrics.RecordRequestPredictedTTFT(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.PredictedTTFT/1000)
		if reqCtx.SchedulingRequest.TTFTSLO > 0 {
			metrics.RecordRequestTTFTWithSLO(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.TTFT, reqCtx.SchedulingRequest.TTFTSLO)
		}
	}

	if reqCtx.AvgTPOT > 0 {
		logger.V(logutil.DEBUG).Info("Averages calculated", "avgActualTPOT", reqCtx.AvgTPOT, "avgPredictedTPOT", reqCtx.AvgPredictedTPOT)
		metrics.RecordRequestTPOT(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.AvgTPOT/1000)
		metrics.RecordRequestPredictedTPOT(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.AvgPredictedTPOT/1000)
		if reqCtx.SchedulingRequest.AvgTPOTSLO > 0 {
			metrics.RecordRequestTPOTWithSLO(ctx, reqCtx.IncomingModelName, reqCtx.TargetModelName, reqCtx.AvgTPOT, reqCtx.SchedulingRequest.AvgTPOTSLO)
		}
	}
	logger.V(logutil.DEBUG).Info("SLO Aware Routing Mode", "PredictorBasedScheduling", request.PredictorBasedScheduling)

	podName := types.NamespacedName{
		Name:      targetPod.NamespacedName.Name,
		Namespace: targetPod.NamespacedName.Namespace,
	}

	if err := t.datastore.PodRemoveRequest(podName, request.Headers[requtil.RequestIdHeaderKey]); err != nil {
		logger.V(logutil.DEBUG).Error(err, "SLORequestTracker: Failed to remove request from queue", "requestID", request.Headers[requtil.RequestIdHeaderKey])
	}
}

func (t *SLORequestTracker) CheckPredictor(logger logr.Logger, targetPod *backend.Pod) bool {
	if targetPod == nil {
		logger.V(logutil.DEBUG).Info("SLORequestTracker: Skipping PostResponse because no target pod was provided.")
		return false
	}
	if t.latencypredictor == nil {
		logger.V(logutil.DEBUG).Info("SLORequestTracker: Skipping PostResponse because predictor missing")
		return false
	}
	return true
}
