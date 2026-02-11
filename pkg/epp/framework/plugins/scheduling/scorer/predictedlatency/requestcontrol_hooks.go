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

package predictedlatency

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/go-logr/logr"
	"github.com/jellydator/ttlcache/v3"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/sidecars/latencypredictorasync"
)

const (
	// Experimental_DefaultPrefillProfile is the default profile name for prefill pods in disaggregated serving.
	// This constant identifies prefill endpoints for load tracking and training data collection.
	// This is hardcoded for now until we land on a canonical approach for plugins to identify
	// prefill and decode endpoints (See https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/2080)
	Experimental_DefaultPrefillProfile = "prefill"
)

var _ requestcontrol.PreRequest = &PredictedLatency{}
var _ requestcontrol.ResponseReceived = &PredictedLatency{}
var _ requestcontrol.ResponseStreaming = &PredictedLatency{}
var _ requestcontrol.ResponseComplete = &PredictedLatency{}

type predictedLatencyCtx struct {
	schedulingRequest         schedulingtypes.LLMRequest
	targetMetadata            *fwkdl.EndpointMetadata
	schedulingResult          *schedulingtypes.SchedulingResult
	lastSeenMetrics           map[string]*fwkdl.Metrics
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

	prefixCacheScoresForEndpoints map[string]float64

	// ttftSLO is the target time to first token SLO for the request.
	ttftSLO float64
	// TPOTSLO is the target time per output token SLO for the request.
	avgTPOTSLO float64

	// predictedTTFTForScheduling is the map of pod names to predicted TTFT values for scheduling.
	predictionsForScheduling []endpointPredictionResult

	// boolean set if request has valid endpoint based on predictions
	hasValidEndpoint bool
}

func newPredictedLatencyContext(request *schedulingtypes.LLMRequest) *predictedLatencyCtx {
	return &predictedLatencyCtx{
		schedulingRequest:             *request,
		lastSeenMetrics:               make(map[string]*fwkdl.Metrics),
		prefixCacheScoresForEndpoints: make(map[string]float64),
		predictionsForScheduling:      make([]endpointPredictionResult, 0),
		hasValidEndpoint:              true,
	}
}

func (s *PredictedLatency) getPredictedLatencyContextForRequest(request *schedulingtypes.LLMRequest) (*predictedLatencyCtx, error) {
	id := request.Headers[requtil.RequestIdHeaderKey]
	if item := s.sloContextStore.Get(id); item != nil {
		return item.Value(), nil
	}
	return nil, fmt.Errorf("SLO context not found for request ID: %s", id)
}

func (s *PredictedLatency) setPredictedLatencyContextForRequest(request *schedulingtypes.LLMRequest, ctx *predictedLatencyCtx) {
	id := request.Headers[requtil.RequestIdHeaderKey]
	s.sloContextStore.Set(id, ctx, ttlcache.DefaultTTL)
}

func (s *PredictedLatency) deletePredictedLatencyContextForRequest(request *schedulingtypes.LLMRequest) {
	id := request.Headers[requtil.RequestIdHeaderKey]
	s.sloContextStore.Delete(id)
}

// --- RequestControl Hooks ---

func (t *PredictedLatency) PreRequest(ctx context.Context, request *schedulingtypes.LLMRequest, schedulingResult *schedulingtypes.SchedulingResult) {
	logger := log.FromContext(ctx)
	if request == nil {
		logger.V(logutil.DEBUG).Info("PredictedLatency.PreRequest: request is nil, skipping")
		return
	}

	if schedulingResult == nil || len(schedulingResult.ProfileResults) == 0 {
		logger.V(logutil.TRACE).Info("PredictedLatency: Skipping PreRequest because no scheduling result was provided.")
		return
	}

	targetMetadata := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName].TargetEndpoints[0].GetMetadata()
	if !t.checkPredictor(logger, targetMetadata) {
		return
	}

	endpointName := types.NamespacedName{
		Name:      targetMetadata.NamespacedName.Name,
		Namespace: targetMetadata.NamespacedName.Namespace,
	}

	logger.V(logutil.TRACE).Info("request ID for SLO tracking", "requestID", request.Headers[requtil.RequestIdHeaderKey], "endpointName", endpointName)
	if request.Headers[requtil.RequestIdHeaderKey] == "" {
		logger.V(logutil.DEBUG).Error(errors.New("missing request ID"), "PredictedLatency.PreRequest: Request is missing request ID header")
		return
	}

	id := request.Headers[requtil.RequestIdHeaderKey]

	// Get or create queue for this endpoint using sync.Map
	actual, _ := t.runningRequestLists.LoadOrStore(endpointName, newRequestPriorityQueue())
	endpointRequestList := actual.(*requestPriorityQueue)

	predictedLatencyCtx, err := t.getPredictedLatencyContextForRequest(request)
	if err != nil {
		id := request.Headers[requtil.RequestIdHeaderKey]
		logger.V(logutil.DEBUG).Error(err, "PredictedLatency.PreRequest: Failed to get SLO context for request", "requestID", id)
		return
	}

	added := endpointRequestList.Add(id, predictedLatencyCtx.avgTPOTSLO)
	if !added {
		logger.V(logutil.TRACE).Info("PredictedLatency: Item already exists in queue", "endpointName", endpointName, "requestID", id)
	}

	// For disaggregated serving, also track the prefill pod if it was selected
	if prefillResult, exists := schedulingResult.ProfileResults[Experimental_DefaultPrefillProfile]; exists && prefillResult != nil {
		if len(prefillResult.TargetEndpoints) > 0 {
			prefillPod := prefillResult.TargetEndpoints[0].GetMetadata()
			prefillEndpointName := types.NamespacedName{
				Name:      prefillPod.NamespacedName.Name,
				Namespace: prefillPod.NamespacedName.Namespace,
			}

			// Get or create queue for prefill endpoint
			prefillActual, _ := t.runningRequestLists.LoadOrStore(prefillEndpointName, newRequestPriorityQueue())
			prefillRequestList := prefillActual.(*requestPriorityQueue)

			prefillAdded := prefillRequestList.Add(id, predictedLatencyCtx.avgTPOTSLO)
			if !prefillAdded {
				logger.V(logutil.TRACE).Info("PredictedLatency: Prefill item already exists in queue", "endpointName", prefillEndpointName, "requestID", id)
			} else {
				logger.V(logutil.DEBUG).Info("Tracked prefill pod in running requests",
					"prefillPod", prefillPod.NamespacedName.Name,
					"requestID", id)
			}
		}
	}

	// Set up SLO request context
	predictedLatencyCtx.targetMetadata = targetMetadata
	predictedLatencyCtx.schedulingResult = schedulingResult
	predictedLatencyCtx.requestReceivedTimestamp = time.Now()
	refreshLastSeenMetrics(ctx, predictedLatencyCtx)
	t.setPredictedLatencyContextForRequest(request, predictedLatencyCtx)

	if err := processPreRequestForLatencyPrediction(ctx, t.latencypredictor, t.config.EndpointRoleLabel, predictedLatencyCtx); err != nil {
		logger.V(logutil.DEBUG).Error(err, "Process PreRequest in latencypredictor failed")
	}
}

func (t *PredictedLatency) ResponseReceived(ctx context.Context, request *schedulingtypes.LLMRequest, response *requestcontrol.Response, targetMetadata *fwkdl.EndpointMetadata) {
	logger := log.FromContext(ctx)
	if request == nil {
		logger.V(logutil.DEBUG).Info("PredictedLatency.ResponseReceived: request is nil, skipping")
		return
	}

	predictedLatencyCtx, err := t.getPredictedLatencyContextForRequest(request)
	if err != nil {
		id := request.Headers[requtil.RequestIdHeaderKey]
		logger.V(logutil.DEBUG).Error(err, "PredictedLatency.ResponseReceived: Failed to get context for request", "requestID", id)
		return
	}

	schedulingResult := predictedLatencyCtx.schedulingResult
	if schedulingResult == nil {
		return
	}

	if prefillResult, exists := schedulingResult.ProfileResults[Experimental_DefaultPrefillProfile]; exists && prefillResult != nil {
		if len(prefillResult.TargetEndpoints) > 0 {
			now := time.Now()
			prefillTTFT := float64(now.Sub(predictedLatencyCtx.requestReceivedTimestamp).Milliseconds())

			prefillEndpoint := prefillResult.TargetEndpoints[0]
			prefillMetadata := prefillEndpoint.GetMetadata()

			prefillMetrics, exists := predictedLatencyCtx.lastSeenMetrics[Experimental_DefaultPrefillProfile]
			if !exists || prefillMetrics == nil {
				logger.V(logutil.DEBUG).Info("No metrics available for prefill profile, skipping training")
				return
			}

			prefixCacheScore := predictedLatencyCtx.prefixCacheScoresForEndpoints[prefillMetadata.NamespacedName.Name]

			logger.V(logutil.DEBUG).Info("Recording prefill training data",
				"requestID", request.Headers[requtil.RequestIdHeaderKey],
				"prefillTTFT_ms", prefillTTFT,
				"prefillPod", prefillMetadata.NamespacedName.Name,
				"prefixCacheScore", prefixCacheScore)

			entry := buildTrainingEntry(
				t.config.EndpointRoleLabel,
				prefillMetadata,
				prefillMetrics,
				predictedLatencyCtx.schedulingRequest.Body.Completions.Prompt,
				prefillTTFT,
				0,
				now,
				0,
				prefixCacheScore,
			)

			if err := t.latencypredictor.AddTrainingDataBulk([]latencypredictor.TrainingEntry{entry}); err != nil {
				logger.V(logutil.DEBUG).Error(err, "Failed to record prefill training data")
			}
		}
	}
}

// --- Response Hooks when body chunks received---
func (t *PredictedLatency) ResponseStreaming(ctx context.Context, request *schedulingtypes.LLMRequest, response *requestcontrol.Response, targetMetadata *fwkdl.EndpointMetadata) {
	logger := log.FromContext(ctx)
	if request == nil {
		logger.V(logutil.DEBUG).Info("PredictedLatency.ResponseStreaming: request is nil, skipping")
		return
	}
	if !t.checkPredictor(logger, targetMetadata) || response.EndOfStream || !t.config.StreamingMode {
		return
	}

	now := time.Now()
	predictedLatencyCtx, err := t.getPredictedLatencyContextForRequest(request)
	if err != nil {
		id := request.Headers[requtil.RequestIdHeaderKey]
		logger.V(logutil.TRACE).Error(err, "PredictedLatency.ResponseStreaming: Failed to get SLO context for request", "requestID", id)
		return
	}

	if predictedLatencyCtx.ttft == 0 {
		processFirstTokenForLatencyPrediction(ctx, t.latencypredictor, t.config.StreamingMode, t.config.EndpointRoleLabel, predictedLatencyCtx, now, t.config.SamplingMean, t.config.MaxSampledTokens)
	} else {
		processTokenForLatencyPrediction(ctx, t.latencypredictor, t.config.EndpointRoleLabel, predictedLatencyCtx, targetMetadata, now, t.config.SamplingMean, t.config.MaxSampledTokens)
	}

}

func (t *PredictedLatency) ResponseComplete(ctx context.Context, request *schedulingtypes.LLMRequest, response *requestcontrol.Response, metadata *fwkdl.EndpointMetadata) {
	logger := log.FromContext(ctx)
	if request == nil {
		logger.V(logutil.DEBUG).Info("PredictedLatency.ResponseComplete: request is nil, skipping")
		return
	}
	targetMetadata := metadata
	if !t.checkPredictor(logger, targetMetadata) {
		return
	}

	predictedLatencyCtx, err := t.getPredictedLatencyContextForRequest(request)
	if err != nil {
		id := request.Headers[requtil.RequestIdHeaderKey]
		logger.V(logutil.DEBUG).Error(err, "PredictedLatency.ResponseComplete: Failed to get SLO context for request", "requestID", id)
		return
	}
	now := time.Now()
	if !t.config.StreamingMode {
		processFirstTokenForLatencyPrediction(ctx, t.latencypredictor, t.config.StreamingMode, t.config.EndpointRoleLabel, predictedLatencyCtx, now, t.config.SamplingMean, t.config.MaxSampledTokens)
	}

	if predictedLatencyCtx.ttft > 0 {
		logger.V(logutil.TRACE).Info("Averages calculated", "avgActualTTFT", predictedLatencyCtx.ttft, "avgPredictedTTFT", predictedLatencyCtx.predictedTTFT)
		metrics.RecordRequestTTFT(ctx, predictedLatencyCtx.incomingModelName, request.TargetModel, predictedLatencyCtx.ttft/1000)
		metrics.RecordRequestPredictedTTFT(ctx, predictedLatencyCtx.incomingModelName, request.TargetModel, predictedLatencyCtx.predictedTTFT/1000)
		if predictedLatencyCtx.ttftSLO > 0 {
			metrics.RecordRequestTTFTWithSLO(ctx, predictedLatencyCtx.incomingModelName, request.TargetModel, predictedLatencyCtx.ttft, predictedLatencyCtx.ttftSLO)
		}
	}

	if predictedLatencyCtx.avgTPOT > 0 {
		logger.V(logutil.TRACE).Info("Averages calculated", "avgActualTPOT", predictedLatencyCtx.avgTPOT, "avgPredictedTPOT", predictedLatencyCtx.avgPredictedTPOT)
		metrics.RecordRequestTPOT(ctx, predictedLatencyCtx.incomingModelName, request.TargetModel, predictedLatencyCtx.avgTPOT/1000)
		metrics.RecordRequestPredictedTPOT(ctx, predictedLatencyCtx.incomingModelName, request.TargetModel, predictedLatencyCtx.avgPredictedTPOT/1000)
		if predictedLatencyCtx.avgTPOTSLO > 0 {
			metrics.RecordRequestTPOTWithSLO(ctx, predictedLatencyCtx.incomingModelName, request.TargetModel, predictedLatencyCtx.avgTPOT, predictedLatencyCtx.avgTPOTSLO)
		}
	}

	id := request.Headers[requtil.RequestIdHeaderKey]

	// For disaggregated serving, also remove the prefill pod from tracking
	schedulingResult := predictedLatencyCtx.schedulingResult
	if schedulingResult != nil {
		if prefillResult, exists := schedulingResult.ProfileResults[Experimental_DefaultPrefillProfile]; exists && prefillResult != nil {
			if len(prefillResult.TargetEndpoints) > 0 {
				prefillPod := prefillResult.TargetEndpoints[0].GetMetadata()
				prefillEndpointName := types.NamespacedName{
					Name:      prefillPod.NamespacedName.Name,
					Namespace: prefillPod.NamespacedName.Namespace,
				}
				t.removeRequestFromEndpoint(prefillEndpointName, id)
				logger.V(logutil.DEBUG).Info("Removed prefill pod from running requests",
					"prefillPod", prefillPod.NamespacedName.Name,
					"requestID", id)
			}
		}
	}

	t.removeRequestFromQueue(id, predictedLatencyCtx)
	t.deletePredictedLatencyContextForRequest(request)
}

func (t *PredictedLatency) checkPredictor(logger logr.Logger, metadata *fwkdl.EndpointMetadata) bool {
	if metadata == nil {
		logger.V(logutil.TRACE).Info("PredictedLatency: Skipping hook because no target metadata was provided.")
		return false
	}
	if t.latencypredictor == nil {
		logger.V(logutil.TRACE).Info("PredictedLatency: Skipping hook because predictor missing")
		return false
	}
	return true
}
