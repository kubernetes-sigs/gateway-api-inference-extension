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
	"strings"
	"time"

	"github.com/go-logr/logr"
	"github.com/jellydator/ttlcache/v3"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	reqcommon "sigs.k8s.io/gateway-api-inference-extension/pkg/common/request"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
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
var _ requestcontrol.ResponseHeader = &PredictedLatency{}
var _ requestcontrol.ResponseBody = &PredictedLatency{}
var _ requestcontrol.AdmissionPlugin = &PredictedLatency{}

type predictedLatencyCtx struct {
	schedulingRequest         schedulingtypes.LLMRequest
	targetMetadata            *fwkdl.EndpointMetadata
	prefillTargetMetadata     *fwkdl.EndpointMetadata
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

	// promptText is the cached text representation of the request prompt, computed once.
	promptText      string
	inputTokenCount int // word-count estimate of prompt length, cached for counter bookkeeping

	prefixCacheScoresForEndpoints map[string]float64

	// ttftSLO is the target time to first token SLO for the request.
	ttftSLO float64
	// TPOTSLO is the target time per output token SLO for the request.
	avgTPOTSLO float64

	// predictedTTFTForScheduling is the map of pod names to predicted TTFT values for scheduling.
	predictionsForScheduling map[string]endpointPredictionResult

	// boolean set if request has valid endpoint based on predictions
	hasValidEndpoint bool

	// Snapshots of per-pod token-in-flight counters captured at dispatch time (PreRequest).
	// Used when building training entries so that training data reflects scheduling-time load.
	prefillTokensAtDispatch          int64 // snapshot from decode pod counter (used for TPOT training + prediction)
	prefillTokensAtDispatchOnPrefill int64 // snapshot from prefill pod counter (used for TTFT training in disaggregated mode)
	decodeTokensAtDispatch           int64
}

func newPredictedLatencyContext(request *schedulingtypes.LLMRequest) *predictedLatencyCtx {
	var promptText string
	if request.Body != nil {
		promptText = request.Body.PromptText()
	}
	return &predictedLatencyCtx{
		schedulingRequest:             *request,
		promptText:                    promptText,
		inputTokenCount:               len(strings.Fields(promptText)),
		lastSeenMetrics:               make(map[string]*fwkdl.Metrics),
		prefixCacheScoresForEndpoints: make(map[string]float64),
		predictionsForScheduling:      make(map[string]endpointPredictionResult),
		hasValidEndpoint:              true,
	}
}

func (s *PredictedLatency) getPredictedLatencyContextForRequest(request *schedulingtypes.LLMRequest) (*predictedLatencyCtx, error) {
	id := request.Headers[reqcommon.RequestIdHeaderKey]
	if item := s.sloContextStore.Get(id); item != nil {
		return item.Value(), nil
	}
	return nil, fmt.Errorf("SLO context not found for request ID: %s", id)
}

func (s *PredictedLatency) setPredictedLatencyContextForRequest(request *schedulingtypes.LLMRequest, ctx *predictedLatencyCtx) {
	id := request.Headers[reqcommon.RequestIdHeaderKey]
	s.sloContextStore.Set(id, ctx, ttlcache.DefaultTTL)
}

func (s *PredictedLatency) deletePredictedLatencyContextForRequest(request *schedulingtypes.LLMRequest) {
	id := request.Headers[reqcommon.RequestIdHeaderKey]
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

	logger.V(logutil.TRACE).Info("request ID for SLO tracking", "requestID", request.Headers[reqcommon.RequestIdHeaderKey], "endpointName", endpointName)
	if request.Headers[reqcommon.RequestIdHeaderKey] == "" {
		logger.V(logutil.DEBUG).Error(errors.New("missing request ID"), "PredictedLatency.PreRequest: Request is missing request ID header")
		return
	}

	id := request.Headers[reqcommon.RequestIdHeaderKey]

	// Get or create queue for this endpoint using sync.Map
	actual, _ := t.runningRequestLists.LoadOrStore(endpointName, newRequestPriorityQueue())
	endpointRequestList := actual.(*requestPriorityQueue)

	predictedLatencyCtx, err := t.getPredictedLatencyContextForRequest(request)
	if err != nil {
		id := request.Headers[reqcommon.RequestIdHeaderKey]
		logger.V(logutil.DEBUG).Info("PredictedLatency.PreRequest: Failed to get SLO context for request", "error", err, "requestID", id)
		return
	}

	added := endpointRequestList.Add(id, predictedLatencyCtx.avgTPOTSLO)
	if !added {
		logger.V(logutil.TRACE).Info("PredictedLatency: Item already exists in queue", "endpointName", endpointName, "requestID", id)
	}

	// Set up SLO request context
	predictedLatencyCtx.targetMetadata = targetMetadata
	if prefillResult, exists := schedulingResult.ProfileResults[Experimental_DefaultPrefillProfile]; exists && prefillResult != nil && len(prefillResult.TargetEndpoints) > 0 {
		prefillMetadata := prefillResult.TargetEndpoints[0].GetMetadata()
		predictedLatencyCtx.prefillTargetMetadata = prefillMetadata
		logger.V(logutil.DEBUG).Info("Prefill target identified for request", "requestID", id, "prefillEndpoint", prefillMetadata.NamespacedName.String())
	} else {
		logger.V(logutil.DEBUG).Info("No prefill target identified for request", "requestID", id)
	}
	predictedLatencyCtx.schedulingResult = schedulingResult
	predictedLatencyCtx.requestReceivedTimestamp = time.Now()
	refreshLastSeenMetrics(ctx, predictedLatencyCtx)

	// Update per-pod token-in-flight counters and snapshot for training.
	// prefillTokensInFlight is tracked on both the prefill pod (if disaggregated) and the decode pod.
	// decodeTokensInFlight is zeroed out — it is only accurate in pure streaming mode,
	// and kv_cache_percentage already captures decode load sufficiently.
	decodePodKey := endpointName.String()
	if predictedLatencyCtx.prefillTargetMetadata != nil {
		prefillPodKey := predictedLatencyCtx.prefillTargetMetadata.NamespacedName.String()
		t.podCounter(&t.prefillTokensInFlight, prefillPodKey).Add(int64(predictedLatencyCtx.inputTokenCount))
		predictedLatencyCtx.prefillTokensAtDispatchOnPrefill = t.podCounter(&t.prefillTokensInFlight, prefillPodKey).Load()
	}
	t.podCounter(&t.prefillTokensInFlight, decodePodKey).Add(int64(predictedLatencyCtx.inputTokenCount))
	predictedLatencyCtx.prefillTokensAtDispatch = t.podCounter(&t.prefillTokensInFlight, decodePodKey).Load()
	predictedLatencyCtx.decodeTokensAtDispatch = 0

	processPreRequestForLatencyPrediction(ctx, predictedLatencyCtx)
}

func (t *PredictedLatency) ResponseHeader(ctx context.Context, request *schedulingtypes.LLMRequest, response *requestcontrol.Response, targetMetadata *fwkdl.EndpointMetadata) {
	logger := log.FromContext(ctx)
	if request == nil {
		logger.V(logutil.DEBUG).Info("PredictedLatency.ResponseReceived: request is nil, skipping")
		return
	}
}

// ResponseBody now handles both per-chunk processing and request completion logic.
func (t *PredictedLatency) ResponseBody(ctx context.Context, request *schedulingtypes.LLMRequest, response *requestcontrol.Response, targetMetadata *fwkdl.EndpointMetadata) {
	logger := log.FromContext(ctx)
	if request == nil {
		logger.V(logutil.DEBUG).Info("PredictedLatency.ResponseBody: request is nil, skipping")
		return
	}
	if !t.checkPredictor(logger, targetMetadata) {
		return
	}

	now := time.Now()
	predictedLatencyCtx, err := t.getPredictedLatencyContextForRequest(request)
	if err != nil {
		id := request.Headers[reqcommon.RequestIdHeaderKey]
		logger.V(logutil.DEBUG).Info("PredictedLatency.ResponseBody: Failed to get SLO context", "error", err, "requestID", id)
		return
	}

	if predictedLatencyCtx.ttft == 0 {
		// For streaming, we record TTFT immediately.
		// For non-streaming, we wait until EndOfStream.
		if t.config.StreamingMode {
			processFirstTokenForLatencyPrediction(ctx, t.latencypredictor, t.config.StreamingMode, t.config.EndpointRoleLabel, predictedLatencyCtx, now, t.config.SamplingMean, t.config.MaxSampledTokens)

			// In disaggregated streaming, prefill is done once TTFT is hit.
			if predictedLatencyCtx.prefillTargetMetadata != nil {
				prefillPodKey := predictedLatencyCtx.prefillTargetMetadata.NamespacedName.String()
				if t.podCounter(&t.prefillTokensInFlight, prefillPodKey).Add(-int64(predictedLatencyCtx.inputTokenCount)) == 0 {
					t.prefillTokensInFlight.Delete(prefillPodKey)
				}
			}
		}
	} else {
		// This will be executed only if the predictedLatencyCtx.ttft is populated. Thus, this will be only populated for the token in the streaming response.
		processTokenForLatencyPrediction(ctx, t.latencypredictor, t.config.EndpointRoleLabel, predictedLatencyCtx, targetMetadata, now, t.config.SamplingMean, t.config.MaxSampledTokens)
	}

	if response.EndOfStream {
		// Track whether ResponseStreaming already handled TTFT (and prefill counter decrement).
		ttftNotYetRecorded := predictedLatencyCtx.ttft == 0
		// Only record TTFT training data in non-streaming mode.
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

		// Compute avgTPOT as (e2e - ttft) / (tokens - 1) for a more accurate overall average
		if predictedLatencyCtx.ttft > 0 && predictedLatencyCtx.generatedTokenCount > 1 {
			e2eMs := float64(now.Sub(predictedLatencyCtx.requestReceivedTimestamp).Milliseconds())
			predictedLatencyCtx.avgTPOT = (e2eMs - predictedLatencyCtx.ttft) / float64(predictedLatencyCtx.generatedTokenCount-1)
		}

		if predictedLatencyCtx.avgTPOT > 0 {
			logger.V(logutil.TRACE).Info("Averages calculated", "avgActualTPOT", predictedLatencyCtx.avgTPOT, "avgPredictedTPOT", predictedLatencyCtx.avgPredictedTPOT)
			metrics.RecordRequestTPOT(ctx, predictedLatencyCtx.incomingModelName, request.TargetModel, predictedLatencyCtx.avgTPOT/1000)
			metrics.RecordRequestPredictedTPOT(ctx, predictedLatencyCtx.incomingModelName, request.TargetModel, predictedLatencyCtx.avgPredictedTPOT/1000)
			if predictedLatencyCtx.avgTPOTSLO > 0 {
				metrics.RecordRequestTPOTWithSLO(ctx, predictedLatencyCtx.incomingModelName, request.TargetModel, predictedLatencyCtx.avgTPOT, predictedLatencyCtx.avgTPOTSLO)
			}

			// Record one TPOT training entry per request using avgTPOT and dispatch-time metrics
			if m, err := getLatestMetricsForProfile(predictedLatencyCtx, ""); err == nil {
				entry := buildTrainingEntry(
					t.config.EndpointRoleLabel,
					targetMetadata,
					m,
					predictedLatencyCtx.promptText,
					0, // TTFT not recorded for TPOT
					predictedLatencyCtx.avgTPOT,
					now,
					0, // not used for TPOT prediction
					0, // TPOT does not use prefix cache score
				)
				entry.PrefillTokensInFlight = predictedLatencyCtx.prefillTokensAtDispatch
				entry.DecodeTokensInFlight = predictedLatencyCtx.decodeTokensAtDispatch
				if err := t.latencypredictor.AddTrainingDataBulk([]latencypredictor.TrainingEntry{entry}); err != nil {
					logger.V(logutil.DEBUG).Error(err, "record TPOT training failed")
				}
			}
		}

		// Decrement per-pod token-in-flight counters now that the request is complete.
		decodePodKey := targetMetadata.NamespacedName.String()
		// If TTFT was not yet recorded when we entered EOS, the prefill pod counter
		// was never decremented in ResponseStreaming — decrement it here.
		if ttftNotYetRecorded && predictedLatencyCtx.prefillTargetMetadata != nil {
			prefillPodKey := predictedLatencyCtx.prefillTargetMetadata.NamespacedName.String()
			if t.podCounter(&t.prefillTokensInFlight, prefillPodKey).Add(-int64(predictedLatencyCtx.inputTokenCount)) == 0 {
				t.prefillTokensInFlight.Delete(prefillPodKey)
			}
		}
		if t.podCounter(&t.prefillTokensInFlight, decodePodKey).Add(-int64(predictedLatencyCtx.inputTokenCount)) == 0 {
			t.prefillTokensInFlight.Delete(decodePodKey)
		}

		id := request.Headers[reqcommon.RequestIdHeaderKey]
		t.removeRequestFromQueue(id, predictedLatencyCtx)
		t.deletePredictedLatencyContextForRequest(request)
	}
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

func (t *PredictedLatency) AdmitRequest(ctx context.Context, request *schedulingtypes.LLMRequest, endpoints []schedulingtypes.Endpoint) error {
	logger := log.FromContext(ctx)
	if request == nil {
		// This should not happen as the framework should not call AdmitRequest with a nil request, but we defensively check to avoid panics
		logger.V(logutil.DEBUG).Info("PredictedLatency.AdmitRequest: request is nil, skipping")
		return nil
	}

	predictedLatencyCtx, err := t.getPredictedLatencyContextForRequest(request)
	if err != nil {
		// If we can't find the predictedLatency context, we log the error but allow the request to proceed. This is a fail-open approach to avoid rejecting requests due to internal errors in our plugin.
		id := request.Headers[reqcommon.RequestIdHeaderKey]
		logger.V(logutil.DEBUG).Error(err, "PredictedLatency.AdmitRequest: Failed to get PredictedLatency context for request", "requestID", id)
		return nil
	}

	// If there is no valid pod for the request, reject it
	if !predictedLatencyCtx.hasValidEndpoint && request.Objectives.Priority < 0 {
		logger.V(logutil.DEBUG).Info("PredictedLatency.AdmitRequest: Rejecting a sheddable request as no valid endpoint available due to slo violation", "requestID", request.Headers[reqcommon.RequestIdHeaderKey])
		return errors.New("no valid endpoint available to serve the request")
	}

	return nil
}
