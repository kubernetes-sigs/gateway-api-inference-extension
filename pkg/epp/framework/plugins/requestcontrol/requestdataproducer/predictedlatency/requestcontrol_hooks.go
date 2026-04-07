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
	"time"

	"github.com/go-logr/logr"
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

var _ requestcontrol.PreRequest = &PredictedLatency{}
var _ requestcontrol.ResponseHeader = &PredictedLatency{}
var _ requestcontrol.ResponseBody = &PredictedLatency{}

// --- RequestControl Hooks ---

func (t *PredictedLatency) PreRequest(ctx context.Context, request *schedulingtypes.InferenceRequest, schedulingResult *schedulingtypes.SchedulingResult) {
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

func (t *PredictedLatency) ResponseHeader(ctx context.Context, request *schedulingtypes.InferenceRequest, response *requestcontrol.Response, targetMetadata *fwkdl.EndpointMetadata) {
	logger := log.FromContext(ctx)
	if request == nil {
		logger.V(logutil.DEBUG).Info("PredictedLatency.ResponseReceived: request is nil, skipping")
		return
	}
}

// ResponseBody handles both per-chunk processing and request completion logic.
func (t *PredictedLatency) ResponseBody(ctx context.Context, request *schedulingtypes.InferenceRequest, response *requestcontrol.Response, targetMetadata *fwkdl.EndpointMetadata) {
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
		if t.config.StreamingMode && !response.EndOfStream {
			processFirstTokenForLatencyPrediction(ctx, t.latencypredictor, t.config.StreamingMode, t.config.EndpointRoleLabel, predictedLatencyCtx, now, t.config.SamplingMean, t.config.MaxDecodeTokenSamplesForPrediction)

			if predictedLatencyCtx.prefillTargetMetadata != nil {
				prefillPodKey := predictedLatencyCtx.prefillTargetMetadata.NamespacedName.String()
				if t.podCounter(&t.prefillTokensInFlight, prefillPodKey).Add(-int64(predictedLatencyCtx.inputTokenCount)) == 0 {
					t.prefillTokensInFlight.Delete(prefillPodKey)
				}
			}
		}
	} else {
		processTokenForLatencyPrediction(ctx, t.latencypredictor, t.config.EndpointRoleLabel, predictedLatencyCtx, targetMetadata, now, t.config.SamplingMean, t.config.MaxDecodeTokenSamplesForPrediction)
	}

	if response.EndOfStream {
		ttftNotYetRecorded := predictedLatencyCtx.ttft == 0
		if !t.config.StreamingMode {
			processFirstTokenForLatencyPrediction(ctx, t.latencypredictor, t.config.StreamingMode, t.config.EndpointRoleLabel, predictedLatencyCtx, now, t.config.SamplingMean, t.config.MaxDecodeTokenSamplesForPrediction)
		}

		if predictedLatencyCtx.ttft > 0 {
			// In non-streaming mode, TTFT represents full e2e latency.
			logger.V(logutil.TRACE).Info("Averages calculated", "avgActualTTFT", predictedLatencyCtx.ttft, "avgPredictedTTFT", predictedLatencyCtx.predictedTTFT)
			metrics.RecordRequestTTFT(ctx, predictedLatencyCtx.incomingModelName, request.TargetModel, predictedLatencyCtx.ttft/1000)
			metrics.RecordRequestPredictedTTFT(ctx, predictedLatencyCtx.incomingModelName, request.TargetModel, predictedLatencyCtx.predictedTTFT/1000)
			if predictedLatencyCtx.ttftSLO > 0 {
				metrics.RecordRequestTTFTWithSLO(ctx, predictedLatencyCtx.incomingModelName, request.TargetModel, predictedLatencyCtx.ttft, predictedLatencyCtx.ttftSLO)
			}
		}

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

			if m, err := getLatestMetricsForProfile(predictedLatencyCtx, ""); err == nil {
				entry := buildTrainingEntry(
					t.config.EndpointRoleLabel,
					targetMetadata,
					m,
					predictedLatencyCtx.promptText,
					0,
					predictedLatencyCtx.avgTPOT,
					now,
					0,
					0,
				)
				entry.PrefillTokensInFlight = predictedLatencyCtx.prefillTokensAtDispatch
				entry.DecodeTokensInFlight = predictedLatencyCtx.decodeTokensAtDispatch
				if err := t.latencypredictor.AddTrainingDataBulk([]latencypredictor.TrainingEntry{entry}); err != nil {
					logger.V(logutil.DEBUG).Error(err, "record TPOT training failed")
				}
			}
		}

		decodePodKey := targetMetadata.NamespacedName.String()
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

// processPreRequestForLatencyPrediction looks up the stored prediction for the target endpoint.
func processPreRequestForLatencyPrediction(ctx context.Context, predictedLatencyCtx *predictedLatencyCtx) {
	logger := log.FromContext(ctx)
	targetName := predictedLatencyCtx.targetMetadata.NamespacedName.Name
	if m := predictedLatencyCtx.prefillTargetMetadata; m != nil {
		targetName = m.NamespacedName.Name
	}
	if storedPred, ok := predictedLatencyCtx.predictionsForScheduling[targetName]; ok {
		logger.V(logutil.DEBUG).Info("PreRequest TTFT from stored prediction", "value_ms", storedPred.TTFT, "endpoint", targetName)
		predictedLatencyCtx.predictedTTFT = storedPred.TTFT
	} else {
		logger.V(logutil.DEBUG).Info("PreRequest: no stored prediction found for target endpoint", "endpoint", targetName)
		predictedLatencyCtx.predictedTTFT = 0
	}
	predictedLatencyCtx.lastTokenTimestamp = time.Now()
}

// processFirstTokenForLatencyPrediction records actual TTFT, trains, predicts first TPOT.
func processFirstTokenForLatencyPrediction(
	ctx context.Context,
	predictor latencypredictor.PredictorInterface,
	streamingMode bool,
	endpointRoleLabel string,
	predictedLatencyCtx *predictedLatencyCtx,
	now time.Time,
	samplingMean float64,
	maxDecodeTokenSamplesForPrediction int,
) {
	logger := log.FromContext(ctx)

	initializeSampler(ctx, predictedLatencyCtx, samplingMean, maxDecodeTokenSamplesForPrediction)
	predictedLatencyCtx.ttft = float64(now.Sub(predictedLatencyCtx.requestReceivedTimestamp).Milliseconds())
	predictedLatencyCtx.generatedTokenCount = 1

	if prefillTargetMetadata := predictedLatencyCtx.prefillTargetMetadata; prefillTargetMetadata != nil {
		prefillMetrics, err := getLatestMetricsForProfile(predictedLatencyCtx, Experimental_DefaultPrefillProfile)
		if err == nil {
			prefillPrefixCacheScore := predictedLatencyCtx.prefixCacheScoresForEndpoints[prefillTargetMetadata.NamespacedName.Name]
			logger.V(logutil.DEBUG).Info("Recording prefill TTFT training data",
				"ttft_ms", predictedLatencyCtx.ttft,
				"prefillPod", prefillTargetMetadata.NamespacedName.Name,
				"prefixCacheScore", prefillPrefixCacheScore)
			recordTTFTTrainingData(ctx, predictor, endpointRoleLabel, predictedLatencyCtx, prefillMetrics, prefillTargetMetadata, now, prefillPrefixCacheScore)
		}
	} else {
		m, err := getLatestMetricsForProfile(predictedLatencyCtx, "")
		if err != nil {
			logger.V(logutil.DEBUG).Info("Skipping TTFT training due to missing metrics or schedulingResult", "error", err)
			return
		}
		targetEndpointMetadata := predictedLatencyCtx.targetMetadata
		prefixCacheScore := predictedLatencyCtx.prefixCacheScoresForEndpoints[targetEndpointMetadata.NamespacedName.Name]
		logger.V(logutil.DEBUG).Info("Recording TTFT training data", "ttft_ms", predictedLatencyCtx.ttft, "predicted_ttft_ms", predictedLatencyCtx.predictedTTFT, "prefixCacheScore", prefixCacheScore)
		recordTTFTTrainingData(ctx, predictor, endpointRoleLabel, predictedLatencyCtx, m, targetEndpointMetadata, now, prefixCacheScore)
	}

	if streamingMode {
		predictFirstTPOT(ctx, predictedLatencyCtx)
	}

	predictedLatencyCtx.lastTokenTimestamp = now
	refreshLastSeenMetrics(ctx, predictedLatencyCtx)
}

func initializeSampler(ctx context.Context, predictedLatencyCtx *predictedLatencyCtx, samplingMean float64, maxDecodeTokenSamplesForPrediction int) {
	if predictedLatencyCtx.decodeTokenSampler == nil {
		logger := log.FromContext(ctx)
		requestID := predictedLatencyCtx.schedulingRequest.Headers[reqcommon.RequestIdHeaderKey]
		predictedLatencyCtx.decodeTokenSampler = newDecodeTokenSampler(requestID, samplingMean, maxDecodeTokenSamplesForPrediction)
		logger.V(logutil.DEBUG).Info("Initialized token sampler for first token", "request_id", requestID, "next_prediction_token", predictedLatencyCtx.decodeTokenSampler.getNextSampleToken())
	}
}

func predictFirstTPOT(ctx context.Context, predictedLatencyCtx *predictedLatencyCtx) {
	logger := log.FromContext(ctx)
	targetName := predictedLatencyCtx.targetMetadata.NamespacedName.Name
	if storedPred, ok := predictedLatencyCtx.predictionsForScheduling[targetName]; ok {
		logger.V(logutil.DEBUG).Info("first TPOT from stored prediction", "value_ms", storedPred.TPOT)
		predictedLatencyCtx.predictedTPOTObservations = append(predictedLatencyCtx.predictedTPOTObservations, storedPred.TPOT)
		predictedLatencyCtx.avgPredictedTPOT = calculateRunningAverage(predictedLatencyCtx.avgPredictedTPOT, storedPred.TPOT, len(predictedLatencyCtx.predictedTPOTObservations))
	} else {
		logger.V(logutil.DEBUG).Info("first TPOT: no stored prediction found for target endpoint", "endpoint", targetName)
		predictedLatencyCtx.predictedTPOTObservations = append(predictedLatencyCtx.predictedTPOTObservations, 0)
		predictedLatencyCtx.avgPredictedTPOT = calculateRunningAverage(predictedLatencyCtx.avgPredictedTPOT, 0, len(predictedLatencyCtx.predictedTPOTObservations))
	}
}

// processTokenForLatencyPrediction records actual inter-token latency, sampled predictions, and advances timestamp.
func processTokenForLatencyPrediction(
	ctx context.Context,
	predictor latencypredictor.PredictorInterface,
	endpointRoleLabel string,
	predictedLatencyCtx *predictedLatencyCtx,
	targetEndpointMetadata *fwkdl.EndpointMetadata,
	now time.Time,
	samplingMean float64,
	maxDecodeTokenSamplesForPrediction int,
) {
	logger := log.FromContext(ctx)

	if predictedLatencyCtx.decodeTokenSampler == nil {
		requestID := predictedLatencyCtx.schedulingRequest.Headers[reqcommon.RequestIdHeaderKey]
		predictedLatencyCtx.decodeTokenSampler = newDecodeTokenSampler(requestID, samplingMean, maxDecodeTokenSamplesForPrediction)
		logger.V(logutil.DEBUG).Info("Initialized token sampler for subsequent tokens", "request_id", requestID, "next_prediction_token", predictedLatencyCtx.decodeTokenSampler.getNextSampleToken())
	}

	latencyMs := float64(now.Sub(predictedLatencyCtx.lastTokenTimestamp).Milliseconds())
	predictedLatencyCtx.generatedTokenCount++

	if predictedLatencyCtx.generatedTokenCount == 2 || predictedLatencyCtx.decodeTokenSampler.shouldPredict(predictedLatencyCtx.generatedTokenCount) {
		predictedLatencyCtx.tpotObservations = append(predictedLatencyCtx.tpotObservations, latencyMs)
	}
	if predictedLatencyCtx.generatedTokenCount == 2 {
		logger.V(logutil.DEBUG).Info("First inter-token latency observed",
			"actual_tpot_ms", latencyMs,
			"predicted_tpot_ms", predictedLatencyCtx.avgPredictedTPOT)
	}

	m, err := getLatestMetricsForProfile(predictedLatencyCtx, "")
	if err != nil {
		logger.V(logutil.DEBUG).Info("Skipping TPOT prediction due to missing metrics or schedulingResult", "error", err)
		return
	}

	if predictedLatencyCtx.decodeTokenSampler.shouldPredict(predictedLatencyCtx.generatedTokenCount) {
		in := buildPredictionRequest(
			endpointRoleLabel,
			targetEndpointMetadata,
			m,
			predictedLatencyCtx.promptText,
			predictedLatencyCtx.generatedTokenCount,
			0,
		)
		start := time.Now()
		p, err := predictor.Predict(ctx, in)
		dur := time.Since(start)
		if err != nil || p == nil {
			logger.V(logutil.DEBUG).Error(err, "TPOT predict failed", "duration_ms", dur.Milliseconds())
			predictedLatencyCtx.predictedTPOTObservations = append(predictedLatencyCtx.predictedTPOTObservations, 0)
			predictedLatencyCtx.avgPredictedTPOT = calculateRunningAverage(predictedLatencyCtx.avgPredictedTPOT, 0, len(predictedLatencyCtx.predictedTPOTObservations))
		} else {
			logger.V(logutil.DEBUG).Info("TPOT predict succeeded", "value_ms", p.TPOT, "duration_ms", dur.Milliseconds())
			predictedLatencyCtx.predictedTPOTObservations = append(predictedLatencyCtx.predictedTPOTObservations, p.TPOT)
			predictedLatencyCtx.avgPredictedTPOT = calculateRunningAverage(predictedLatencyCtx.avgPredictedTPOT, p.TPOT, len(predictedLatencyCtx.predictedTPOTObservations))
		}
		metrics.RecordRequestTPOTPredictionDuration(ctx, predictedLatencyCtx.schedulingRequest.TargetModel, predictedLatencyCtx.incomingModelName, dur.Seconds())
		predictedLatencyCtx.decodeTokenSampler.recordPrediction(predictedLatencyCtx.generatedTokenCount)
	}

	predictedLatencyCtx.lastTokenTimestamp = now
	refreshLastSeenMetrics(ctx, predictedLatencyCtx)
}
