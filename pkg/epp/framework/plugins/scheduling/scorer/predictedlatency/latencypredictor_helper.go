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

// Package requestcontrol contains helpers to decouple latency-predictor logic.
package predictedlatency

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	reqcommon "sigs.k8s.io/gateway-api-inference-extension/pkg/common/request"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/sidecars/latencypredictorasync"
)

// buildPredictionRequest constructs a prediction request from endpoint metrics and request data.
// If endpointRoleLabel is configured, it extracts the role from the endpoint's labels and
// populates the PodType field, enabling role-aware predictions (e.g., prefill vs decode).
func buildPredictionRequest(
	endpointRoleLabel string,
	targetEndpointMetadata *fwkdl.EndpointMetadata,
	metrics *fwkdl.Metrics,
	prompt string,
	generatedTokens int,
	prefixCacheScore float64,
) latencypredictor.PredictionRequest {
	podType := ""
	if endpointRoleLabel != "" && targetEndpointMetadata != nil && targetEndpointMetadata.Labels != nil {
		podType = targetEndpointMetadata.Labels[endpointRoleLabel]
	}

	return latencypredictor.PredictionRequest{
		KVCachePercentage:  metrics.KVCacheUsagePercent,
		InputTokenLength:   len(strings.Fields(prompt)), // Simple word-based tokenization
		NumRequestWaiting:  metrics.WaitingQueueSize,
		NumRequestRunning:  metrics.RunningRequestsSize,
		NumTokensGenerated: generatedTokens,
		PrefixCacheScore:   prefixCacheScore,
		PodType:            podType,
	}
}

// refreshLastSeenMetrics updates predictedLatencyCtx.LastSeenMetrics for both primary and Experimental_DefaultPrefillProfile.
func refreshLastSeenMetrics(ctx context.Context, predictedLatencyCtx *predictedLatencyCtx) {
	if sr := predictedLatencyCtx.schedulingResult; sr != nil {

		for profileName, profileResult := range sr.ProfileResults {
			if profileResult != nil && profileResult.TargetEndpoints != nil && len(profileResult.TargetEndpoints) > 0 {
				predictedLatencyCtx.lastSeenMetrics[profileName] = profileResult.TargetEndpoints[0].GetMetrics().Clone()
			}
		}
	} else {
		log.FromContext(ctx).V(logutil.DEBUG).Info("No scheduling result found, skipping metrics refresh")
	}
}

// getLatestMetricsForProfile retrieves the latest metrics for the specified profile from predictedLatencyCtx.LastSeenMetrics.
// If profileName is empty, it defaults to the primary profile.
func getLatestMetricsForProfile(predictedLatencyCtx *predictedLatencyCtx, profileName string) (*fwkdl.Metrics, error) {
	if len(predictedLatencyCtx.lastSeenMetrics) == 0 {
		return nil, errors.New("no last seen metrics available for prediction")
	}

	if profileName == "" && predictedLatencyCtx.schedulingResult != nil {
		profileName = predictedLatencyCtx.schedulingResult.PrimaryProfileName
	}

	if metrics, exists := predictedLatencyCtx.lastSeenMetrics[profileName]; exists {
		return metrics, nil
	}

	return nil, fmt.Errorf("no metrics found for profile %s", profileName)
}

// processPreRequestForLatencyPrediction looks up the stored prediction for the target endpoint
// and uses it for TTFT, avoiding a redundant prediction call.
func processPreRequestForLatencyPrediction(
	ctx context.Context,
	predictor latencypredictor.PredictorInterface,
	predictedLatencyCtx *predictedLatencyCtx,
) {
	logger := log.FromContext(ctx)

	m, err := getLatestMetricsForProfile(predictedLatencyCtx, "")
	if err != nil {
		logger.V(logutil.DEBUG).Info("Skipping TTFT prediction due to missing metrics", "error", err)
		return
	}

	targetPod := predictedLatencyCtx.targetMetadata
	prefix_cache_score := predictedLatencyCtx.prefixCacheScoresForEndpoints[targetPod.NamespacedName.Name]

	in := latencypredictor.PredictionRequest{
		KVCachePercentage:  m.KVCacheUsagePercent,
		InputTokenLength:   len(strings.Fields(predictedLatencyCtx.schedulingRequest.LLM.Body.Completions.Prompt)),
		NumRequestWaiting:  m.WaitingQueueSize,
		NumRequestRunning:  m.RunningRequestsSize,
		NumTokensGenerated: 0,
		PrefixCacheScore:   prefix_cache_score,
	}

	// Predict TTFT
	start := time.Now()
	p, err := predictor.Predict(ctx, in)
	dur := time.Since(start)
	switch {
	case err != nil:
		logger.V(logutil.DEBUG).Error(err, "header TTFT predict failed", "duration_ms", dur.Milliseconds())
		predictedLatencyCtx.predictedTTFT = 0
	case p == nil:
		logger.V(logutil.DEBUG).Info("header TTFT predict nil", "duration_ms", dur.Milliseconds())
		predictedLatencyCtx.predictedTTFT = 0
	default:
		logger.V(logutil.DEBUG).Info("header TTFT succeeded", "value_ms", p.TTFT, "duration_ms", dur.Milliseconds())
		metrics.RecordRequestTTFTPredictionDuration(ctx, predictedLatencyCtx.schedulingRequest.LLM.TargetModel, predictedLatencyCtx.incomingModelName, dur.Seconds())

		predictedLatencyCtx.predictedTTFT = p.TTFT
	}

	// Advance timestamp for first token reference
	predictedLatencyCtx.lastTokenTimestamp = time.Now()
}

// processFirstTokenForLatencyPrediction records actual TTFT, trains, predicts first TPOT, updates predictedLatencyCtx, and advances timestamp.
func processFirstTokenForLatencyPrediction(
	ctx context.Context,
	predictor latencypredictor.PredictorInterface,
	streamingMode bool,
	predictedLatencyCtx *predictedLatencyCtx,
	now time.Time,
	samplingMean float64,
	maxSampledTokens int,
) {
	logger := log.FromContext(ctx)

	initializeSampler(ctx, predictedLatencyCtx, samplingMean, maxSampledTokens)
	// Actual TTFT
	predictedLatencyCtx.ttft = float64(now.Sub(predictedLatencyCtx.requestReceivedTimestamp).Milliseconds())
	predictedLatencyCtx.generatedTokenCount = 1

	if prefillTargetMetadata := predictedLatencyCtx.prefillTargetMetadata; prefillTargetMetadata != nil {
		// Disaggregated mode: record TTFT for prefill pod only (TTFT is dominated by prefill work)
		prefillMetrics, err := getLatestMetricsForProfile(predictedLatencyCtx, Experimental_DefaultPrefillProfile)
		if err == nil {
			prefillPrefixCacheScore := predictedLatencyCtx.prefixCacheScoresForEndpoints[prefillTargetMetadata.NamespacedName.Name]
			logger.V(logutil.DEBUG).Info("Recording prefill TTFT training data",
				"ttft_ms", predictedLatencyCtx.ttft,
				"prefillPod", prefillTargetMetadata.NamespacedName.Name,
				"prefixCacheScore", prefillPrefixCacheScore)
			recordTTFTTrainingData(ctx, predictor, predictedLatencyCtx, prefillMetrics, now, prefillPrefixCacheScore)
		}
	} else {
		// Monolithic mode: record TTFT for the single pod
		m, err := getLatestMetricsForProfile(predictedLatencyCtx, "")
		if err != nil {
			logger.V(logutil.DEBUG).Info("Skipping TTFT training due to missing metrics or schedulingResult", "error", err)
			return
		}
		targetEndpointMetadata := predictedLatencyCtx.targetMetadata
		prefixCacheScore := predictedLatencyCtx.prefixCacheScoresForEndpoints[targetEndpointMetadata.NamespacedName.Name]
		logger.V(logutil.DEBUG).Info("Recording TTFT training data", "ttft_ms", predictedLatencyCtx.ttft, "predicted_ttft_ms", predictedLatencyCtx.predictedTTFT, "prefixCacheScore", prefixCacheScore)
		recordTTFTTrainingData(ctx, predictor, predictedLatencyCtx, m, now, prefixCacheScore)
	}

	if streamingMode {
		predictFirstTPOT(ctx, predictor, predictedLatencyCtx)
	}

	// Advance timestamp
	predictedLatencyCtx.lastTokenTimestamp = now
	// Refresh metrics
	refreshLastSeenMetrics(ctx, predictedLatencyCtx)
}

func initializeSampler(ctx context.Context, predictedLatencyCtx *predictedLatencyCtx, samplingMean float64, maxSampledTokens int) {
	if predictedLatencyCtx.tokenSampler == nil {
		logger := log.FromContext(ctx)
		requestID := predictedLatencyCtx.schedulingRequest.LLM.Headers[reqcommon.RequestIdHeaderKey]
		predictedLatencyCtx.tokenSampler = newTokenSampler(requestID, samplingMean, maxSampledTokens)
		logger.V(logutil.DEBUG).Info("Initialized token sampler for first token", "request_id", requestID, "next_prediction_token", predictedLatencyCtx.tokenSampler.getNextSampleToken())
	}
}

func recordTTFTTrainingData(
	ctx context.Context,
	predictor latencypredictor.PredictorInterface,
	predictedLatencyCtx *predictedLatencyCtx,
	m *fwkdl.Metrics,
	now time.Time,
	prefixCacheScore float64,
) {
	logger := log.FromContext(ctx)
	// Train TTFT
	entry := latencypredictor.TrainingEntry{
		KVCachePercentage:  m.KVCacheUsagePercent,
		InputTokenLength:   len(strings.Fields(predictedLatencyCtx.schedulingRequest.LLM.Body.Completions.Prompt)),
		ActualTTFT:         predictedLatencyCtx.ttft,
		ActualTPOT:         0,
		Timestamp:          now,
		NumRequestWaiting:  m.WaitingQueueSize,
		NumRequestRunning:  m.RunningRequestsSize,
		NumTokensGenerated: 0,
		PrefixCacheScore:   prefixCacheScore,
	}
	entry.DecodeTokensInFlight = predictedLatencyCtx.decodeTokensAtDispatch
	if err := predictor.AddTrainingDataBulk([]latencypredictor.TrainingEntry{entry}); err != nil {
		logger.V(logutil.DEBUG).Error(err, "record TTFT training failed")
	}
}

func predictFirstTPOT(
	ctx context.Context,
	predictor latencypredictor.PredictorInterface,
	predictedLatencyCtx *predictedLatencyCtx,
) {
	logger := log.FromContext(ctx)
	m, err := getLatestMetricsForProfile(predictedLatencyCtx, "")
	if err != nil {
		logger.V(logutil.DEBUG).Info("Skipping first TPOT prediction due to missing metrics",
			"error", err)
		return
	}

	// Predict first TPOT
	in := latencypredictor.PredictionRequest{
		KVCachePercentage:  m.KVCacheUsagePercent,
		InputTokenLength:   len(strings.Fields(predictedLatencyCtx.schedulingRequest.LLM.Body.Completions.Prompt)),
		NumRequestWaiting:  m.WaitingQueueSize,
		NumRequestRunning:  m.RunningRequestsSize,
		NumTokensGenerated: predictedLatencyCtx.generatedTokenCount,
		PrefixCacheScore:   0,
	}
	start := time.Now()
	p, err := predictor.Predict(ctx, in)
	dur := time.Since(start)
	if err != nil || p == nil {
		logger.V(logutil.DEBUG).Error(err, "first TPOT predict failed", "duration_ms", dur.Milliseconds())
		predictedLatencyCtx.predictedTPOTObservations = append(predictedLatencyCtx.predictedTPOTObservations, 0)
		predictedLatencyCtx.avgPredictedTPOT = calculateRunningAverage(predictedLatencyCtx.avgPredictedTPOT, 0, len(predictedLatencyCtx.predictedTPOTObservations))
	}
	metrics.RecordRequestTPOTPredictionDuration(ctx, predictedLatencyCtx.schedulingRequest.LLM.TargetModel, predictedLatencyCtx.incomingModelName, dur.Seconds())
}

// processTokenForLatencyPrediction records actual inter-token latency, trains, predicts sampled TPOT, updates predictedLatencyCtx, and advances timestamp.
func processTokenForLatencyPrediction(
	ctx context.Context,
	predictor latencypredictor.PredictorInterface,
	predictedLatencyCtx *predictedLatencyCtx,
	now time.Time,
	samplingMean float64,
	maxSampledTokens int,
) {
	logger := log.FromContext(ctx)

	// Initialize sampler if not yet
	if predictedLatencyCtx.tokenSampler == nil {
		requestID := predictedLatencyCtx.schedulingRequest.LLM.Headers[reqcommon.RequestIdHeaderKey]
		predictedLatencyCtx.tokenSampler = newTokenSampler(requestID, samplingMean, maxSampledTokens)
		logger.V(logutil.DEBUG).Info("Initialized token sampler for subsequent tokens", "request_id", requestID, "next_prediction_token", predictedLatencyCtx.tokenSampler.getNextSampleToken())
	}

	// Inter-token latency
	latencyMs := float64(now.Sub(predictedLatencyCtx.lastTokenTimestamp).Milliseconds())
	predictedLatencyCtx.generatedTokenCount++

	// record sampled TPOT observations (avgTPOT is computed in ResponseComplete from e2e latency)
	if predictedLatencyCtx.generatedTokenCount == 2 || predictedLatencyCtx.tokenSampler.shouldPredict(predictedLatencyCtx.generatedTokenCount) {
		predictedLatencyCtx.tpotObservations = append(predictedLatencyCtx.tpotObservations, latencyMs)
	}
	if predictedLatencyCtx.generatedTokenCount == 2 {
		// debug log actual and predicted tpot
		logger.V(logutil.DEBUG).Info("First inter-token latency observed",
			"actual_tpot_ms", latencyMs,
			"predicted_tpot_ms", predictedLatencyCtx.avgPredictedTPOT)
	}

	// TPOT training is now done once per request in ResponseComplete using avgTPOT

	m, err := getLatestMetricsForProfile(predictedLatencyCtx, "")
	if err != nil {
		logger.V(logutil.DEBUG).Info("Skipping TPOT prediction due to missing metrics or schedulingResult",
			"error", err)
		return
	}
	// Record actual TPOT
	entry := latencypredictor.TrainingEntry{
		KVCachePercentage:  m.KVCacheUsagePercent,
		InputTokenLength:   len(strings.Fields(predictedLatencyCtx.schedulingRequest.LLM.Body.Completions.Prompt)),
		ActualTTFT:         0,
		ActualTPOT:         latencyMs,
		Timestamp:          now,
		NumRequestWaiting:  m.WaitingQueueSize,
		NumRequestRunning:  m.RunningRequestsSize,
		NumTokensGenerated: predictedLatencyCtx.generatedTokenCount - 1,
		PrefixCacheScore:   0, // TPOT does not use prefix cache score
	}
	if err := predictor.AddTrainingDataBulk([]latencypredictor.TrainingEntry{entry}); err != nil {
		logger.V(logutil.DEBUG).Error(err, "record TPOT training failed")
	}

	// Sampled predict
	if predictedLatencyCtx.tokenSampler.shouldPredict(predictedLatencyCtx.generatedTokenCount) {
		in := latencypredictor.PredictionRequest{
			KVCachePercentage:  m.KVCacheUsagePercent,
			InputTokenLength:   len(strings.Fields(predictedLatencyCtx.schedulingRequest.LLM.Body.Completions.Prompt)),
			NumRequestWaiting:  m.WaitingQueueSize,
			NumRequestRunning:  m.RunningRequestsSize,
			NumTokensGenerated: predictedLatencyCtx.generatedTokenCount,
			PrefixCacheScore:   0, // TPOT does not use prefix cache score
		}
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
		metrics.RecordRequestTPOTPredictionDuration(ctx, predictedLatencyCtx.schedulingRequest.LLM.TargetModel, predictedLatencyCtx.incomingModelName, dur.Seconds())

		predictedLatencyCtx.tokenSampler.recordPrediction(predictedLatencyCtx.generatedTokenCount)
	}

	// Advance timestamp
	predictedLatencyCtx.lastTokenTimestamp = now
	// Refresh metrics
	refreshLastSeenMetrics(ctx, predictedLatencyCtx)
}

// bulkPredictWithMetrics performs bulk predictions for multiple pods using their metrics states.
// Returns predictions in the same order as the input slices.
func bulkPredictWithMetrics(
	ctx context.Context,
	predictedLatencyContext *predictedLatencyCtx,
	predictor latencypredictor.PredictorInterface,
	metricsStates []*fwkdl.Metrics,
	endpointRoleLabel string,
	targetEndpointsMetadatas []*fwkdl.EndpointMetadata,
	prompts []string,
	generatedTokenCounts []int,
	prefixCacheScores []float64,
	prefillTokensInFlights []int64,
) ([]*latencypredictor.PredictionResponse, error) {
	logger := log.FromContext(ctx)

	// Validate input lengths
	if len(targetEndpointsMetadatas) != len(metricsStates) || len(metricsStates) != len(prompts) || len(prompts) != len(generatedTokenCounts) || len(generatedTokenCounts) != len(prefixCacheScores) {
		return nil, fmt.Errorf("input slice lengths must match: endpoints=%d, metrics=%d, prompts=%d, tokenCounts=%d, prefixScores=%d",
			len(targetEndpointsMetadatas), len(metricsStates), len(prompts), len(generatedTokenCounts), len(prefixCacheScores))
	}

	if len(metricsStates) == 0 {
		return []*latencypredictor.PredictionResponse{}, nil
	}

	// Validate that no metrics state is nil
	for i, metricsState := range metricsStates {
		if metricsState == nil {
			return nil, fmt.Errorf("metrics state at index %d cannot be nil", i)
		}
	}

	// Validate that no endpoint metadata is nil
	for i, endpointMetadata := range targetEndpointsMetadatas {
		if endpointMetadata == nil {
			return nil, fmt.Errorf("endpoint metadata at index %d cannot be nil", i)
		}
	}

	// Build bulk prediction requests
	bulkRequests := make([]latencypredictor.PredictionRequest, len(metricsStates))
	for i := range metricsStates {
		bulkRequests[i] = buildPredictionRequest(
			endpointRoleLabel,
			targetEndpointsMetadatas[i],
			metricsStates[i],
			prompts[i],
			generatedTokenCounts[i],
			prefixCacheScores[i],
		)
		if i < len(prefillTokensInFlights) {
			bulkRequests[i].PrefillTokensInFlight = prefillTokensInFlights[i]
		}
	}

	// Perform bulk prediction
	start := time.Now()
	bulkResponse, err := predictor.PredictBulkStrict(ctx, bulkRequests)
	duration := time.Since(start)

	if err != nil {
		logger.V(logutil.DEBUG).Error(err, "bulk prediction failed",
			"duration_ms", duration.Milliseconds(),
			"request_count", len(bulkRequests))
		return nil, err
	}

	if bulkResponse == nil {
		logger.V(logutil.DEBUG).Info("bulk prediction returned nil",
			"duration_ms", duration.Milliseconds())
		return nil, errors.New("bulk prediction returned nil result")
	}

	if predictedLatencyContext != nil {
		metrics.RecordRequestTTFTPredictionDuration(ctx, predictedLatencyContext.schedulingRequest.LLM.TargetModel, predictedLatencyContext.incomingModelName, duration.Seconds())
		metrics.RecordRequestTPOTPredictionDuration(ctx, predictedLatencyContext.schedulingRequest.LLM.TargetModel, predictedLatencyContext.incomingModelName, duration.Seconds())
	}
	// Convert to pointer slice for consistency with single prediction
	results := make([]*latencypredictor.PredictionResponse, len(bulkResponse.Predictions))
	for i := range bulkResponse.Predictions {
		results[i] = &bulkResponse.Predictions[i]
	}

	logger.V(logutil.DEBUG).Info("bulk prediction succeeded",
		"duration_ms", duration.Milliseconds(),
		"request_count", len(bulkRequests),
		"successful_predictions", bulkResponse.SuccessfulPredictions,
		"failed_predictions", bulkResponse.FailedPredictions,
		"processing_time_ms", bulkResponse.ProcessingTimeMs)

	// Log detailed results if at trace level
	if logger.V(logutil.TRACE).Enabled() {
		for i, result := range results {
			logger.V(logutil.TRACE).Info("bulk prediction result",
				"index", i,
				"ttft_ms", result.TTFT,
				"tpot_ms", result.TPOT,
				"input_tokens", bulkRequests[i].InputTokenLength,
				"generated_tokens", bulkRequests[i].NumTokensGenerated,
				"kv_cache_percent", bulkRequests[i].KVCachePercentage,
				"waiting_queue", bulkRequests[i].NumRequestWaiting,
				"running_requests", bulkRequests[i].NumRequestRunning,
				"prefix_cache_score", bulkRequests[i].PrefixCacheScore)
		}
	}

	return results, nil
}

// calculateRunningAverage calculates the running average efficiently
func calculateRunningAverage(currentAvg float64, newValue float64, count int) float64 {
	if count == 0 {
		return 0
	}
	if count == 1 {
		return newValue
	}
	return currentAvg + (newValue-currentAvg)/float64(count)
}
