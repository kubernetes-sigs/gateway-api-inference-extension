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
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
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

// buildTrainingEntry constructs a training entry from actual latency measurements.
// If endpointRoleLabel is configured, it extracts the role from the endpoint's labels and
// populates the PodType field, enabling role-specific model training.
func buildTrainingEntry(
	endpointRoleLabel string,
	targetEndpointMetadata *fwkdl.EndpointMetadata,
	metrics *fwkdl.Metrics,
	prompt string,
	actualTTFT float64,
	actualTPOT float64,
	timestamp time.Time,
	generatedTokens int,
	prefixCacheScore float64,
) latencypredictor.TrainingEntry {
	podType := ""
	if endpointRoleLabel != "" && targetEndpointMetadata != nil && targetEndpointMetadata.Labels != nil {
		podType = targetEndpointMetadata.Labels[endpointRoleLabel]
	}

	return latencypredictor.TrainingEntry{
		KVCachePercentage:  metrics.KVCacheUsagePercent,
		InputTokenLength:   len(strings.Fields(prompt)), // Simple word-based tokenization
		ActualTTFT:         actualTTFT,
		ActualTPOT:         actualTPOT,
		Timestamp:          timestamp,
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
	predictedLatencyCtx *predictedLatencyCtx,
) {
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

	// Advance timestamp for first token reference
	predictedLatencyCtx.lastTokenTimestamp = time.Now()
}

// processFirstTokenForLatencyPrediction records actual TTFT, trains, predicts first TPOT, updates predictedLatencyCtx, and advances timestamp.
func processFirstTokenForLatencyPrediction(
	ctx context.Context,
	predictor latencypredictor.PredictorInterface,
	streamingMode bool,
	endpointRoleLabel string,
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
			recordTTFTTrainingData(ctx, predictor, endpointRoleLabel, predictedLatencyCtx, prefillMetrics, prefillTargetMetadata, now, prefillPrefixCacheScore)
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
		recordTTFTTrainingData(ctx, predictor, endpointRoleLabel, predictedLatencyCtx, m, targetEndpointMetadata, now, prefixCacheScore)
	}

	if streamingMode {
		predictFirstTPOT(ctx, predictedLatencyCtx)
	}

	// Advance timestamp
	predictedLatencyCtx.lastTokenTimestamp = now
	// Refresh metrics
	refreshLastSeenMetrics(ctx, predictedLatencyCtx)
}

func initializeSampler(ctx context.Context, predictedLatencyCtx *predictedLatencyCtx, samplingMean float64, maxSampledTokens int) {
	if predictedLatencyCtx.tokenSampler == nil {
		logger := log.FromContext(ctx)
		requestID := predictedLatencyCtx.schedulingRequest.Headers[requtil.RequestIdHeaderKey]
		predictedLatencyCtx.tokenSampler = newTokenSampler(requestID, samplingMean, maxSampledTokens)
		logger.V(logutil.DEBUG).Info("Initialized token sampler for first token", "request_id", requestID, "next_prediction_token", predictedLatencyCtx.tokenSampler.getNextSampleToken())
	}
}

func recordTTFTTrainingData(
	ctx context.Context,
	predictor latencypredictor.PredictorInterface,
	endpointRoleLabel string,
	predictedLatencyCtx *predictedLatencyCtx,
	m *fwkdl.Metrics,
	targetEndpointMetadata *fwkdl.EndpointMetadata,
	now time.Time,
	prefixCacheScore float64,
) {
	logger := log.FromContext(ctx)
	entry := buildTrainingEntry(
		endpointRoleLabel,
		targetEndpointMetadata,
		m,
		predictedLatencyCtx.promptText,
		predictedLatencyCtx.ttft,
		0, // TTFT training
		now,
		0,
		prefixCacheScore,
	)
	if err := predictor.AddTrainingDataBulk([]latencypredictor.TrainingEntry{entry}); err != nil {
		logger.V(logutil.DEBUG).Error(err, "record TTFT training failed")
	}
}

func predictFirstTPOT(
	ctx context.Context,
	predictedLatencyCtx *predictedLatencyCtx,
) {
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

// processTokenForLatencyPrediction records actual inter-token latency, trains, predicts sampled TPOT, updates predictedLatencyCtx, and advances timestamp.
func processTokenForLatencyPrediction(
	ctx context.Context,
	predictor latencypredictor.PredictorInterface,
	endpointRoleLabel string,
	predictedLatencyCtx *predictedLatencyCtx,
	targetEndpointMetadata *fwkdl.EndpointMetadata,
	now time.Time,
	samplingMean float64,
	maxSampledTokens int,
) {
	logger := log.FromContext(ctx)

	// Initialize sampler if not yet
	if predictedLatencyCtx.tokenSampler == nil {
		requestID := predictedLatencyCtx.schedulingRequest.Headers[requtil.RequestIdHeaderKey]
		predictedLatencyCtx.tokenSampler = newTokenSampler(requestID, samplingMean, maxSampledTokens)
		logger.V(logutil.DEBUG).Info("Initialized token sampler for subsequent tokens", "request_id", requestID, "next_prediction_token", predictedLatencyCtx.tokenSampler.getNextSampleToken())
	}

	// Inter-token latency
	latencyMs := float64(now.Sub(predictedLatencyCtx.lastTokenTimestamp).Milliseconds())
	predictedLatencyCtx.generatedTokenCount++

	// log the inter-token latency for predicted samples
	if predictedLatencyCtx.generatedTokenCount == 2 || predictedLatencyCtx.tokenSampler.shouldPredict(predictedLatencyCtx.generatedTokenCount) { // tricky logic, since next sample token is always +1 from current token
		predictedLatencyCtx.tpotObservations = append(predictedLatencyCtx.tpotObservations, latencyMs)
		predictedLatencyCtx.avgTPOT = calculateRunningAverage(predictedLatencyCtx.avgTPOT, latencyMs, len(predictedLatencyCtx.tpotObservations))
	}
	if predictedLatencyCtx.generatedTokenCount == 2 {
		// debug log actual and predicted tpot
		logger.V(logutil.DEBUG).Info("First inter-token latency observed",
			"actual_tpot_ms", latencyMs,
			"predicted_tpot_ms", predictedLatencyCtx.avgPredictedTPOT)
	}

	m, err := getLatestMetricsForProfile(predictedLatencyCtx, "")
	if err != nil {
		logger.V(logutil.DEBUG).Info("Skipping TPOT training due to missing metrics or schedulingResult",
			"error", err)
		return
	}
	entry := buildTrainingEntry(
		endpointRoleLabel,
		targetEndpointMetadata,
		m,
		predictedLatencyCtx.promptText,
		0, // TTFT not recorded for TPOT
		latencyMs,
		now,
		predictedLatencyCtx.generatedTokenCount-1,
		0, // TPOT does not use prefix cache score
	)
	if err := predictor.AddTrainingDataBulk([]latencypredictor.TrainingEntry{entry}); err != nil {
		logger.V(logutil.DEBUG).Error(err, "record TPOT training failed")
	}

	// Sampled predict
	if predictedLatencyCtx.tokenSampler.shouldPredict(predictedLatencyCtx.generatedTokenCount) {
		in := buildPredictionRequest(
			endpointRoleLabel,
			targetEndpointMetadata,
			m,
			predictedLatencyCtx.promptText,
			predictedLatencyCtx.generatedTokenCount,
			0, // TPOT does not use prefix cache score
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
		metrics.RecordRequestTTFTPredictionDuration(ctx, predictedLatencyContext.schedulingRequest.TargetModel, predictedLatencyContext.incomingModelName, duration.Seconds())
		metrics.RecordRequestTPOTPredictionDuration(ctx, predictedLatencyContext.schedulingRequest.TargetModel, predictedLatencyContext.incomingModelName, duration.Seconds())
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
