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
package predicted_latency

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/util/logging"
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/sidecars/latencypredictorasync"
)

// refreshLastSeenMetrics updates predictedLatencyCtx.LastSeenMetrics from the latest scheduling result.
func refreshLastSeenMetrics(ctx context.Context, predictedLatencyCtx *predictedLatencyCtx) {
	if sr := predictedLatencyCtx.schedulingResult; sr != nil {
		if pr := sr.ProfileResults[sr.PrimaryProfileName]; pr != nil && pr.TargetEndpoints != nil {
			for profileName, profileResult := range sr.ProfileResults {
				if profileResult != nil && profileResult.TargetEndpoints != nil && len(profileResult.TargetEndpoints) > 0 {
					predictedLatencyCtx.lastSeenMetrics[profileName] = profileResult.TargetEndpoints[0].GetMetrics().Clone()
				}
			}
		}
	} else {
		log.FromContext(ctx).V(logutil.DEBUG).Info("No scheduling result found, skipping metrics refresh")
	}
}

// getLatestMetricsForProfile retrieves the latest metrics for prediction from predictedLatencyCtx.LastSeenMetrics.
func getLatestMetricsForProfile(predictedLatencyCtx *predictedLatencyCtx) (*datalayer.Metrics, error) {
	if len(predictedLatencyCtx.lastSeenMetrics) == 0 {
		return nil, errors.New("no last seen metrics available for prediction")
	}

	primaryProfileName := predictedLatencyCtx.schedulingResult.PrimaryProfileName
	if metrics, exists := predictedLatencyCtx.lastSeenMetrics[primaryProfileName]; exists {
		return metrics, nil
	}

	return nil, fmt.Errorf("no metrics found for primary profile %s", primaryProfileName)
}

// processPreRequestForLatencyPrediction refreshes metrics, applies TTFT prediction, updates predictedLatencyCtx.PredictedTTFT and timestamp.
func processPreRequestForLatencyPrediction(
	ctx context.Context,
	predictor latencypredictor.PredictorInterface,
	predictedLatencyCtx *predictedLatencyCtx,
) error {
	logger := log.FromContext(ctx)

	// just for debugging, print the req context scheduling result cycle state
	// print the raw scores in scheduling result

	// Build prediction request
	m, err := getLatestMetricsForProfile(predictedLatencyCtx)
	if err != nil {
		logger.V(logutil.DEBUG).Info("Skipping prediction due to missing metrics", "error", err)
		return err
	}

	targetPod := predictedLatencyCtx.targetMetadata
	prefix_cache_score := predictedLatencyCtx.prefixCacheScoresForEndpoints[targetPod.NamespacedName.Name]

	in := latencypredictor.PredictionRequest{
		KVCachePercentage:  m.KVCacheUsagePercent,
		InputTokenLength:   len(strings.Fields(predictedLatencyCtx.schedulingRequest.Body.Completions.Prompt)),
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
		metrics.RecordRequestTTFTPredictionDuration(ctx, predictedLatencyCtx.schedulingRequest.TargetModel, predictedLatencyCtx.incomingModelName, dur.Seconds())

		predictedLatencyCtx.predictedTTFT = p.TTFT
	}

	// Advance timestamp for first token reference
	predictedLatencyCtx.lastTokenTimestamp = time.Now()
	return err
}

// processFirstTokenForLatencyPrediction records actual TTFT, trains, predicts first ITL, updates predictedLatencyCtx, and advances timestamp.
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
	m, err := getLatestMetricsForProfile(predictedLatencyCtx)
	if err != nil {
		logger.V(logutil.DEBUG).Info("Skipping prediction due to missing metrics", "error", err)
		return
	}
	targetPod := predictedLatencyCtx.targetMetadata
	prefixCacheScore := predictedLatencyCtx.prefixCacheScoresForEndpoints[targetPod.NamespacedName.Name]
	logger.V(logutil.DEBUG).Info("Recording TTFT training data", "ttft_ms", predictedLatencyCtx.ttft, "prefixCacheScore", prefixCacheScore)
	recordTTFTTrainingData(ctx, predictor, predictedLatencyCtx, m, now, prefixCacheScore)

	if streamingMode {
		predictFirstITL(ctx, predictor, predictedLatencyCtx)
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
	predictedLatencyCtx *predictedLatencyCtx,
	m *datalayer.Metrics,
	now time.Time,
	prefixCacheScore float64,
) {
	logger := log.FromContext(ctx)
	// Train TTFT
	entry := latencypredictor.TrainingEntry{
		KVCachePercentage:  m.KVCacheUsagePercent,
		InputTokenLength:   len(strings.Fields(predictedLatencyCtx.schedulingRequest.Body.Completions.Prompt)),
		ActualTTFT:         predictedLatencyCtx.ttft,
		ActualITL:         0,
		Timestamp:          now,
		NumRequestWaiting:  m.WaitingQueueSize,
		NumRequestRunning:  m.RunningRequestsSize,
		NumTokensGenerated: 0,
		PrefixCacheScore:   prefixCacheScore,
	}
	if err := predictor.AddTrainingDataBulk([]latencypredictor.TrainingEntry{entry}); err != nil {
		logger.V(logutil.DEBUG).Error(err, "record TTFT training failed")
	}
}

func predictFirstITL(
	ctx context.Context,
	predictor latencypredictor.PredictorInterface,
	predictedLatencyCtx *predictedLatencyCtx,
) {
	logger := log.FromContext(ctx)
	m, err := getLatestMetricsForProfile(predictedLatencyCtx)
	if err != nil {
		logger.V(logutil.DEBUG).Info("Skipping first ITL prediction due to missing metrics",
			"error", err)
		return
	}

	// Predict first ITL
	in := latencypredictor.PredictionRequest{
		KVCachePercentage:  m.KVCacheUsagePercent,
		InputTokenLength:   len(strings.Fields(predictedLatencyCtx.schedulingRequest.Body.Completions.Prompt)),
		NumRequestWaiting:  m.WaitingQueueSize,
		NumRequestRunning:  m.RunningRequestsSize,
		NumTokensGenerated: predictedLatencyCtx.generatedTokenCount,
		PrefixCacheScore:   0,
	}
	start := time.Now()
	p, err := predictor.Predict(ctx, in)
	dur := time.Since(start)
	if err != nil || p == nil {
		logger.V(logutil.DEBUG).Error(err, "first ITL predict failed", "duration_ms", dur.Milliseconds())
		predictedLatencyCtx.predictedITLObservations = append(predictedLatencyCtx.predictedITLObservations, 0)
		predictedLatencyCtx.avgPredictedITL = calculateRunningAverage(predictedLatencyCtx.avgPredictedITL, 0, len(predictedLatencyCtx.predictedITLObservations))
	} else {
		logger.V(logutil.DEBUG).Info("first ITL succeeded", "value_ms", p.ITL, "duration_ms", dur.Milliseconds())
		predictedLatencyCtx.predictedITLObservations = append(predictedLatencyCtx.predictedITLObservations, p.ITL)
		predictedLatencyCtx.avgPredictedITL = calculateRunningAverage(predictedLatencyCtx.avgPredictedITL, p.ITL, len(predictedLatencyCtx.predictedITLObservations))
	}
	metrics.RecordRequestITLPredictionDuration(ctx, predictedLatencyCtx.schedulingRequest.TargetModel, predictedLatencyCtx.incomingModelName, dur.Seconds())
}

// processTokenForLatencyPrediction records actual inter-token latency, trains, predicts sampled ITL, updates predictedLatencyCtx, and advances timestamp.
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
		requestID := predictedLatencyCtx.schedulingRequest.Headers[requtil.RequestIdHeaderKey]
		predictedLatencyCtx.tokenSampler = newTokenSampler(requestID, samplingMean, maxSampledTokens)
		logger.V(logutil.DEBUG).Info("Initialized token sampler for subsequent tokens", "request_id", requestID, "next_prediction_token", predictedLatencyCtx.tokenSampler.getNextSampleToken())
	}

	// Inter-token latency
	latencyMs := float64(now.Sub(predictedLatencyCtx.lastTokenTimestamp).Milliseconds())
	predictedLatencyCtx.generatedTokenCount++

	// log the inter-token latency for predicted samples
	if predictedLatencyCtx.generatedTokenCount == 2 || predictedLatencyCtx.tokenSampler.shouldPredict(predictedLatencyCtx.generatedTokenCount) { // tricky logic, since next sample token is always +1 from current token
		predictedLatencyCtx.itlObservations = append(predictedLatencyCtx.itlObservations, latencyMs)
		predictedLatencyCtx.avgITL = calculateRunningAverage(predictedLatencyCtx.avgITL, latencyMs, len(predictedLatencyCtx.itlObservations))
	}

	m, err := getLatestMetricsForProfile(predictedLatencyCtx)
	if err != nil {
		logger.V(logutil.DEBUG).Info("Skipping first ITL prediction due to missing metrics",
			"error", err)
		return
	}
	// Record actual ITL
	entry := latencypredictor.TrainingEntry{
		KVCachePercentage:  m.KVCacheUsagePercent,
		InputTokenLength:   len(strings.Fields(predictedLatencyCtx.schedulingRequest.Body.Completions.Prompt)),
		ActualTTFT:         0,
		ActualITL:         latencyMs,
		Timestamp:          now,
		NumRequestWaiting:  m.WaitingQueueSize,
		NumRequestRunning:  m.RunningRequestsSize,
		NumTokensGenerated: predictedLatencyCtx.generatedTokenCount - 1,
		PrefixCacheScore:   0, // ITL does not use prefix cache score
	}
	if err := predictor.AddTrainingDataBulk([]latencypredictor.TrainingEntry{entry}); err != nil {
		logger.V(logutil.DEBUG).Error(err, "record ITL training failed")
	}

	// Sampled predict
	if predictedLatencyCtx.tokenSampler.shouldPredict(predictedLatencyCtx.generatedTokenCount) {
		in := latencypredictor.PredictionRequest{
			KVCachePercentage:  m.KVCacheUsagePercent,
			InputTokenLength:   len(strings.Fields(predictedLatencyCtx.schedulingRequest.Body.Completions.Prompt)),
			NumRequestWaiting:  m.WaitingQueueSize,
			NumRequestRunning:  m.RunningRequestsSize,
			NumTokensGenerated: predictedLatencyCtx.generatedTokenCount,
			PrefixCacheScore:   0, // ITL does not use prefix cache score
		}
		start := time.Now()
		p, err := predictor.Predict(ctx, in)
		dur := time.Since(start)
		if err != nil || p == nil {
			logger.V(logutil.DEBUG).Error(err, "ITL predict failed", "duration_ms", dur.Milliseconds())
			predictedLatencyCtx.predictedITLObservations = append(predictedLatencyCtx.predictedITLObservations, 0)
			predictedLatencyCtx.avgPredictedITL = calculateRunningAverage(predictedLatencyCtx.avgPredictedITL, 0, len(predictedLatencyCtx.predictedITLObservations))
		} else {
			logger.V(logutil.DEBUG).Info("ITL predict succeeded", "value_ms", p.ITL, "duration_ms", dur.Milliseconds())
			predictedLatencyCtx.predictedITLObservations = append(predictedLatencyCtx.predictedITLObservations, p.ITL)
			predictedLatencyCtx.avgPredictedITL = calculateRunningAverage(predictedLatencyCtx.avgPredictedITL, p.ITL, len(predictedLatencyCtx.predictedITLObservations))
		}
		metrics.RecordRequestITLPredictionDuration(ctx, predictedLatencyCtx.schedulingRequest.TargetModel, predictedLatencyCtx.incomingModelName, dur.Seconds())

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
	predictor latencypredictor.PredictorInterface,
	metricsStates []*datalayer.Metrics,
	prompts []string,
	generatedTokenCounts []int,
	prefixCacheScores []float64,
) ([]*latencypredictor.PredictionResponse, error) {
	logger := log.FromContext(ctx)

	// Validate input lengths
	if len(metricsStates) != len(prompts) || len(prompts) != len(generatedTokenCounts) || len(generatedTokenCounts) != len(prefixCacheScores) {
		return nil, fmt.Errorf("input slice lengths must match: metrics=%d, prompts=%d, tokenCounts=%d, prefixScores=%d",
			len(metricsStates), len(prompts), len(generatedTokenCounts), len(prefixCacheScores))
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

	// Build bulk prediction requests
	bulkRequests := make([]latencypredictor.PredictionRequest, len(metricsStates))
	for i := range metricsStates {
		bulkRequests[i] = latencypredictor.PredictionRequest{
			KVCachePercentage:  metricsStates[i].KVCacheUsagePercent,
			InputTokenLength:   len(strings.Fields(prompts[i])),
			NumRequestWaiting:  metricsStates[i].WaitingQueueSize,
			NumRequestRunning:  metricsStates[i].RunningRequestsSize,
			NumTokensGenerated: generatedTokenCounts[i],
			PrefixCacheScore:   prefixCacheScores[i],
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
				"itl_ms", result.ITL,
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
