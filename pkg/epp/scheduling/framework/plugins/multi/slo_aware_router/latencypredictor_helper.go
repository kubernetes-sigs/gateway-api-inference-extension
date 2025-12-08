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
package slo_aware_router

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/sidecars/latencypredictorasync"
)

// refreshLastSeenMetrics updates sloCtx.LastSeenMetrics from the latest scheduling result.
func refreshLastSeenMetrics(ctx context.Context, sloCtx *sloRequestContext) {
	if sr := sloCtx.schedulingResult; sr != nil {
		if pr := sr.ProfileResults[sr.PrimaryProfileName]; pr != nil && pr.TargetPods != nil {
			for profileName, profileResult := range sr.ProfileResults {
				if profileResult != nil && profileResult.TargetPods != nil && len(profileResult.TargetPods) > 0 {
					sloCtx.lastSeenMetrics[profileName] = profileResult.TargetPods[0].GetMetrics().Clone()
				}
			}
		}
	} else {
		log.FromContext(ctx).V(logutil.DEBUG).Info("No scheduling result found, skipping metrics refresh")
	}
}

// GetMetricsForPrediction retrieves the latest metrics for prediction from sloCtx.LastSeenMetrics.
func getLatestMetricsForProfile(sloCtx *sloRequestContext) (*backendmetrics.MetricsState, error) {
	if len(sloCtx.lastSeenMetrics) == 0 {
		return nil, errors.New("no last seen metrics available for prediction")
	}

	primaryProfileName := sloCtx.schedulingResult.PrimaryProfileName
	if metrics, exists := sloCtx.lastSeenMetrics[primaryProfileName]; exists {
		return metrics, nil
	}

	return nil, fmt.Errorf("no metrics found for primary profile %s", primaryProfileName)
}

// ProcessHeader refreshes metrics, applies TTFT prediction, updates sloCtx.PredictedTTFT and timestamp.
func processHeaderForLatencyPrediction(
	ctx context.Context,
	predictor latencypredictor.PredictorInterface,
	sloCtx *sloRequestContext,
) error {
	logger := log.FromContext(ctx)

	// just for debugging, print the req context scheduling result cycle state
	// print the raw scores in scheduling result

	// Build prediction request
	m, err := getLatestMetricsForProfile(sloCtx)
	if err != nil {
		logger.V(logutil.DEBUG).Info("Skipping prediction due to missing metrics", "error", err)
		return err
	}

	targetPod := sloCtx.targetPod
	prefix_cache_score := sloCtx.prefixCacheScoresForPods[targetPod.String()]

	in := latencypredictor.PredictionRequest{
		KVCachePercentage:  m.KVCacheUsagePercent,
		InputTokenLength:   len(strings.Fields(sloCtx.schedulingRequest.Body.Completions.Prompt)),
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
		sloCtx.predictedTTFT = 0
	case p == nil:
		logger.V(logutil.DEBUG).Info("header TTFT predict nil", "duration_ms", dur.Milliseconds())
		sloCtx.predictedTTFT = 0
	default:
		logger.V(logutil.DEBUG).Info("header TTFT succeeded", "value_ms", p.TTFT, "duration_ms", dur.Milliseconds())
		metrics.RecordRequestTTFTPredictionDuration(ctx, sloCtx.schedulingRequest.TargetModel, sloCtx.incomingModelName, dur.Seconds())

		sloCtx.predictedTTFT = p.TTFT
	}

	// Advance timestamp for first token reference
	sloCtx.lastTokenTimestamp = time.Now()
	refreshLastSeenMetrics(ctx, sloCtx)
	return err
}

// ProcessFirstToken records actual TTFT, trains, predicts first TPOT, updates sloCtx, and advances timestamp.
func processFirstTokenForLatencyPrediction(
	ctx context.Context,
	predictor latencypredictor.PredictorInterface,
	sloCtx *sloRequestContext,
	now time.Time,
) {
	logger := log.FromContext(ctx)

	initializeSampler(ctx, sloCtx)

	// Actual TTFT
	sloCtx.ttft = float64(now.Sub(sloCtx.requestReceivedTimestamp).Milliseconds())
	sloCtx.generatedTokenCount = 1
	m, err := getLatestMetricsForProfile(sloCtx)
	if err != nil {
		logger.V(logutil.DEBUG).Info("Skipping prediction due to missing metrics", "error", err)
		return
	}
	targetPod := sloCtx.targetPod
	prefixCacheScore := sloCtx.prefixCacheScoresForPods[targetPod.String()]

	recordTTFTTrainingData(ctx, predictor, sloCtx, m, now, prefixCacheScore)

	predictFirstTPOT(ctx, predictor, sloCtx)

	// Advance timestamp
	sloCtx.lastTokenTimestamp = now
	// Refresh metrics
	refreshLastSeenMetrics(ctx, sloCtx)
}

func initializeSampler(ctx context.Context, sloCtx *sloRequestContext) {
	if sloCtx.tokenSampler == nil {
		logger := log.FromContext(ctx)
		requestID := sloCtx.schedulingRequest.Headers[requtil.RequestIdHeaderKey]
		sloCtx.tokenSampler = newTokenSampler(requestID, DefaultSamplingMean, MaxSampledTokens)
		logger.V(logutil.DEBUG).Info("Initialized token sampler for first token", "request_id", requestID, "next_prediction_token", sloCtx.tokenSampler.getNextSampleToken())
	}
}

func recordTTFTTrainingData(
	ctx context.Context,
	predictor latencypredictor.PredictorInterface,
	sloCtx *sloRequestContext,
	m *backendmetrics.MetricsState,
	now time.Time,
	prefixCacheScore float64,
) {
	logger := log.FromContext(ctx)
	// Train TTFT
	entry := latencypredictor.TrainingEntry{
		KVCachePercentage:  m.KVCacheUsagePercent,
		InputTokenLength:   len(strings.Fields(sloCtx.schedulingRequest.Body.Completions.Prompt)),
		ActualTTFT:         sloCtx.ttft,
		ActualTPOT:         0,
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

func predictFirstTPOT(
	ctx context.Context,
	predictor latencypredictor.PredictorInterface,
	sloCtx *sloRequestContext,
) {
	logger := log.FromContext(ctx)
	m, err := getLatestMetricsForProfile(sloCtx)
	if err != nil {
		logger.V(logutil.DEBUG).Info("Skipping first TPOT prediction due to missing metrics",
			"error", err)
		return
	}

	// Predict first TPOT
	in := latencypredictor.PredictionRequest{
		KVCachePercentage:  m.KVCacheUsagePercent,
		InputTokenLength:   len(strings.Fields(sloCtx.schedulingRequest.Body.Completions.Prompt)),
		NumRequestWaiting:  m.WaitingQueueSize,
		NumRequestRunning:  m.RunningRequestsSize,
		NumTokensGenerated: sloCtx.generatedTokenCount,
		PrefixCacheScore:   0,
	}
	start := time.Now()
	p, err := predictor.Predict(ctx, in)
	dur := time.Since(start)
	if err != nil || p == nil {
		logger.V(logutil.DEBUG).Error(err, "first TPOT predict failed", "duration_ms", dur.Milliseconds())
		sloCtx.predictedTPOTObservations = append(sloCtx.predictedTPOTObservations, 0)
		sloCtx.avgPredictedTPOT = calculateRunningAverage(sloCtx.avgPredictedTPOT, 0, len(sloCtx.predictedTPOTObservations))
	} else {
		logger.V(logutil.DEBUG).Info("first TPOT succeeded", "value_ms", p.TPOT, "duration_ms", dur.Milliseconds())
		sloCtx.predictedTPOTObservations = append(sloCtx.predictedTPOTObservations, p.TPOT)
		sloCtx.avgPredictedTPOT = calculateRunningAverage(sloCtx.avgPredictedTPOT, p.TPOT, len(sloCtx.predictedTPOTObservations))
	}
	metrics.RecordRequestTPOTPredictionDuration(ctx, sloCtx.schedulingRequest.TargetModel, sloCtx.incomingModelName, dur.Seconds())
}

// ProcessToken records actual inter-token latency, trains, predicts sampled TPOT, updates sloCtx, and advances timestamp.
func processTokenForLatencyPrediction(
	ctx context.Context,
	predictor latencypredictor.PredictorInterface,
	sloCtx *sloRequestContext,
	now time.Time,
) {
	logger := log.FromContext(ctx)

	// Initialize sampler if not yet
	if sloCtx.tokenSampler == nil {
		requestID := sloCtx.schedulingRequest.Headers[requtil.RequestIdHeaderKey]
		sloCtx.tokenSampler = newTokenSampler(requestID, DefaultSamplingMean, MaxSampledTokens)
		logger.V(logutil.DEBUG).Info("Initialized token sampler for subsequent tokens", "request_id", requestID, "next_prediction_token", sloCtx.tokenSampler.getNextSampleToken())
	}

	// Inter-token latency
	latencyMs := float64(now.Sub(sloCtx.lastTokenTimestamp).Milliseconds())
	sloCtx.generatedTokenCount++

	// log the inter-token latency for predicted samples
	if sloCtx.generatedTokenCount == 2 || sloCtx.tokenSampler.shouldPredict(sloCtx.generatedTokenCount) { // tricky logic, since next sample token is always +1 from current token
		sloCtx.tpotObservations = append(sloCtx.tpotObservations, latencyMs)
		sloCtx.avgTPOT = calculateRunningAverage(sloCtx.avgTPOT, latencyMs, len(sloCtx.tpotObservations))
	}

	m, err := getLatestMetricsForProfile(sloCtx)
	if err != nil {
		logger.V(logutil.DEBUG).Info("Skipping first TPOT prediction due to missing metrics",
			"error", err)
		return
	}
	// Record actual TPOT
	entry := latencypredictor.TrainingEntry{
		KVCachePercentage:  m.KVCacheUsagePercent,
		InputTokenLength:   len(strings.Fields(sloCtx.schedulingRequest.Body.Completions.Prompt)),
		ActualTTFT:         0,
		ActualTPOT:         latencyMs,
		Timestamp:          now,
		NumRequestWaiting:  m.WaitingQueueSize,
		NumRequestRunning:  m.RunningRequestsSize,
		NumTokensGenerated: sloCtx.generatedTokenCount - 1,
		PrefixCacheScore:   0, // TPOT does not use prefix cache score
	}
	if err := predictor.AddTrainingDataBulk([]latencypredictor.TrainingEntry{entry}); err != nil {
		logger.V(logutil.DEBUG).Error(err, "record TPOT training failed")
	}

	// Sampled predict
	if sloCtx.tokenSampler.shouldPredict(sloCtx.generatedTokenCount) {
		in := latencypredictor.PredictionRequest{
			KVCachePercentage:  m.KVCacheUsagePercent,
			InputTokenLength:   len(strings.Fields(sloCtx.schedulingRequest.Body.Completions.Prompt)),
			NumRequestWaiting:  m.WaitingQueueSize,
			NumRequestRunning:  m.RunningRequestsSize,
			NumTokensGenerated: sloCtx.generatedTokenCount,
			PrefixCacheScore:   0, // TPOT does not use prefix cache score
		}
		start := time.Now()
		p, err := predictor.Predict(ctx, in)
		dur := time.Since(start)
		if err != nil || p == nil {
			logger.V(logutil.DEBUG).Error(err, "TPOT predict failed", "duration_ms", dur.Milliseconds())
			sloCtx.predictedTPOTObservations = append(sloCtx.predictedTPOTObservations, 0)
			sloCtx.avgPredictedTPOT = calculateRunningAverage(sloCtx.avgPredictedTPOT, 0, len(sloCtx.predictedTPOTObservations))
		} else {
			logger.V(logutil.DEBUG).Info("TPOT predict succeeded", "value_ms", p.TPOT, "duration_ms", dur.Milliseconds())
			sloCtx.predictedTPOTObservations = append(sloCtx.predictedTPOTObservations, p.TPOT)
			sloCtx.avgPredictedTPOT = calculateRunningAverage(sloCtx.avgPredictedTPOT, p.TPOT, len(sloCtx.predictedTPOTObservations))
		}
		metrics.RecordRequestTPOTPredictionDuration(ctx, sloCtx.schedulingRequest.TargetModel, sloCtx.incomingModelName, dur.Seconds())

		sloCtx.tokenSampler.recordPrediction(sloCtx.generatedTokenCount)
	}

	// Advance timestamp
	sloCtx.lastTokenTimestamp = now
	// Refresh metrics
	refreshLastSeenMetrics(ctx, sloCtx)
}

// bulkPredictWithMetrics performs bulk predictions for multiple pods using their metrics states.
// Returns predictions in the same order as the input slices.
func bulkPredictWithMetrics(
	ctx context.Context,
	predictor latencypredictor.PredictorInterface,
	metricsStates []*backendmetrics.MetricsState,
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
