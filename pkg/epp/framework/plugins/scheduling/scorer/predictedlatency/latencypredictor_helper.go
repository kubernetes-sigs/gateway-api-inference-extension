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
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
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
func getLatestMetricsForProfile(predictedLatencyCtx *predictedLatencyCtx) (*fwkdl.Metrics, error) {
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
	requestBuilder PredictionRequestBuilder,
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

	target_endpoint_metadata := predictedLatencyCtx.targetMetadata
	prefix_cache_score := predictedLatencyCtx.prefixCacheScoresForEndpoints[target_endpoint_metadata.NamespacedName.Name]

	// Build prediction request using the builder (ensures pod type is included for P/D)
	in := requestBuilder.BuildPredictionRequest(
		ctx,
		target_endpoint_metadata,
		m,
		predictedLatencyCtx.schedulingRequest.Body.Completions.Prompt,
		0, // NumTokensGenerated is 0 for pre-request TTFT prediction
		prefix_cache_score,
	)

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

// processFirstTokenForLatencyPrediction records actual TTFT, trains, predicts first TPOT, updates predictedLatencyCtx, and advances timestamp.
func processFirstTokenForLatencyPrediction(
	ctx context.Context,
	predictor latencypredictor.PredictorInterface,
	streamingMode bool,
	requestBuilder PredictionRequestBuilder,
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
	targetEndpointMetadata := predictedLatencyCtx.targetMetadata
	prefixCacheScore := predictedLatencyCtx.prefixCacheScoresForEndpoints[targetEndpointMetadata.NamespacedName.Name]
	logger.V(logutil.DEBUG).Info("Recording TTFT training data", "ttft_ms", predictedLatencyCtx.ttft, "prefixCacheScore", prefixCacheScore)
	recordTTFTTrainingData(ctx, predictor, requestBuilder, predictedLatencyCtx, m, targetEndpointMetadata, now, prefixCacheScore)

	if streamingMode {
		predictFirstTPOT(ctx, predictor, requestBuilder, predictedLatencyCtx, targetEndpointMetadata)
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
	requestBuilder PredictionRequestBuilder,
	predictedLatencyCtx *predictedLatencyCtx,
	m *fwkdl.Metrics,
	targetEndpointMetadata *fwkdl.EndpointMetadata,
	now time.Time,
	prefixCacheScore float64,
) {
	logger := log.FromContext(ctx)
	// Build training entry using the builder
	entry := requestBuilder.BuildTrainingEntry(
		ctx,
		targetEndpointMetadata,
		m,
		predictedLatencyCtx.schedulingRequest.Body.Completions.Prompt,
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
	predictor latencypredictor.PredictorInterface,
	requestBuilder PredictionRequestBuilder,
	predictedLatencyCtx *predictedLatencyCtx,
	targetEndpointMetadata *fwkdl.EndpointMetadata,
) {
	logger := log.FromContext(ctx)
	m, err := getLatestMetricsForProfile(predictedLatencyCtx)
	if err != nil {
		logger.V(logutil.DEBUG).Info("Skipping first TPOT prediction due to missing metrics",
			"error", err)
		return
	}

	// Build prediction request using the builder (ensures pod type is included for P/D)
	in := requestBuilder.BuildPredictionRequest(
		ctx,
		targetEndpointMetadata,
		m,
		predictedLatencyCtx.schedulingRequest.Body.Completions.Prompt,
		predictedLatencyCtx.generatedTokenCount,
		0, // TPOT does not use prefix cache score
	)
	start := time.Now()
	p, err := predictor.Predict(ctx, in)
	dur := time.Since(start)
	if err != nil || p == nil {
		logger.V(logutil.DEBUG).Error(err, "first TPOT predict failed", "duration_ms", dur.Milliseconds())
		predictedLatencyCtx.predictedTPOTObservations = append(predictedLatencyCtx.predictedTPOTObservations, 0)
		predictedLatencyCtx.avgPredictedTPOT = calculateRunningAverage(predictedLatencyCtx.avgPredictedTPOT, 0, len(predictedLatencyCtx.predictedTPOTObservations))
	} else {
		logger.V(logutil.DEBUG).Info("first TPOT succeeded", "value_ms", p.TPOT, "duration_ms", dur.Milliseconds())
		predictedLatencyCtx.predictedTPOTObservations = append(predictedLatencyCtx.predictedTPOTObservations, p.TPOT)
		predictedLatencyCtx.avgPredictedTPOT = calculateRunningAverage(predictedLatencyCtx.avgPredictedTPOT, p.TPOT, len(predictedLatencyCtx.predictedTPOTObservations))
	}
	metrics.RecordRequestTPOTPredictionDuration(ctx, predictedLatencyCtx.schedulingRequest.TargetModel, predictedLatencyCtx.incomingModelName, dur.Seconds())
}

// processTokenForLatencyPrediction records actual inter-token latency, trains, predicts sampled TPOT, updates predictedLatencyCtx, and advances timestamp.
func processTokenForLatencyPrediction(
	ctx context.Context,
	predictor latencypredictor.PredictorInterface,
	requestBuilder PredictionRequestBuilder,
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

	m, err := getLatestMetricsForProfile(predictedLatencyCtx)
	if err != nil {
		logger.V(logutil.DEBUG).Info("Skipping first TPOT prediction due to missing metrics",
			"error", err)
		return
	}
	// Record actual TPOT using builder
	entry := requestBuilder.BuildTrainingEntry(
		ctx,
		targetEndpointMetadata,
		m,
		predictedLatencyCtx.schedulingRequest.Body.Completions.Prompt,
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
		// Build prediction request using the builder (ensures pod type is included for P/D)
		in := requestBuilder.BuildPredictionRequest(
			ctx,
			targetEndpointMetadata,
			m,
			predictedLatencyCtx.schedulingRequest.Body.Completions.Prompt,
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
	predictor latencypredictor.PredictorInterface,
	metricsStates []*fwkdl.Metrics,
	requestBuilder PredictionRequestBuilder,
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

	// Build bulk prediction requests using the builder
	bulkRequests := make([]latencypredictor.PredictionRequest, len(metricsStates))
	for i := range metricsStates {
		bulkRequests[i] = requestBuilder.BuildPredictionRequest(
			ctx,
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
