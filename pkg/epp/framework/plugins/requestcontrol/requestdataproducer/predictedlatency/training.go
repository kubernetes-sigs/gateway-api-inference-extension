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

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/sidecars/latencypredictorasync"
)

// buildPredictionRequest constructs a prediction request from endpoint metrics and request data.
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
		InputTokenLength:   len(strings.Fields(prompt)),
		NumRequestWaiting:  metrics.WaitingQueueSize,
		NumRequestRunning:  metrics.RunningRequestsSize,
		NumTokensGenerated: generatedTokens,
		PrefixCacheScore:   prefixCacheScore,
		PodType:            podType,
	}
}

// buildTrainingEntry constructs a training entry from actual latency measurements.
func buildTrainingEntry(
	endpointRoleLabel string,
	targetEndpointMetadata *fwkdl.EndpointMetadata,
	m *fwkdl.Metrics,
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
		KVCachePercentage:  m.KVCacheUsagePercent,
		InputTokenLength:   len(strings.Fields(prompt)),
		ActualTTFT:         actualTTFT,
		ActualTPOT:         actualTPOT,
		Timestamp:          timestamp,
		NumRequestWaiting:  m.WaitingQueueSize,
		NumRequestRunning:  m.RunningRequestsSize,
		NumTokensGenerated: generatedTokens,
		PrefixCacheScore:   prefixCacheScore,
		PodType:            podType,
	}
}

// recordTTFTTrainingData sends a TTFT training entry to the predictor sidecar.
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
		0,
		now,
		0,
		prefixCacheScore,
	)
	if predictedLatencyCtx.prefillTokensAtDispatchOnPrefill > 0 {
		entry.PrefillTokensInFlight = predictedLatencyCtx.prefillTokensAtDispatchOnPrefill
	} else {
		entry.PrefillTokensInFlight = predictedLatencyCtx.prefillTokensAtDispatch
	}
	entry.DecodeTokensInFlight = predictedLatencyCtx.decodeTokensAtDispatch
	if err := predictor.AddTrainingDataBulk([]latencypredictor.TrainingEntry{entry}); err != nil {
		logger.V(logutil.DEBUG).Error(err, "record TTFT training failed")
	}
}

// refreshLastSeenMetrics updates predictedLatencyCtx.lastSeenMetrics from scheduling results.
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

// getLatestMetricsForProfile retrieves the latest metrics for the specified profile.
func getLatestMetricsForProfile(predictedLatencyCtx *predictedLatencyCtx, profileName string) (*fwkdl.Metrics, error) {
	if len(predictedLatencyCtx.lastSeenMetrics) == 0 {
		return nil, errors.New("no last seen metrics available for prediction")
	}

	if profileName == "" && predictedLatencyCtx.schedulingResult != nil {
		profileName = predictedLatencyCtx.schedulingResult.PrimaryProfileName
	}

	if m, exists := predictedLatencyCtx.lastSeenMetrics[profileName]; exists {
		return m, nil
	}

	return nil, fmt.Errorf("no metrics found for profile %s", profileName)
}

// bulkPredictWithMetrics performs bulk predictions for multiple pods using their metrics states.
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

	if len(targetEndpointsMetadatas) != len(metricsStates) || len(metricsStates) != len(prompts) || len(prompts) != len(generatedTokenCounts) || len(generatedTokenCounts) != len(prefixCacheScores) {
		return nil, fmt.Errorf("input slice lengths must match: endpoints=%d, metrics=%d, prompts=%d, tokenCounts=%d, prefixScores=%d",
			len(targetEndpointsMetadatas), len(metricsStates), len(prompts), len(generatedTokenCounts), len(prefixCacheScores))
	}

	if len(metricsStates) == 0 {
		return []*latencypredictor.PredictionResponse{}, nil
	}

	for i, metricsState := range metricsStates {
		if metricsState == nil {
			return nil, fmt.Errorf("metrics state at index %d cannot be nil", i)
		}
	}

	for i, endpointMetadata := range targetEndpointsMetadatas {
		if endpointMetadata == nil {
			return nil, fmt.Errorf("endpoint metadata at index %d cannot be nil", i)
		}
	}

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

// calculateRunningAverage calculates the running average efficiently.
func calculateRunningAverage(currentAvg float64, newValue float64, count int) float64 {
	if count == 0 {
		return 0
	}
	if count == 1 {
		return newValue
	}
	return currentAvg + (newValue-currentAvg)/float64(count)
}
