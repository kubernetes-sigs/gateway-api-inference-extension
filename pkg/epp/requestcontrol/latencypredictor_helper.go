/*
© 2025 The Kubernetes Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License.
*/

// Package requestcontrol contains helpers to decouple latency-predictor logic.
package requestcontrol

import (
	"context"
	"fmt"
	"strings"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/handlers"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/latencypredictorasync"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
)

// RefreshLastSeenMetrics updates reqCtx.LastSeenMetrics from the latest scheduling result.
func RefreshLastSeenMetrics(ctx context.Context, reqCtx *handlers.RequestContext) {
	if sr := reqCtx.SchedulingResult; sr != nil {
		if pr := sr.ProfileResults[sr.PrimaryProfileName]; pr != nil && pr.TargetPods != nil {
			for profileName, profileResult := range sr.ProfileResults {
				if profileResult != nil && profileResult.TargetPods != nil && len(profileResult.TargetPods) > 0 {
					reqCtx.LastSeenMetrics[profileName] = profileResult.TargetPods[0].GetMetrics().Clone()
				}
			}
		}
	} else {
		log.FromContext(ctx).V(logutil.DEBUG).Info("No scheduling result found, skipping metrics refresh")
	}
}

// GetTargetPodForProfile retrieves the target pod for a given profile.
// If profile is empty or not found, it uses the primary profile. Returns nil if not found.
func GetTargetPodForProfile(
	ctx context.Context,
	schedulingResult *schedulingtypes.SchedulingResult,
	profile string,
) schedulingtypes.Pod {
	logger := log.FromContext(ctx)

	if schedulingResult == nil || schedulingResult.ProfileResults == nil {
		logger.V(logutil.DEBUG).Info("No scheduling result available for target pod lookup")
		return nil
	}

	// Always fallback to primary profile if profile not specified or not found
	targetProfile := profile
	if targetProfile == "" {
		targetProfile = schedulingResult.PrimaryProfileName
	}

	// Get the profile result, fallback to primary if not found
	profileResult, exists := schedulingResult.ProfileResults[targetProfile]
	if !exists || profileResult == nil {
		logger.V(logutil.DEBUG).Info("Profile not found, using primary profile",
			"requested_profile", targetProfile,
			"primary_profile", schedulingResult.PrimaryProfileName)
		targetProfile = schedulingResult.PrimaryProfileName
		profileResult, exists = schedulingResult.ProfileResults[targetProfile]
		if !exists || profileResult == nil {
			logger.V(logutil.DEBUG).Info("Primary profile also not found",
				"primary_profile", targetProfile)
			return nil
		}
	}

	// Check if target pods exist for this profile
	if len(profileResult.TargetPods) == 0 {
		logger.V(logutil.DEBUG).Info("No target pods found for profile",
			"profile", targetProfile)
		return nil
	}

	// Return the first target pod (typically there's only one)
	targetPod := profileResult.TargetPods[0]
	podInfo := targetPod.GetPod()

	logger.V(logutil.DEBUG).Info("Found target pod for profile",
		"pod", fmt.Sprintf("%s/%s", podInfo.NamespacedName.Name, podInfo.NamespacedName.Namespace),
		"profile", targetProfile,
		"requested_profile", profile)

	return targetPod
}

// GetMetricsForPrediction retrieves the latest metrics for prediction from reqCtx.LastSeenMetrics.
func GetLatestMetricsForProfile(ctx context.Context, reqCtx *handlers.RequestContext, profileName string) (*backendmetrics.MetricsState, error) {
	if len(reqCtx.LastSeenMetrics) == 0 {
		return nil, fmt.Errorf("no last seen metrics available for prediction")
	}

	// Use the primary profile's metrics for prediction
	if metrics, exists := reqCtx.LastSeenMetrics[profileName]; exists {
		return metrics, nil
	}

	log.FromContext(ctx).V(logutil.DEBUG).Info("No metrics found for profile, trying primary profile", "profile_name", profileName)

	primaryProfileName := reqCtx.SchedulingResult.PrimaryProfileName
	if metrics, exists := reqCtx.LastSeenMetrics[primaryProfileName]; exists {
		return metrics, nil
	}

	return nil, fmt.Errorf("no metrics found for primary profile %s", primaryProfileName)
}

// ProcessHeader refreshes metrics, applies TTFT prediction, updates reqCtx.PredictedTTFT and timestamp.
func ProcessHeaderForLatencyPrediction(
	ctx context.Context,
	predictor latencypredictor.PredictorInterface,
	reqCtx *handlers.RequestContext,
) error {
	logger := log.FromContext(ctx)

	// Refresh metrics
	RefreshLastSeenMetrics(ctx, reqCtx)
	//DebugPrintRawScores(ctx, reqCtx)

	//just for debugging, print the req context scheduling result cycle state
	//print the raw scores in scheduling result

	// Build prediction request
	//check if prefill profile name is set, if not use primary profile name
	m, err := GetLatestMetricsForProfile(ctx, reqCtx, "prefill")
	if err != nil {
		logger.V(logutil.DEBUG).Info("Skipping prediction due to missing metrics", "error", err)
		return err
	}

	targetPod := GetTargetPodForProfile(ctx, reqCtx.SchedulingResult, "prefill")
	prefix_cache_score := GetPrefixCacheScoreForPod(ctx, reqCtx.SchedulingResult, targetPod, "prefill")

	in := latencypredictor.PredictionRequest{
		KVCachePercentage:  m.KVCacheUsagePercent,
		InputTokenLength:   len(strings.Fields(reqCtx.Prompt)),
		NumRequestWaiting:  m.WaitingQueueSize,
		NumRequestRunning:  m.RunningQueueSize,
		NumTokensGenerated: 0,
		PrefixCacheScore:   prefix_cache_score,
	}

	// Predict TTFT
	start := time.Now()
	p, err := predictor.Predict(ctx, in)
	dur := time.Since(start)
	if err != nil {
		logger.V(logutil.DEBUG).Error(err, "header TTFT predict failed", "duration_ms", dur.Milliseconds())
		reqCtx.PredictedTTFT = 0
	} else if p == nil {
		logger.V(logutil.DEBUG).Info("header TTFT predict nil", "duration_ms", dur.Milliseconds())
		reqCtx.PredictedTTFT = 0
	} else {
		logger.V(logutil.DEBUG).Info("header TTFT succeeded", "value_ms", p.TTFT, "duration_ms", dur.Milliseconds())
		metrics.RecordRequestTTFTPredictionDuration(ctx, reqCtx.TargetModelName, reqCtx.IncomingModelName, dur.Seconds())

		reqCtx.PredictedTTFT = p.TTFT
	}

	// Advance timestamp for first token reference
	reqCtx.LastTokenTimestamp = time.Now()
	return err
}

// ProcessFirstToken records actual TTFT, trains, predicts first TPOT, updates reqCtx, and advances timestamp.
func ProcessFirstTokenForLatencyPrediction(
	ctx context.Context,
	predictor latencypredictor.PredictorInterface,
	reqCtx *handlers.RequestContext,
	now time.Time,
) {
	logger := log.FromContext(ctx)

	// Initialize sampler
	if reqCtx.TokenSampler == nil {
		requestID := reqCtx.Request.Headers[requtil.RequestIdHeaderKey]
		reqCtx.TokenSampler = requtil.NewTokenSampler(requestID, defaultSamplingMean, maxSampledTokens)
		logger.V(logutil.DEBUG).Info("Initialized token sampler for first token", "request_id", requestID, "next_prediction_token", reqCtx.TokenSampler.GetNextSampleToken())
	}

	// Actual TTFT
	reqCtx.TTFT = float64(now.Sub(reqCtx.RequestReceivedTimestamp).Milliseconds())
	reqCtx.GeneratedTokenCount = 1
	m, err := GetLatestMetricsForProfile(ctx, reqCtx, "prefill")
	if err != nil {
		logger.V(logutil.DEBUG).Info("Skipping prediction due to missing metrics", "error", err)
		return
	}
	targetPod := GetTargetPodForProfile(ctx, reqCtx.SchedulingResult, "prefill")
	prefix_cache_score := GetPrefixCacheScoreForPod(ctx, reqCtx.SchedulingResult, targetPod, "prefill")

	// Train TTFT
	entry := latencypredictor.TrainingEntry{
		KVCachePercentage:  m.KVCacheUsagePercent,
		InputTokenLength:   len(strings.Fields(reqCtx.Prompt)),
		ActualTTFT:         reqCtx.TTFT,
		ActualTPOT:         0,
		Timestamp:          now,
		NumRequestWaiting:  m.WaitingQueueSize,
		NumRequestRunning:  m.RunningQueueSize,
		NumTokensGenerated: 0,
		PrefixCacheScore:   prefix_cache_score,
	}
	if err := predictor.AddTrainingDataBulk([]latencypredictor.TrainingEntry{entry}); err != nil {
		logger.V(logutil.DEBUG).Error(err, "record TTFT training failed")
	}
	m, err = GetLatestMetricsForProfile(ctx, reqCtx, reqCtx.SchedulingResult.PrimaryProfileName)
	if err != nil {
		logger.V(logutil.DEBUG).Info("Skipping first TPOT prediction due to missing metrics",
			"error", err)
		return
	}

	// Predict first TPOT
	in := latencypredictor.PredictionRequest{
		KVCachePercentage:  m.KVCacheUsagePercent,
		InputTokenLength:   len(strings.Fields(reqCtx.Prompt)),
		NumRequestWaiting:  m.WaitingQueueSize,
		NumRequestRunning:  m.RunningQueueSize,
		NumTokensGenerated: reqCtx.GeneratedTokenCount,
		PrefixCacheScore:   0,
	}
	start := time.Now()
	p, err := predictor.Predict(ctx, in)
	dur := time.Since(start)
	if err != nil || p == nil {
		logger.V(logutil.DEBUG).Error(err, "first TPOT predict failed", "duration_ms", dur.Milliseconds())
		reqCtx.PredictedTPOTObservations = append(reqCtx.PredictedTPOTObservations, 0)
		reqCtx.AvgPredictedTPOT = calculateRunningAverage(reqCtx.AvgPredictedTPOT, 0, len(reqCtx.PredictedTPOTObservations))
	} else {
		logger.V(logutil.DEBUG).Info("first TPOT succeeded", "value_ms", p.TPOT, "duration_ms", dur.Milliseconds())
		reqCtx.PredictedTPOTObservations = append(reqCtx.PredictedTPOTObservations, p.TPOT)
		reqCtx.AvgPredictedTPOT = calculateRunningAverage(reqCtx.AvgPredictedTPOT, p.TPOT, len(reqCtx.PredictedTPOTObservations))
	}
	metrics.RecordRequestTPOTPredictionDuration(ctx, reqCtx.TargetModelName, reqCtx.IncomingModelName, dur.Seconds())

	// Advance timestamp
	reqCtx.LastTokenTimestamp = now
	// Refresh metrics
	RefreshLastSeenMetrics(ctx, reqCtx)
}

// ProcessToken records actual inter-token latency, trains, predicts sampled TPOT, updates reqCtx, and advances timestamp.
func ProcessTokenForLatencyPrediction(
	ctx context.Context,
	predictor latencypredictor.PredictorInterface,
	reqCtx *handlers.RequestContext,
	now time.Time,
) {
	logger := log.FromContext(ctx)

	// Initialize sampler if not yet
	if reqCtx.TokenSampler == nil {
		requestID := reqCtx.Request.Headers[requtil.RequestIdHeaderKey]
		reqCtx.TokenSampler = requtil.NewTokenSampler(requestID, defaultSamplingMean, maxSampledTokens)
		logger.V(logutil.DEBUG).Info("Initialized token sampler for subsequent tokens", "request_id", requestID, "next_prediction_token", reqCtx.TokenSampler.GetNextSampleToken())
	}

	// Inter-token latency
	latencyMs := float64(now.Sub(reqCtx.LastTokenTimestamp).Milliseconds())
	reqCtx.GeneratedTokenCount++

	//log the inter-token latency for predicted samples
	if reqCtx.GeneratedTokenCount == 2 || reqCtx.TokenSampler.ShouldPredict(reqCtx.GeneratedTokenCount) { //tricky logic, since next sample token is always +1 from current token
		reqCtx.TPOTObservations = append(reqCtx.TPOTObservations, latencyMs)
		reqCtx.AvgTPOT = calculateRunningAverage(reqCtx.AvgTPOT, latencyMs, len(reqCtx.TPOTObservations))
	}

	m, err := GetLatestMetricsForProfile(ctx, reqCtx, reqCtx.SchedulingResult.PrimaryProfileName)
	if err != nil {
		logger.V(logutil.DEBUG).Info("Skipping first TPOT prediction due to missing metrics",
			"error", err)
		return
	}
	// Record actual TPOT
	entry := latencypredictor.TrainingEntry{
		KVCachePercentage:  m.KVCacheUsagePercent,
		InputTokenLength:   len(strings.Fields(reqCtx.Prompt)),
		ActualTTFT:         0,
		ActualTPOT:         latencyMs,
		Timestamp:          now,
		NumRequestWaiting:  m.WaitingQueueSize,
		NumRequestRunning:  m.RunningQueueSize,
		NumTokensGenerated: reqCtx.GeneratedTokenCount - 1,
		PrefixCacheScore:   0, // TPOT does not use prefix cache score
	}
	if err := predictor.AddTrainingDataBulk([]latencypredictor.TrainingEntry{entry}); err != nil {
		logger.V(logutil.DEBUG).Error(err, "record TPOT training failed")
	}

	// Sampled predict
	if reqCtx.TokenSampler.ShouldPredict(reqCtx.GeneratedTokenCount) {
		in := latencypredictor.PredictionRequest{
			KVCachePercentage:  m.KVCacheUsagePercent,
			InputTokenLength:   len(strings.Fields(reqCtx.Prompt)),
			NumRequestWaiting:  m.WaitingQueueSize,
			NumRequestRunning:  m.RunningQueueSize,
			NumTokensGenerated: reqCtx.GeneratedTokenCount,
			PrefixCacheScore:   0, // TPOT does not use prefix cache score
		}
		start := time.Now()
		p, err := predictor.Predict(ctx, in)
		dur := time.Since(start)
		if err != nil || p == nil {
			logger.V(logutil.DEBUG).Error(err, "TPOT predict failed", "duration_ms", dur.Milliseconds())
			reqCtx.PredictedTPOTObservations = append(reqCtx.PredictedTPOTObservations, 0)
			reqCtx.AvgPredictedTPOT = calculateRunningAverage(reqCtx.AvgPredictedTPOT, 0, len(reqCtx.PredictedTPOTObservations))
		} else {
			logger.V(logutil.DEBUG).Info("TPOT predict succeeded", "value_ms", p.TPOT, "duration_ms", dur.Milliseconds())
			reqCtx.PredictedTPOTObservations = append(reqCtx.PredictedTPOTObservations, p.TPOT)
			reqCtx.AvgPredictedTPOT = calculateRunningAverage(reqCtx.AvgPredictedTPOT, p.TPOT, len(reqCtx.PredictedTPOTObservations))
		}
		metrics.RecordRequestTPOTPredictionDuration(ctx, reqCtx.TargetModelName, reqCtx.IncomingModelName, dur.Seconds())

		reqCtx.TokenSampler.RecordPrediction(reqCtx.GeneratedTokenCount)
	}

	// Advance timestamp
	reqCtx.LastTokenTimestamp = now
	// Refresh metrics
	RefreshLastSeenMetrics(ctx, reqCtx)
}

// PredictWithMetrics predicts TTFT or TPOT based on provided metrics state and token count.
func PredictWithMetrics(
	ctx context.Context,
	predictor latencypredictor.PredictorInterface,
	metricsState *backendmetrics.MetricsState,
	prompt string,
	generatedTokenCount int,
	prefixcachescore float64,
) (*latencypredictor.PredictionResponse, error) {
	logger := log.FromContext(ctx)

	if metricsState == nil {
		return nil, fmt.Errorf("metrics state cannot be nil")
	}

	// Build prediction request
	in := latencypredictor.PredictionRequest{
		KVCachePercentage:  metricsState.KVCacheUsagePercent,
		InputTokenLength:   len(strings.Fields(prompt)),
		NumRequestWaiting:  metricsState.WaitingQueueSize,
		NumRequestRunning:  metricsState.RunningQueueSize,
		NumTokensGenerated: generatedTokenCount,
		PrefixCacheScore:   prefixcachescore,
	}

	// Perform prediction
	start := time.Now()
	result, err := predictor.Predict(ctx, in)
	duration := time.Since(start)

	if err != nil {
		logger.V(logutil.DEBUG).Error(err, "prediction failed",
			"duration_ms", duration.Milliseconds(),
			"input_tokens", in.InputTokenLength,
			"generated_tokens", generatedTokenCount,
			"kv_cache_percent", in.KVCachePercentage,
			"waiting_queue", in.NumRequestWaiting,
			"running_queue", in.NumRequestRunning,
			"prefix_cache_score", in.PrefixCacheScore)
		return nil, err
	}

	if result == nil {
		logger.V(logutil.DEBUG).Info("prediction returned nil",
			"duration_ms", duration.Milliseconds())
		return nil, fmt.Errorf("prediction returned nil result")
	}

	logger.V(logutil.DEBUG).Info("prediction succeeded",
		"tpot_ms", result.TPOT,
		"ttft_ms", result.TTFT,
		"duration_ms", duration.Milliseconds(),
		"input_tokens", in.InputTokenLength,
		"generated_tokens", generatedTokenCount,
		"kv_cache_percent", in.KVCachePercentage,
		"waiting_queue", in.NumRequestWaiting,
		"running_queue", in.NumRequestRunning,
		"prefix_cache_score", in.PrefixCacheScore)

	return result, nil
}

// Fixed DebugPrintRawScores for map[string]map[Pod]float64 structure
func DebugPrintRawScores(ctx context.Context, reqCtx *handlers.RequestContext) {
	logger := log.FromContext(ctx)

	if reqCtx.SchedulingResult == nil || reqCtx.SchedulingResult.AllProfileRunResults == nil {
		logger.V(logutil.DEBUG).Info("No raw scheduling results available for debug")
		return
	}

	logger.V(logutil.DEBUG).Info("=== RAW SCHEDULING RESULTS DEBUG START ===",
		"total_profiles", len(reqCtx.SchedulingResult.AllProfileRunResults))

	// Print raw results for all profiles
	for profileName, profileResult := range reqCtx.SchedulingResult.AllProfileRunResults {
		if profileResult == nil {
			logger.V(logutil.DEBUG).Info("Profile result is nil", "profile", profileName)
			continue
		}

		// Get the target pod (selected pod) for this profile
		var targetPodName string
		if len(profileResult.TargetPods) > 0 {
			targetPod := profileResult.TargetPods[0].GetPod()
			targetPodName = fmt.Sprintf("%s/%s", targetPod.NamespacedName.Name, targetPod.NamespacedName.Namespace)
		} else {
			targetPodName = "NO_TARGET_POD_SELECTED"
		}

		logger.V(logutil.DEBUG).Info("Raw Profile",
			"profile", profileName,
			"target_pod", targetPodName,
			"target_pod_count", len(profileResult.TargetPods))

		// Check if raw scores are available for this profile
		if len(profileResult.RawScores) == 0 {
			logger.V(logutil.DEBUG).Info("No raw scores available for profile",
				"profile", profileName)
			continue
		}

		// Print scores for each scorer type
		totalScorers := 0
		for scorerType, podScores := range profileResult.RawScores {
			totalScorers++

			// Convert to loggable format and identify target pod score
			loggableScores := make(map[string]float64)
			var targetPodScore float64
			var targetPodFound bool

			for pod, score := range podScores {
				podKey := fmt.Sprintf("%s/%s", pod.GetPod().NamespacedName.Name, pod.GetPod().NamespacedName.Namespace)
				loggableScores[podKey] = score

				// Check if this is the target pod
				if podKey == targetPodName {
					targetPodScore = score
					targetPodFound = true
				}
			}

			// Log all scores for this scorer
			logger.V(logutil.DEBUG).Info("Scorer raw scores",
				"profile", profileName,
				"scorer_type", scorerType,
				"all_scores", loggableScores,
				"pod_count", len(podScores))

			// Highlight target pod score for this scorer
			if targetPodFound {
				logger.V(logutil.DEBUG).Info("Target pod score for scorer",
					"profile", profileName,
					"scorer_type", scorerType,
					"target_pod", targetPodName,
					"score", targetPodScore)
			} else if len(profileResult.TargetPods) > 0 {
				logger.V(logutil.DEBUG).Info("Target pod not found in scorer scores",
					"profile", profileName,
					"scorer_type", scorerType,
					"target_pod", targetPodName)
			}
		}

		// Profile summary
		logger.V(logutil.DEBUG).Info("Profile Summary",
			"profile", profileName,
			"target_pod", targetPodName,
			"total_scorers", totalScorers,
			"total_scorer_types", len(profileResult.RawScores))
	}

	logger.V(logutil.DEBUG).Info("=== RAW SCHEDULING RESULTS DEBUG END ===")
}

// GetPrefixCacheScoreForPod retrieves the prefix cache score for a given pod and profile.
// If profile is empty or not found, it uses the primary profile. Returns 0.0 if not found.
func GetPrefixCacheScoreForPod(
	ctx context.Context,
	schedulingResult *schedulingtypes.SchedulingResult,
	targetPod schedulingtypes.Pod,
	profile string,
) float64 {
	logger := log.FromContext(ctx)

	if targetPod == nil {
		logger.V(logutil.DEBUG).Info("Target pod is nil, returning 0.0 prefix cache score")
		return 0.0
	}

	podInfo := targetPod.GetPod()
	podName := fmt.Sprintf("%s/%s", podInfo.NamespacedName.Name, podInfo.NamespacedName.Namespace)

	if schedulingResult == nil || schedulingResult.AllProfileRunResults == nil {
		logger.V(logutil.DEBUG).Info("No scheduling result available for prefix cache score lookup")
		return 0.0
	}

	// Always fallback to primary profile if profile not specified or not found
	targetProfile := profile
	if targetProfile == "" {
		targetProfile = schedulingResult.PrimaryProfileName
	}

	// Get the profile result, fallback to primary if not found
	profileResult, exists := schedulingResult.AllProfileRunResults[targetProfile]
	if !exists || profileResult == nil {
		logger.V(logutil.DEBUG).Info("Profile not found, using primary profile",
			"requested_profile", targetProfile,
			"primary_profile", schedulingResult.PrimaryProfileName)
		targetProfile = schedulingResult.PrimaryProfileName
		profileResult, exists = schedulingResult.AllProfileRunResults[targetProfile]
		if !exists || profileResult == nil {
			logger.V(logutil.DEBUG).Info("Primary profile also not found",
				"primary_profile", targetProfile)
			return 0.0
		}
	}

	// Check if prefix-cache scorer exists
	prefixCacheScores, exists := profileResult.RawScores["prefix-cache-scorer"]
	if !exists {
		logger.V(logutil.DEBUG).Info("Prefix cache scorer not found in profile",
			"profile", targetProfile)
		return 0.0
	}

	// Find the target pod in the scores - FIX: Compare name and namespace separately
	for pod, score := range prefixCacheScores {
		podInfoInScores := pod.GetPod()
		if podInfoInScores.NamespacedName.Name == podInfo.NamespacedName.Name &&
			podInfoInScores.NamespacedName.Namespace == podInfo.NamespacedName.Namespace {
			logger.V(logutil.DEBUG).Info("Found prefix cache score for pod",
				"pod", podName,
				"profile", targetProfile,
				"score", score)
			return score
		}
	}

	logger.V(logutil.DEBUG).Info("Pod not found in prefix cache scores",
		"pod", podName,
		"profile", targetProfile)
	return 0.0
}
