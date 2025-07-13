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

// GetMetricsForPrediction retrieves the latest metrics for prediction from reqCtx.LastSeenMetrics.
func GetLatestMetricsForProfile(ctx context.Context, reqCtx *handlers.RequestContext, profileName string) (*backendmetrics.MetricsState, error) {
    if len(reqCtx.LastSeenMetrics) == 0 {
		return nil, fmt.Errorf("no last seen metrics available for prediction")
	}

	// Use the primary profile's metrics for prediction
	if metrics, exists := reqCtx.LastSeenMetrics[profileName]; exists {
		return metrics, nil
	}

    log.FromContext(ctx).V(logutil.DEBUG).Info("No metrics found for profile", "profile_name", profileName, "trying primary profile")

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

    //just for debugging, print the req context scheduling result cycle state
    logger.V(logutil.DEBUG).Info("Processing header for latency prediction", "scheduling_result", reqCtx.SchedulingResult,
        "cycle_state", reqCtx.SchedulingCycleState)

    // Build prediction request
	//check if prefill profile name is set, if not use primary profile name
	m, err := GetLatestMetricsForProfile(ctx, reqCtx, "prefill")
	if err != nil {
		logger.V(logutil.DEBUG).Info("Skipping prediction due to missing metrics", "error", err)
		return err
	}

	in := latencypredictor.PredictionRequest{
        KVCachePercentage:  m.KVCacheUsagePercent,
        InputTokenLength:   len(strings.Fields(reqCtx.Prompt)),
        NumRequestWaiting:  m.WaitingQueueSize,
        NumRequestRunning:  m.RunningQueueSize,
        NumTokensGenerated: 0,
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
            "running_queue", in.NumRequestRunning)
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
        "running_queue", in.NumRequestRunning)

    return result, nil
}