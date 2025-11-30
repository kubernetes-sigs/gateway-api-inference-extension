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

	"sigs.k8s.io/controller-runtime/pkg/log"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/sidecars/latencypredictorasync"
)

type podPredictionResult struct {
	Pod              schedulingtypes.Pod
	TTFT             float64
	TPOT             float64
	TTFTValid        bool
	TPOTValid        bool
	IsValid          bool
	Error            error
	Headroom         float64 // Headroom for the pod, if applicable
	TTFTHeadroom     float64 // TTFT headroom for the pod
	PrefixCacheScore float64 // Prefix cache score for the pod
}

// generatePredictions creates prediction results for all candidate pods
func (s *SLOAwareRouter) generatePredictions(ctx context.Context, state *schedulingtypes.CycleState, request *schedulingtypes.LLMRequest, sloCtx *sloRequestContext, candidatePods []schedulingtypes.Pod) ([]podPredictionResult, error) {
	logger := log.FromContext(ctx)
	predictions := make([]podPredictionResult, 0, len(candidatePods))

	// Prepare inputs for bulk prediction
	metricsStates := make([]*backendmetrics.MetricsState, len(candidatePods))
	prompts := make([]string, len(candidatePods))
	generatedTokenCounts := make([]int, len(candidatePods))
	prefixCacheScores := make([]float64, len(candidatePods))

	for i, pod := range candidatePods {
		logger.V(logutil.TRACE).Info("Candidate pod for scheduling", "pod", pod.GetPod().String(), "metrics", pod.GetMetrics().String())

		// Get prefix cache score for the pod
		prefixCacheScore := s.getPrefixCacheScoreForPod(ctx, state, pod)
		sloCtx.prefixCacheScoresForPods[pod.GetPod().String()] = prefixCacheScore

		logger.V(logutil.DEBUG).Info("Prefix cache score for pod", "pod", pod.GetPod().String(), "prefixCacheScore", prefixCacheScore)

		metricsStates[i] = pod.GetMetrics()
		prompts[i] = request.Body.Completions.Prompt
		generatedTokenCounts[i] = 1
		prefixCacheScores[i] = prefixCacheScore
	}

	// Bulk predict
	bulkPredictions, err := bulkPredictWithMetrics(ctx, s.latencypredictor, metricsStates, prompts, generatedTokenCounts, prefixCacheScores)
	if err != nil {
		logger.V(logutil.DEBUG).Error(err, "Bulk prediction failed")
		return nil, err
	}

	// Process results
	for i, pod := range candidatePods {
		prediction := bulkPredictions[i]
		predResult := podPredictionResult{Pod: pod}

		predResult.PrefixCacheScore = prefixCacheScores[i]
		predResult.TTFT = prediction.TTFT
		predResult.TPOT = prediction.TPOT

		podMinTPOTSLO := s.getPodMinTPOTSLO(pod)
		predResult.TTFTValid, predResult.TPOTValid, predResult.IsValid, predResult.Headroom, predResult.TTFTHeadroom = s.validatePrediction(prediction, sloCtx, podMinTPOTSLO)

		logger.V(logutil.DEBUG).Info("Prediction for scheduling",
			"pod", pod.GetPod().String(),
			"prefixCacheScore", predResult.PrefixCacheScore,
			"TTFT", prediction.TTFT,
			"TPOT", prediction.TPOT,
			"buffer", SLOBufferFactor,
			"podMinTPOTSLO", podMinTPOTSLO,
			"ttftSLO", sloCtx.ttftSLO,
			"requestTPOTSLO", sloCtx.avgTPOTSLO,
			"tpotHeadroom", predResult.Headroom,
			"ttftHeadroom", predResult.TTFTHeadroom,
			"tpotValid", predResult.TPOTValid,
			"ttftValid", predResult.TTFTValid,
			"headroomStrategy", s.headroomStrategy)

		predictions = append(predictions, predResult)
	}

	return predictions, nil
}

// updateRequestContextWithPredictions updates the request context with prediction data
func (s *SLOAwareRouter) updateRequestContextWithPredictions(sloCtx *sloRequestContext, predictions []podPredictionResult) {
	for _, pred := range predictions {
		if pred.Error == nil {
			podKey := pred.Pod.GetPod().String()
			if sloCtx.predictedTTFTForScheduling == nil {
				sloCtx.predictedTTFTForScheduling = make(map[string]float64)
			}
			if sloCtx.predictedTPOTForScheduling == nil {
				sloCtx.predictedTPOTForScheduling = make(map[string]float64)
			}
			sloCtx.predictedTTFTForScheduling[podKey] = pred.TTFT
			sloCtx.predictedTPOTForScheduling[podKey] = pred.TPOT
		}
	}
}

func (s *SLOAwareRouter) validatePrediction(
	pred *latencypredictor.PredictionResponse,
	sloCtx *sloRequestContext,
	podMinTPOTSLO float64,
) (ttftOk, tpotOk, isValid bool, headroom float64, ttftHeadroom float64) {

	bufferedTPOT := sloCtx.avgTPOTSLO * s.config.SLOBufferFactor
	// a podMinTPOTSLO of 0 means no either no requests, or no TPOT SLOs specified on running requests
	if podMinTPOTSLO > 0 {
		if podMinTPOTSLO < sloCtx.avgTPOTSLO {
			log.FromContext(context.Background()).V(logutil.DEBUG).Info("Pod min TPOT SLO is less than the req SLO, adjusting", "podMinTPOTSLO", podMinTPOTSLO, "bufferedTPOT", sloCtx.avgTPOTSLO)
		}
		bufferedTPOT = min(bufferedTPOT, podMinTPOTSLO*s.config.SLOBufferFactor)
	}

	tpotOk = pred.TPOT < bufferedTPOT
	ttftOk = pred.TTFT < sloCtx.ttftSLO

	isValid = ttftOk && tpotOk
	headroom = bufferedTPOT - pred.TPOT
	ttftHeadroom = sloCtx.ttftSLO - pred.TTFT
	return
}
