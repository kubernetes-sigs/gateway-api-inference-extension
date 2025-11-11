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
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/sidecars/latencypredictorasync"
)

type PodPredictionResult struct {
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
func (s *SLOAwareRouter) generatePredictions(ctx context.Context, state *schedulingtypes.CycleState, request *schedulingtypes.LLMRequest, sloCtx *SLORequestContext, candidatePods []schedulingtypes.Pod) ([]PodPredictionResult, error) {
	logger := log.FromContext(ctx)
	predictions := make([]PodPredictionResult, 0, len(candidatePods))

	for _, pod := range candidatePods {
		predResult := PodPredictionResult{Pod: pod}

		logger.V(logutil.TRACE).Info("Candidate pod for scheduling", "pod", pod.GetPod().String(), "metrics", pod.GetMetrics().String())

		// Get prefix cache score for the pod
		prefixCacheScore := s.getPrefixCacheScoreForPod(ctx, state, pod)

		sloCtx.PrefixCacheScoresForPods[pod.GetPod().String()] = prefixCacheScore

		logger.V(logutil.DEBUG).Info("Prefix cache score for pod", "pod", pod.GetPod().String(), "prefixCacheScore", prefixCacheScore)

		// Generate prediction
		prediction, err := PredictWithMetrics(ctx, s.latencypredictor, pod.GetMetrics(), request.Body.Completions.Prompt, 1, prefixCacheScore)
		if err != nil {
			logger.V(logutil.DEBUG).Error(err, "Skipping pod due to prediction error", "pod", pod.GetPod().String(), "error", err)
			predResult.Error = err
			return nil, err
		}
		predResult.PrefixCacheScore = prefixCacheScore
		predResult.TTFT = prediction.TTFT
		predResult.TPOT = prediction.TPOT
		podMinTPOTSLO := 0.0
		//if pod.GetPod().RunningRequests.Peek() != nil {
		//	podMinTPOTSLO = pod.GetPod().RunningRequests.Peek().TPOT
		//}
		// Do this:
		podMinTPOTSLO = s.getPodMinTPOTSLO(pod)
		predResult.TTFTValid, predResult.TPOTValid, predResult.IsValid, predResult.Headroom, predResult.TTFTHeadroom = s.validatePrediction(prediction, sloCtx, podMinTPOTSLO)

		logger.V(logutil.DEBUG).Info("Prediction for scheduling",
			"pod", pod.GetPod().String(),
			"prefixCacheScore", prefixCacheScore,
			"TTFT", prediction.TTFT,
			"TPOT", prediction.TPOT,
			"buffer", SLOBufferFactor,
			"podMinTPOTSLO", podMinTPOTSLO,
			"ttftSLO", sloCtx.TTFTSLO,
			"requestTPOTSLO", sloCtx.AvgTPOTSLO,
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
func (s *SLOAwareRouter) updateRequestContextWithPredictions(sloCtx *SLORequestContext, predictions []PodPredictionResult) {
	for _, pred := range predictions {
		if pred.Error == nil {
			podKey := pred.Pod.GetPod().String()
			if sloCtx.PredictedTTFTForScheduling == nil {
				sloCtx.PredictedTTFTForScheduling = make(map[string]float64)
			}
			if sloCtx.PredictedTPOTForScheduling == nil {
				sloCtx.PredictedTPOTForScheduling = make(map[string]float64)
			}
			sloCtx.PredictedTTFTForScheduling[podKey] = pred.TTFT
			sloCtx.PredictedTPOTForScheduling[podKey] = pred.TPOT
		}
	}
}

func (s *SLOAwareRouter) validatePrediction(
	pred *latencypredictor.PredictionResponse,
	sloCtx *SLORequestContext,
	podMinTPOTSLO float64,
) (ttftOk, tpotOk, isValid bool, headroom float64, ttftHeadroom float64) {

	bufferedTPOT := sloCtx.AvgTPOTSLO * SLOBufferFactor
	// a podMinTPOTSLO of 0 means no either no requests, or no TPOT SLOs specified on running requests
	if podMinTPOTSLO > 0 {
		if podMinTPOTSLO < sloCtx.AvgTPOTSLO {
			//print debug message
			log.FromContext(context.Background()).V(logutil.DEBUG).Info("Pod min TPOT SLO is less than the req SLO, adjusting", "podMinTPOTSLO", podMinTPOTSLO, "bufferedTPOT", sloCtx.AvgTPOTSLO)
		}
		bufferedTPOT = min(bufferedTPOT, podMinTPOTSLO*SLOBufferFactor)
	}

	tpotOk = pred.TPOT < bufferedTPOT
	ttftOk = pred.TTFT < sloCtx.TTFTSLO

	isValid = ttftOk && tpotOk
	headroom = bufferedTPOT - pred.TPOT
	ttftHeadroom = sloCtx.TTFTSLO - pred.TTFT
	return
}
