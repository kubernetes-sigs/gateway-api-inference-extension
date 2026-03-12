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

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/sidecars/latencypredictorasync"
)

type endpointPredictionResult struct {
	Endpoint         schedulingtypes.Endpoint
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
func (s *PredictedLatency) generatePredictions(ctx context.Context, predictedLatencyCtx *predictedLatencyCtx, candidateEndpoints []schedulingtypes.Endpoint) ([]endpointPredictionResult, error) {
	logger := log.FromContext(ctx)
	predictions := make([]endpointPredictionResult, 0, len(candidateEndpoints))

	// Prepare inputs for bulk prediction
	metricsStates := make([]*fwkdl.Metrics, len(candidateEndpoints))
	targetEndpointsMetadatas := make([]*fwkdl.EndpointMetadata, len(candidateEndpoints))
	prompts := make([]string, len(candidateEndpoints))
	generatedTokenCounts := make([]int, len(candidateEndpoints))
	prefixCacheScores := make([]float64, len(candidateEndpoints))

	for i, endpoint := range candidateEndpoints {
		logger.V(logutil.TRACE).Info("Candidate pod for scheduling", "endpoint", endpoint.GetMetadata().String(), "metrics", endpoint.GetMetrics().String())

		// Get prefix cache score for the pod
		prefixCacheScore := predictedLatencyCtx.prefixCacheScoresForEndpoints[endpoint.GetMetadata().NamespacedName.Name]

		logger.V(logutil.DEBUG).Info("Prefix cache score for pod", "pod", endpoint.GetMetadata().String(), "prefixCacheScore", prefixCacheScore)

		metricsStates[i] = endpoint.GetMetrics()
		targetEndpointsMetadatas[i] = endpoint.GetMetadata()
		prompts[i] = predictedLatencyCtx.promptText
		generatedTokenCounts[i] = 1
		prefixCacheScores[i] = prefixCacheScore
	}

	// Bulk predict
	bulkPredictions, err := bulkPredictWithMetrics(ctx, predictedLatencyCtx, s.latencypredictor, metricsStates, s.config.EndpointRoleLabel, targetEndpointsMetadatas, prompts, generatedTokenCounts, prefixCacheScores)
	if err != nil {
		logger.V(logutil.DEBUG).Error(err, "Bulk prediction failed")
		return nil, err
	}

	// Process results
	for i, endpoint := range candidateEndpoints {
		prediction := bulkPredictions[i]
		predResult := endpointPredictionResult{Endpoint: endpoint}

		predResult.PrefixCacheScore = prefixCacheScores[i]
		predResult.TTFT = prediction.TTFT
		predResult.TPOT = prediction.TPOT

		podMinTPOTSLO := s.getEndpointMinTPOTSLO(endpoint)
		predResult.TTFTValid, predResult.TPOTValid, predResult.IsValid, predResult.Headroom, predResult.TTFTHeadroom = s.validatePrediction(prediction, predictedLatencyCtx, podMinTPOTSLO)

		logger.V(logutil.DEBUG).Info("Prediction for scheduling",
			"endpoint", endpoint.GetMetadata().String(),
			"prefixCacheScore", predResult.PrefixCacheScore,
			"TTFT", prediction.TTFT,
			"TPOT", prediction.TPOT,
			"buffer", s.config.SLOBufferFactor,
			"podMinTPOTSLO", podMinTPOTSLO,
			"ttftSLO", predictedLatencyCtx.ttftSLO,
			"requestTPOTSLO", predictedLatencyCtx.avgTPOTSLO,
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
func (s *PredictedLatency) updateRequestContextWithPredictions(predictedLatencyCtx *predictedLatencyCtx, predictions []endpointPredictionResult) {
	predMap := make(map[string]endpointPredictionResult, len(predictions))
	for _, pred := range predictions {
		if pred.Endpoint != nil && pred.Endpoint.GetMetadata() != nil {
			predMap[pred.Endpoint.GetMetadata().NamespacedName.Name] = pred
		}
	}
	predictedLatencyCtx.predictionsForScheduling = predMap
}

func (s *PredictedLatency) validatePrediction(
	pred *latencypredictor.PredictionResponse,
	predictedLatencyCtx *predictedLatencyCtx,
	podMinTPOTSLO float64,
) (ttftOk, tpotOk, isValid bool, headroom float64, ttftHeadroom float64) {

	ttftOk = pred.TTFT < predictedLatencyCtx.ttftSLO
	ttftHeadroom = predictedLatencyCtx.ttftSLO - pred.TTFT

	tpotOk = true
	headroom = 0.0

	if s.config.StreamingMode {
		bufferedTPOT := predictedLatencyCtx.avgTPOTSLO * s.config.SLOBufferFactor
		// a podMinTPOTSLO of 0 means no either no requests, or no TPOT SLOs specified on running requests
		if podMinTPOTSLO > 0 {
			if podMinTPOTSLO < predictedLatencyCtx.avgTPOTSLO {
				log.FromContext(context.Background()).V(logutil.DEBUG).Info("Pod min TPOT SLO is less than the req SLO, adjusting", "podMinTPOTSLO", podMinTPOTSLO, "bufferedTPOT", predictedLatencyCtx.avgTPOTSLO)
			}
			bufferedTPOT = min(bufferedTPOT, podMinTPOTSLO*s.config.SLOBufferFactor)
		}

		tpotOk = pred.TPOT < bufferedTPOT
		headroom = bufferedTPOT - pred.TPOT
	}

	isValid = ttftOk && tpotOk

	return
}
