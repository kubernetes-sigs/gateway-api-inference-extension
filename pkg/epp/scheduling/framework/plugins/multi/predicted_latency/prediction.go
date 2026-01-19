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

	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/util/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/sidecars/latencypredictorasync"
)

type endpointPredictionResult struct {
	Endpoint         schedulingtypes.Endpoint
	TTFT             float64
	ITL              float64
	TTFTValid        bool
	ITLValid         bool
	IsValid          bool
	Error            error
	Headroom         float64 // Headroom for the pod, if applicable
	TTFTHeadroom     float64 // TTFT headroom for the pod
	PrefixCacheScore float64 // Prefix cache score for the pod
}

// generatePredictions creates prediction results for all candidate pods
func (s *PredictedLatency) generatePredictions(ctx context.Context, request *schedulingtypes.LLMRequest, predictedLatencyCtx *predictedLatencyCtx, candidateEndpoints []schedulingtypes.Endpoint) ([]endpointPredictionResult, error) {
	logger := log.FromContext(ctx)
	predictions := make([]endpointPredictionResult, 0, len(candidateEndpoints))

	// Prepare inputs for bulk prediction
	metricsStates := make([]*datalayer.Metrics, len(candidateEndpoints))
	prompts := make([]string, len(candidateEndpoints))
	generatedTokenCounts := make([]int, len(candidateEndpoints))
	prefixCacheScores := make([]float64, len(candidateEndpoints))

	for i, endpoint := range candidateEndpoints {
		logger.V(logutil.TRACE).Info("Candidate pod for scheduling", "endpoint", endpoint.GetMetadata().String(), "metrics", endpoint.GetMetrics().String())

		// Get prefix cache score for the pod
		prefixCacheScore := predictedLatencyCtx.prefixCacheScoresForEndpoints[endpoint.GetMetadata().NamespacedName.Name]

		logger.V(logutil.DEBUG).Info("Prefix cache score for pod", "pod", endpoint.GetMetadata().String(), "prefixCacheScore", prefixCacheScore)

		metricsStates[i] = endpoint.GetMetrics()
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
	for i, endpoint := range candidateEndpoints {
		prediction := bulkPredictions[i]
		predResult := endpointPredictionResult{Endpoint: endpoint}

		predResult.PrefixCacheScore = prefixCacheScores[i]
		predResult.TTFT = prediction.TTFT
		predResult.ITL = prediction.ITL

		podMinITLSLO := s.getEndpointMinITLSLO(endpoint)
		predResult.TTFTValid, predResult.ITLValid, predResult.IsValid, predResult.Headroom, predResult.TTFTHeadroom = s.validatePrediction(prediction, predictedLatencyCtx, podMinITLSLO)

		logger.V(logutil.DEBUG).Info("Prediction for scheduling",
			"endpoint", endpoint.GetMetadata().String(),
			"prefixCacheScore", predResult.PrefixCacheScore,
			"TTFT", prediction.TTFT,
			"ITL", prediction.ITL,
			"buffer", s.config.SLOBufferFactor,
			"podMinITLSLO", podMinITLSLO,
			"ttftSLO", predictedLatencyCtx.ttftSLO,
			"requestITLSLO", predictedLatencyCtx.avgITLSLO,
			"itlHeadroom", predResult.Headroom,
			"ttftHeadroom", predResult.TTFTHeadroom,
			"itlValid", predResult.ITLValid,
			"ttftValid", predResult.TTFTValid,
			"headroomStrategy", s.headroomStrategy)

		predictions = append(predictions, predResult)
	}

	return predictions, nil
}

// updateRequestContextWithPredictions updates the request context with prediction data
func (s *PredictedLatency) updateRequestContextWithPredictions(predictedLatencyCtx *predictedLatencyCtx, predictions []endpointPredictionResult) {
	predictedLatencyCtx.predictionsForScheduling = predictions
}

func (s *PredictedLatency) validatePrediction(
	pred *latencypredictor.PredictionResponse,
	predictedLatencyCtx *predictedLatencyCtx,
	podMinITLSLO float64,
) (ttftOk, itlOk, isValid bool, headroom float64, ttftHeadroom float64) {

	ttftOk = pred.TTFT < predictedLatencyCtx.ttftSLO
	ttftHeadroom = predictedLatencyCtx.ttftSLO - pred.TTFT

	itlOk = true
	headroom = 0.0

	if s.config.StreamingMode {
		bufferedITL := predictedLatencyCtx.avgITLSLO * s.config.SLOBufferFactor
		// a podMinITLSLO of 0 means no either no requests, or no ITL SLOs specified on running requests
		if podMinITLSLO > 0 {
			if podMinITLSLO < predictedLatencyCtx.avgITLSLO {
				log.FromContext(context.Background()).V(logutil.DEBUG).Info("Pod min ITL SLO is less than the req SLO, adjusting", "podMinITLSLO", podMinITLSLO, "bufferedITL", predictedLatencyCtx.avgITLSLO)
			}
			bufferedITL = min(bufferedITL, podMinITLSLO*s.config.SLOBufferFactor)
		}

		itlOk = pred.ITL < bufferedITL
		headroom = bufferedITL - pred.ITL
	}

	isValid = ttftOk && itlOk

	return
}
