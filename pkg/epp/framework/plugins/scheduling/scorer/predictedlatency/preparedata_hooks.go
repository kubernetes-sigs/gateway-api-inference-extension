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
	"math"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrlatency "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/latency"
	attrprefix "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

// PrepareRequestData prepares the SLO context for the request, including parsing SLO headers and gathering prefix cache scores abds generating predictions.
func (s *PredictedLatency) PrepareRequestData(ctx context.Context, request *schedulingtypes.LLMRequest, endpoints []schedulingtypes.Endpoint) error {
	predictedLatencyCtx := s.getOrMakePredictedLatencyContextForRequest(request)

	s.parseSLOHeaders(ctx, request, predictedLatencyCtx)
	for _, endpoint := range endpoints {
		podName := endpoint.GetMetadata().NamespacedName.Name
		prefixCacheScore := extractPrefixCacheScore(ctx, endpoint)
		predictedLatencyCtx.prefixCacheScoresForEndpoints[podName] = prefixCacheScore
	}
	predictions, err := s.generatePredictions(ctx, predictedLatencyCtx, endpoints)
	if err == nil && len(predictions) == len(endpoints) {
		s.updateRequestContextWithPredictions(predictedLatencyCtx, predictions)
		s.updateHasValidPod(ctx, predictedLatencyCtx, endpoints)

		// Store predictions in endpoint attributes
		for _, pred := range predictions {
			if pred.Endpoint != nil {
				latencyInfo := attrlatency.NewLatencyPredictionInfo(
					pred.TTFTValid,
					pred.TPOTValid,
					pred.TTFTHeadroom,
					pred.Headroom, // Maps to TPOTHeadroom
					pred.TTFT,
					pred.TPOT,
				)
				pred.Endpoint.Put(attrlatency.LatencyPredictionInfoKey, latencyInfo)
				logger.V(logutil.DEBUG).Info("Stored latency prediction in endpoint",
					"pod", pred.Endpoint.GetMetadata().NamespacedName.Name,
					"ttft", pred.TTFT,
					"tpot", pred.TPOT,
					"ttftValid", pred.TTFTValid,
					"tpotValid", pred.TPOTValid,
					"ttftHeadroom", pred.TTFTHeadroom,
					"tpotHeadroom", pred.Headroom)
			}
		}
	}

	s.setPredictedLatencyContextForRequest(request, predictedLatencyCtx)
	return nil
}

// extractPrefixCacheScore extracts the prefix cache score from endpoint attributes.
// It prioritizes PrecisePrefixCacheMatchInfo (device-tier-aware weighted score) over
// PrefixCacheMatchInfo (unweighted ratio). Returns 0.0 if no valid score is found.
func extractPrefixCacheScore(ctx context.Context, endpoint schedulingtypes.Endpoint) float64 {
	logger := log.FromContext(ctx)
	podName := endpoint.GetMetadata().NamespacedName.Name

	// Try precise prefix cache first (weighted score)
	if precisePrefixCacheInfoRaw, ok := endpoint.Get(attrprefix.PrecisePrefixCacheMatchInfoKey); ok {
		precisePrefixCacheInfo := precisePrefixCacheInfoRaw.(*attrprefix.PrecisePrefixCacheMatchInfo)
		score := precisePrefixCacheInfo.WeightedScore()
		if !math.IsNaN(score) {
			logger.V(logutil.DEBUG).Info("Using precise prefix cache weighted score",
				"pod", podName, "weightedScore", score)
			return score
		}
	}

	// Fall back to approximate prefix cache (unweighted ratio)
	if prefixCacheInfoRaw, ok := endpoint.Get(attrprefix.PrefixCacheMatchInfoKey); ok {
		prefixCacheInfo := prefixCacheInfoRaw.(*attrprefix.PrefixCacheMatchInfo)
		score := float64(prefixCacheInfo.MatchBlocks()) / float64(prefixCacheInfo.TotalBlocks())
		if !math.IsNaN(score) {
			logger.V(logutil.DEBUG).Info("Using approximate prefix cache score",
				"pod", podName, "score", score)
			return score
		}
	}

	logger.V(logutil.DEBUG).Info("No prefix cache score found, defaulting to 0", "pod", podName)
	return 0.0
}

func (p *PredictedLatency) Produces() map[string]any {
	return map[string]any{
		attrlatency.LatencyPredictionInfoKey: attrlatency.LatencyPredictionInfo{},
	}
}

func (p *PredictedLatency) Consumes() map[string]any {
	return map[string]any{
		attrprefix.PrefixCacheMatchInfoKey:        attrprefix.PrefixCacheMatchInfo{},
		attrprefix.PrecisePrefixCacheMatchInfoKey: attrprefix.PrecisePrefixCacheMatchInfo{},
	}
}
