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
	attrprefix "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

// PrepareRequestData prepares the SLO context for the request, including parsing SLO headers and gathering prefix cache scores abds generating predictions.
func (s *PredictedLatency) PrepareRequestData(ctx context.Context, request *schedulingtypes.LLMRequest, endpoints []schedulingtypes.Endpoint) error {
	logger := log.FromContext(ctx)
	predictedLatencyCtx := s.getOrMakePredictedLatencyContextForRequest(request)

	s.parseSLOHeaders(ctx, request, predictedLatencyCtx)
	var prefixCacheScore float64
	for _, endpoint := range endpoints {

		if prefixCacheInfoRaw, ok := endpoint.Get(attrprefix.PrefixCacheMatchInfoKey); ok {
			prefixCacheInfo := prefixCacheInfoRaw.(*attrprefix.PrefixCacheMatchInfo)
			prefixCacheScore = float64(prefixCacheInfo.MatchBlocks()) / float64(prefixCacheInfo.TotalBlocks())
			if !math.IsNaN(prefixCacheScore) {
				logger.V(logutil.DEBUG).Info("Found prefix cache score in pod attribute", "pod", endpoint.GetMetadata().NamespacedName.Name, "score", prefixCacheScore)
			} else {
				prefixCacheScore = 0.0
				logger.V(logutil.DEBUG).Info("Prefix cache score is NaN, defaulting to 0", "pod", endpoint.GetMetadata().NamespacedName.Name)
			}
		} else {
			logger.V(logutil.DEBUG).Info("No prefix cache score found in pod attribute, defaulting to 0", "pod", endpoint.GetMetadata().NamespacedName.Name)
			prefixCacheScore = 0.0
		}
		predictedLatencyCtx.prefixCacheScoresForEndpoints[endpoint.GetMetadata().NamespacedName.Name] = prefixCacheScore
	}
	predictions, err := s.generatePredictions(ctx, predictedLatencyCtx, endpoints)
	if err == nil && len(predictions) == len(endpoints) {
		s.updateRequestContextWithPredictions(predictedLatencyCtx, predictions)
		s.updateHasValidPod(ctx, predictedLatencyCtx, endpoints)
	}

	s.setPredictedLatencyContextForRequest(request, predictedLatencyCtx)
	return nil
}

func (p *PredictedLatency) Produces() map[string]any {
	return map[string]any{}
}

func (p *PredictedLatency) Consumes() map[string]any {
	return map[string]any{attrprefix.PrefixCacheMatchInfoKey: attrprefix.PrefixCacheMatchInfo{}}
}
