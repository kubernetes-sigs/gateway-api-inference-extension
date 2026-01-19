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

package slo_aware_router

import (
	"context"
	"math"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/util/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer/plugins/approximateprefix"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

// PrepareRequestData prepares the SLO context for the request, including parsing SLO headers and gathering prefix cache scores abds generating predictions.
func (s *SLOAwareRouter) PrepareRequestData(ctx context.Context, request *schedulingtypes.LLMRequest, endpoints []schedulingtypes.Endpoint) error {
	logger := log.FromContext(ctx)
	sloCtx := s.getOrMakeSLORequestContext(request)

	s.parseSLOHeaders(ctx, request, sloCtx)
	var prefixCacheScore float64
	for _, endpoint := range endpoints {

		if prefixCacheInfoRaw, ok := endpoint.Get(approximateprefix.PrefixCacheMatchInfoKey); ok {
			prefixCacheInfo := prefixCacheInfoRaw.(*approximateprefix.PrefixCacheMatchInfo)
			prefixCacheScore = float64(prefixCacheInfo.MatchLength()) / float64(prefixCacheInfo.TotalLength())
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
		sloCtx.prefixCacheScoresForEndpoints[endpoint.GetMetadata().NamespacedName.Name] = prefixCacheScore
	}
	s.setSLOContextForRequest(request, sloCtx)
	return nil
}

func (p *SLOAwareRouter) Produces() map[string]any {
	return map[string]any{}
}

func (p *SLOAwareRouter) Consumes() map[string]any {
	return map[string]any{approximateprefix.PrefixCacheMatchInfoKey: approximateprefix.PrefixCacheMatchInfo{}}
}
