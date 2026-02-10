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
	"fmt"
	"math/rand"

	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
)

func (s *PredictedLatency) parseSLOHeaders(ctx context.Context, request *schedulingtypes.LLMRequest, predictedLatencyCtx *predictedLatencyCtx) {
	logger := log.FromContext(ctx)
	var err error

	// Get Request SLOs from request header
	predictedLatencyCtx.ttftSLO, err = parseFloatHeader(*request, ttftSLOHeaderKey)
	if err != nil {
		logger.V(logutil.DEBUG).Error(errutil.Error{Code: errutil.BadRequest, Msg: fmt.Sprintf("%v must be a float: %v", ttftSLOHeaderKey, err)}, "PredictedLatency: Error parsing TTFT SLO from header")
	}

	predictedLatencyCtx.avgTPOTSLO, err = parseFloatHeader(*request, tpotSLOHeaderKey)
	if err != nil {
		logger.V(logutil.DEBUG).Error(errutil.Error{Code: errutil.BadRequest, Msg: fmt.Sprintf("%v must be a float: %v", tpotSLOHeaderKey, err)}, "PredictedLatency: Error parsing TPOT SLO from header")
	}
}

func (s *PredictedLatency) classifyEndpointsByHeadroom(allPreds []endpointPredictionResult) (posHeadroomEndpoints, negHeadroomEndpoints []endpointPredictionResult) {
	for _, p := range allPreds {
		// An endpoint has positive headroom only if BOTH TTFT and TPOT have positive headroom
		if (p.Headroom >= 0) && p.TTFTHeadroom >= 0 {
			posHeadroomEndpoints = append(posHeadroomEndpoints, p)
		} else {
			// An endpoint has negative headroom if EITHER TTFT or TPOT has negative/zero headroom
			negHeadroomEndpoints = append(negHeadroomEndpoints, p)
		}
	}
	return
}

func (s *PredictedLatency) selectEndpointBasedOnStrategy(
	ctx context.Context,
	r *rand.Rand,
	allPreds, posHeadroomEndpoints, negHeadroomEndpoints []endpointPredictionResult,
) schedulingtypes.Endpoint {
	logger := log.FromContext(ctx)
	var selectedEndpoint schedulingtypes.Endpoint

	switch {
	case s.headroomStrategy == headroomStrategyCompositeOnly:
		logger.V(logutil.DEBUG).Info("Selecting from composite scores only")
		selectedEndpoint = s.selectFromCompositeScores(ctx, allPreds, r, headroomStrategyCompositeOnly)
	case len(posHeadroomEndpoints) > 0 && len(negHeadroomEndpoints) > 0:
		// 99% chance to select from positive headroom endpoints, 1% from negative
		if r.Float64() < s.config.EpsilonExploreNeg {
			logger.V(logutil.DEBUG).Info("Selecting from negative headroom endpoints (1% chance)")
			selectedEndpoint = s.selectFromNegativeHeadroomEndpoints(ctx, negHeadroomEndpoints, r)
		} else {
			logger.V(logutil.DEBUG).Info("Selecting from positive headroom endpoints (99% chance)")
			selectedEndpoint = s.selectFromPositiveHeadroomEndpoints(ctx, posHeadroomEndpoints, r)
		}
	case len(posHeadroomEndpoints) > 0:
		// If only positive headroom endpoints exist, select from them
		logger.V(logutil.DEBUG).Info("Only positive headroom endpoints available")
		selectedEndpoint = s.selectFromPositiveHeadroomEndpoints(ctx, posHeadroomEndpoints, r)
	case len(negHeadroomEndpoints) > 0:
		// If only negative headroom endpoints exist, select from them
		logger.V(logutil.DEBUG).Info("Only negative headroom endpoints available")
		selectedEndpoint = s.selectFromNegativeHeadroomEndpoints(ctx, negHeadroomEndpoints, r)
	case len(allPreds) > 0:
		// fallback - select randomly from valid endpoints
		logger.V(logutil.DEBUG).Info("No headroom endpoints available, selecting randomly from valid endpoints")
		selectedEndpoint = allPreds[r.Intn(len(allPreds))].Endpoint
	default:
		// No valid endpoints - return nil (caller handles this)
		logger.V(logutil.DEBUG).Info("No valid endpoints available")
		return nil
	}
	return selectedEndpoint
}

func (s *PredictedLatency) removeRequestFromEndpoint(endpointName types.NamespacedName, requestID string) {
	if value, ok := s.runningRequestLists.Load(endpointName); ok {
		queue := value.(*requestPriorityQueue)
		queue.Remove(requestID)
	}
}

func (s *PredictedLatency) removeRequestFromQueue(requestID string, ctx *predictedLatencyCtx) {
	if ctx == nil || ctx.targetMetadata == nil {
		return
	}
	endpointName := types.NamespacedName{
		Name:      ctx.targetMetadata.NamespacedName.Name,
		Namespace: ctx.targetMetadata.NamespacedName.Namespace,
	}
	s.removeRequestFromEndpoint(endpointName, requestID)
}
