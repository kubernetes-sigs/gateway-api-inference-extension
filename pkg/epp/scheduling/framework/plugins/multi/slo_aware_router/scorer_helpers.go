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
	"fmt"
	"math/rand"

	"sigs.k8s.io/controller-runtime/pkg/log"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

func (s *SLOAwareRouter) parseSLOHeaders(ctx context.Context, request *schedulingtypes.LLMRequest, sloCtx *sloRequestContext) {
	logger := log.FromContext(ctx)
	var err error

	// Get Request SLOs from request header
	sloCtx.ttftSLO, err = parseFloatHeader(*request, ttftSLOHeaderKey)
	if err != nil {
		logger.V(logutil.DEBUG).Error(errutil.Error{Code: errutil.BadRequest, Msg: fmt.Sprintf("%v must be a float: %v", ttftSLOHeaderKey, err)}, "SLOAwareRouter: Error parsing TTFT SLO from header")
	}

	sloCtx.avgTPOTSLO, err = parseFloatHeader(*request, tpotSLOHeaderKey)
	if err != nil {
		logger.V(logutil.DEBUG).Error(errutil.Error{Code: errutil.BadRequest, Msg: fmt.Sprintf("%v must be a float: %v", tpotSLOHeaderKey, err)}, "SLOAwareRouter: Error parsing TPOT SLO from header")
	}
	sloCtx.predictorBasedScheduling = !hasHeader(*request, "x-prediction-based-scheduling-off")
}

func (s *SLOAwareRouter) classifyPodsByHeadroom(allPreds []podPredictionResult) (posHeadroomPods, negHeadroomPods []podPredictionResult) {
	for _, p := range allPreds {
		// A pod has positive headroom only if BOTH TTFT and TPOT have positive headroom
		if p.Headroom > 0 && p.TTFTHeadroom > 0 {
			posHeadroomPods = append(posHeadroomPods, p)
		} else {
			// A pod has negative headroom if EITHER TTFT or TPOT has negative/zero headroom
			negHeadroomPods = append(negHeadroomPods, p)
		}
	}
	return
}

func (s *SLOAwareRouter) selectPodBasedOnStrategy(
	ctx context.Context,
	r *rand.Rand,
	allPreds, posHeadroomPods, negHeadroomPods []podPredictionResult,
) schedulingtypes.Pod {
	logger := log.FromContext(ctx)
	var selectedPod schedulingtypes.Pod

	switch {
	case s.headroomStrategy == headroomStrategyCompositeOnly:
		logger.V(logutil.DEBUG).Info("Selecting from composite scores only")
		selectedPod = s.selectFromCompositeScores(ctx, allPreds, r, headroomStrategyCompositeOnly)
	case len(posHeadroomPods) > 0 && len(negHeadroomPods) > 0:
		// 99% chance to select from positive headroom pods, 1% from negative
		if r.Float64() < s.config.EpsilonExploreNeg {
			logger.V(logutil.DEBUG).Info("Selecting from negative headroom pods (1% chance)")
			selectedPod = s.selectFromNegativeHeadroomPods(ctx, negHeadroomPods, r)
		} else {
			logger.V(logutil.DEBUG).Info("Selecting from positive headroom pods (99% chance)")
			selectedPod = s.selectFromPositiveHeadroomPods(ctx, posHeadroomPods, r)
		}
	case len(posHeadroomPods) > 0:
		// If only positive headroom pods exist, select from them
		logger.V(logutil.DEBUG).Info("Only positive headroom pods available")
		selectedPod = s.selectFromPositiveHeadroomPods(ctx, posHeadroomPods, r)
	case len(negHeadroomPods) > 0:
		// If only negative headroom pods exist, select from them
		logger.V(logutil.DEBUG).Info("Only negative headroom pods available")
		selectedPod = s.selectFromNegativeHeadroomPods(ctx, negHeadroomPods, r)
	case len(allPreds) > 0:
		// fallback - select randomly from valid pods
		logger.V(logutil.DEBUG).Info("No headroom pods available, selecting randomly from valid pods")
		selectedPod = allPreds[r.Intn(len(allPreds))].Pod
	default:
		// No valid pods - return nil (caller handles this)
		logger.V(logutil.DEBUG).Info("No valid pods available")
		return nil
	}
	return selectedPod
}
