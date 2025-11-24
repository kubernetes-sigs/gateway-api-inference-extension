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
	"math"
	"math/rand"

	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// selectFromPositiveHeadroomPods selects a pod from positive headroom pods using headroom strategy
// Updated to incorporate TTFTHeadroom with a configurable blend vs TPOT headroom.
func (s *SLOAwareRouter) selectFromPositiveHeadroomPods(ctx context.Context, posHeadroomPods []podPredictionResult, r *rand.Rand) schedulingtypes.Pod {

	if len(posHeadroomPods) == 1 {
		return posHeadroomPods[0].Pod
	}

	// Apply perfect stickiness (with exploration)
	candidates, sticky := s.epsilonGreedyAffinityGate(ctx, posHeadroomPods, r, "positive", s.config.AffinityGateTau)

	// If perfect stickiness collapsed us to a single pod, short-circuit
	if sticky && len(candidates) == 1 {
		return candidates[0].Pod
	}
	switch s.headroomStrategy {
	case headroomStrategyCompositeMost:
		return s.selectFromCompositeScores(ctx, candidates, r, headroomStrategyCompositeMost)
	case headroomStrategyCompositeLeast:
		return s.selectFromCompositeScores(ctx, candidates, r, headroomStrategyCompositeLeast)
	}

	// Find min/max for TPOT (Headroom) and TTFTHeadroom across positive pods to normalize to [0,1]
	minTPOTH, maxTPOTH, minTTFTH, maxTTFTH := s.calculateHeadroomRanges(candidates)

	// Calculate weights for weighted random selection
	weightedChoices, total := s.calculateWeightedChoices(ctx, candidates, minTPOTH, maxTPOTH, minTTFTH, maxTTFTH)

	return s.performWeightedRandomSelection(weightedChoices, total, candidates, r)
}

// selectFromNegativeHeadroomPods selects a pod from negative headroom pods using hierarchical TTFT/TPOT logic
// Modified to strictly prefer pods with 0 running requests
func (s *SLOAwareRouter) selectFromNegativeHeadroomPods(ctx context.Context, negHeadroomPods []podPredictionResult, r *rand.Rand) schedulingtypes.Pod {
	logger := log.FromContext(ctx)

	if len(negHeadroomPods) == 1 {
		return negHeadroomPods[0].Pod
	}

	// First, separate pods by running request count
	var zeroRunningRequestPods, nonZeroRunningRequestPods []podPredictionResult

	for _, p := range negHeadroomPods {
		runningRequestCount := s.getPodRunningRequestCount(p.Pod)
		if runningRequestCount == 0 {
			zeroRunningRequestPods = append(zeroRunningRequestPods, p)
		} else {
			nonZeroRunningRequestPods = append(nonZeroRunningRequestPods, p)
		}
	}

	logger.V(logutil.DEBUG).Info("Negative headroom pods by running request count",
		"zeroRunningRequests", len(zeroRunningRequestPods),
		"nonZeroRunningRequests", len(nonZeroRunningRequestPods))

	// If we have pods with 0 running requests, strictly prefer them
	if len(zeroRunningRequestPods) > 0 {
		logger.V(logutil.DEBUG).Info("Selecting from pods with zero running requests")
		return s.selectFromNegativeHeadroomPodsInternal(ctx, zeroRunningRequestPods, r)
	}

	// Otherwise, fall back to pods with running requests
	logger.V(logutil.DEBUG).Info("No pods with zero running requests, selecting from pods with running requests")
	return s.selectFromNegativeHeadroomPodsInternal(ctx, nonZeroRunningRequestPods, r)
}

// selectFromNegativeHeadroomPodsInternal handles the actual selection logic for negative headroom pods
func (s *SLOAwareRouter) selectFromNegativeHeadroomPodsInternal(ctx context.Context, negHeadroomPods []podPredictionResult, r *rand.Rand) schedulingtypes.Pod {
	if len(negHeadroomPods) == 1 {
		return negHeadroomPods[0].Pod
	}

	// Apply perfect stickiness (with exploration)
	candidates, sticky := s.epsilonGreedyAffinityGate(ctx, negHeadroomPods, r, "negative", s.config.AffinityGateTau)

	// If perfect stickiness collapsed us to a single pod, short-circuit
	if sticky && len(candidates) == 1 {
		return candidates[0].Pod
	}

	switch s.headroomStrategy {
	case headroomStrategyCompositeMost:
		return s.selectFromCompositeScores(ctx, candidates, r, headroomStrategyCompositeMost)
	case headroomStrategyCompositeLeast:
		return s.selectFromCompositeScores(ctx, candidates, r, headroomStrategyCompositeMost)
	}

	// Build weighted choices for selection
	weightedChoices := make([]choice, 0, len(candidates))
	total := 0

	s.handleNegativeHeadroomPodsHierarchical(ctx, candidates, &weightedChoices, &total, minWeight)

	// Perform weighted random selection
	return s.performWeightedRandomSelection(weightedChoices, total, candidates, r)
}

// weightPodsByBlendedDeficit applies blended weighting using TTFT and TPOT deficits.
// Lower blended deficit => higher weight.
func (ps *SLOAwareRouter) weightPodsByBlendedDeficit(
	ctx context.Context,
	pods []podPredictionResult,
	choices *[]choice,
	total *int,
	minWeight int,
	alpha, beta float64, // weights for TTFT and TPOT deficits
	category string,
) {
	logger := log.FromContext(ctx)
	if len(pods) == 0 {
		return
	}

	const Wrange = 80
	const eps = 1e-9

	// Compute raw deficits (only when headroom is negative)
	type deficits struct {
		pod     podPredictionResult
		ttftDef float64
		tpotDef float64
	}
	defs := make([]deficits, 0, len(pods))

	minTTFT, maxTTFT := math.MaxFloat64, -math.MaxFloat64
	minTPOT, maxTPOT := math.MaxFloat64, -math.MaxFloat64

	for _, p := range pods {
		ttftDef := 0.0
		if p.TTFTHeadroom < 0 {
			ttftDef = -p.TTFTHeadroom
		}
		tpotDef := 0.0
		if p.Headroom < 0 {
			tpotDef = -p.Headroom
		}
		defs = append(defs, deficits{pod: p, ttftDef: ttftDef, tpotDef: tpotDef})

		if ttftDef < minTTFT {
			minTTFT = ttftDef
		}
		if ttftDef > maxTTFT {
			maxTTFT = ttftDef
		}
		if tpotDef < minTPOT {
			minTPOT = tpotDef
		}
		if tpotDef > maxTPOT {
			maxTPOT = tpotDef
		}
	}

	ttftRange := maxTTFT - minTTFT
	tpotRange := maxTPOT - minTPOT

	// Normalize alpha/beta
	if alpha+beta <= 0 {
		alpha, beta = 1.0, 0.0
	} else {
		sum := alpha + beta
		alpha /= sum
		beta /= sum
	}

	logger.V(logutil.DEBUG).Info("Negative headroom blended deficits",
		"category", category,
		"minTTFTDef", minTTFT, "maxTTFTDef", maxTTFT,
		"minTPOTDef", minTPOT, "maxTPOTDef", maxTPOT,
		"alphaTTFT", alpha, "betaTPOT", beta, "podCount", len(pods))

	for _, d := range defs {
		// Normalize deficits to [0,1] within this bucket (0 = best / least violation)
		nTTFT := 0.0
		if ttftRange > eps {
			nTTFT = (d.ttftDef - minTTFT) / (ttftRange + eps)
		}
		nTPOT := 0.0
		if tpotRange > eps {
			nTPOT = (d.tpotDef - minTPOT) / (tpotRange + eps)
		}

		// Blended "badness": higher = worse violation
		blended := alpha*nTTFT + beta*nTPOT

		// Convert to selection weight: lower badness -> higher weight
		// Ensure a floor so no pod is completely excluded within the bucket.
		w := int((1.0-blended)*float64(Wrange)) + minWeight + 1

		*choices = append(*choices, choice{podName: d.pod.Pod, weight: w})
		*total += w

		logger.V(logutil.TRACE).Info("Negative bucket blended weighting",
			"pod", d.pod.Pod.GetPod().String(),
			"ttftDef", d.ttftDef, "tpotDef", d.tpotDef,
			"normTTFT", nTTFT, "normTPOT", nTPOT,
			"blendedBadness", blended, "weight", w)
	}
}

func (s *SLOAwareRouter) handleNegativeHeadroomPodsHierarchical(
	ctx context.Context,
	negHeadroomPods []podPredictionResult,
	choices *[]choice,
	total *int,
	minWeightForNegative int,
) {
	logger := log.FromContext(ctx)

	// Categorize pods by their headroom status
	var negTTFTNegTPOT, negTTFTNonNegTPOT, nonNegTTFTNegTPOT, nonNegTTFTNonNegTPOT []podPredictionResult

	for _, p := range negHeadroomPods {
		switch {
		case p.TTFTHeadroom < 0 && p.Headroom < 0:
			negTTFTNegTPOT = append(negTTFTNegTPOT, p)
		case p.TTFTHeadroom < 0 && p.Headroom >= 0:
			negTTFTNonNegTPOT = append(negTTFTNonNegTPOT, p)
		case p.TTFTHeadroom >= 0 && p.Headroom < 0:
			nonNegTTFTNegTPOT = append(nonNegTTFTNegTPOT, p)
		default:
			nonNegTTFTNonNegTPOT = append(nonNegTTFTNonNegTPOT, p)
		}
	}

	logger.V(logutil.DEBUG).Info("Hierarchical negative headroom pod distribution",
		"totalNegative", len(negHeadroomPods),
		"negTTFT_negTPOT", len(negTTFTNegTPOT),
		"negTTFT_nonNegTPOT", len(negTTFTNonNegTPOT),
		"nonNegTTFT_negTPOT", len(nonNegTTFTNegTPOT),
		"nonNegTTFT_nonNegTPOT", len(nonNegTTFTNonNegTPOT))

	// Priority 1: both TTFT and TPOT negative -> blended deficits (both active)
	if len(negTTFTNegTPOT) > 0 {
		s.weightPodsByBlendedDeficit(ctx, negTTFTNegTPOT, choices, total, minWeightForNegative,
			s.config.NegHeadroomTTFTWeight, s.config.NegHeadroomTPOTWeight, "both_negative")
	}

	// Priority 2: TTFT negative, TPOT non-negative -> blended still works (TPOT deficit=0)
	if len(negTTFTNonNegTPOT) > 0 {
		s.weightPodsByBlendedDeficit(ctx, negTTFTNonNegTPOT, choices, total, minWeightForNegative,
			s.config.NegHeadroomTTFTWeight, s.config.NegHeadroomTPOTWeight, "ttft_negative")
	}

	// Priority 3: TTFT non-negative, TPOT negative -> blended (TTFT deficit=0)
	if len(nonNegTTFTNegTPOT) > 0 {
		s.weightPodsByBlendedDeficit(ctx, nonNegTTFTNegTPOT, choices, total, minWeightForNegative,
			s.config.NegHeadroomTTFTWeight, s.config.NegHeadroomTPOTWeight, "tpot_negative")
	}

	// Priority 4: edge-case bucket -> minimal weight
	for _, p := range nonNegTTFTNonNegTPOT {
		*choices = append(*choices, choice{podName: p.Pod, weight: minWeightForNegative})
		*total += minWeightForNegative
	}
}

func (s *SLOAwareRouter) getPodMinTPOTSLO(pod schedulingtypes.Pod) float64 {
	podName := types.NamespacedName{
		Name:      pod.GetPod().NamespacedName.Name,
		Namespace: pod.GetPod().NamespacedName.Namespace,
	}
	if runningReqs, ok := s.runningRequestLists[podName]; ok && runningReqs.GetSize() > 0 {
		if topReq := runningReqs.Peek(); topReq != nil {
			return topReq.tpot
		}
	}
	return 0 // no running requests or no TPOT SLOs
}

func (s *SLOAwareRouter) getPodRunningRequestCount(pod schedulingtypes.Pod) int {
	podName := types.NamespacedName{
		Name:      pod.GetPod().NamespacedName.Name,
		Namespace: pod.GetPod().NamespacedName.Namespace,
	}
	if runningReqs, ok := s.runningRequestLists[podName]; ok {
		return runningReqs.GetSize()
	}
	return 0 // no running requests
}
