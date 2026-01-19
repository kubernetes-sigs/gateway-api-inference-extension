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
	"math"
	"math/rand"

	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/util/logging"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

// selectFromPositiveHeadroomEndpoints selects a endpoint from positive headroom endpoints using headroom strategy
// Updated to incorporate TTFTHeadroom with a configurable blend vs ITL headroom.
func (s *PredictedLatency) selectFromPositiveHeadroomEndpoints(ctx context.Context, posHeadroomEndpoints []endpointPredictionResult, r *rand.Rand) schedulingtypes.Endpoint {

	if len(posHeadroomEndpoints) == 1 {
		return posHeadroomEndpoints[0].Endpoint
	}

	// Apply perfect stickiness (with exploration)
	candidates, sticky := s.epsilonGreedyAffinityGate(ctx, posHeadroomEndpoints, r, "positive", s.config.AffinityGateTau)

	// If perfect stickiness collapsed us to a single endpoint, short-circuit
	if sticky && len(candidates) == 1 {
		return candidates[0].Endpoint
	}
	switch s.headroomStrategy {
	case headroomStrategyCompositeMost:
		return s.selectFromCompositeScores(ctx, candidates, r, headroomStrategyCompositeMost)
	case headroomStrategyCompositeLeast:
		return s.selectFromCompositeScores(ctx, candidates, r, headroomStrategyCompositeLeast)
	}

	// Find min/max for ITL (Headroom) and TTFTHeadroom across positive endpoints to normalize to [0,1]
	minITLH, maxITLH, minTTFTH, maxTTFTH := s.calculateHeadroomRanges(candidates)

	// Calculate weights for weighted random selection
	weightedChoices, total := s.calculateWeightedChoices(ctx, candidates, minITLH, maxITLH, minTTFTH, maxTTFTH)

	return s.performWeightedRandomSelection(weightedChoices, total, candidates, r)
}

// selectFromNegativeHeadroomEndpoints selects an endpoint from negative headroom endpoints using hierarchical TTFT/ITL logic
// Modified to strictly prefer endpoints with 0 running requests
func (s *PredictedLatency) selectFromNegativeHeadroomEndpoints(ctx context.Context, negHeadroomEndpoints []endpointPredictionResult, r *rand.Rand) schedulingtypes.Endpoint {
	logger := log.FromContext(ctx)

	if len(negHeadroomEndpoints) == 1 {
		return negHeadroomEndpoints[0].Endpoint
	}

	// First, separate endpoints by running request count
	var zeroRunningRequestEndpoints, nonZeroRunningRequestEndpoints []endpointPredictionResult

	for _, e := range negHeadroomEndpoints {
		runningRequestCount := s.getEndpointRunningRequestCount(e.Endpoint)
		if runningRequestCount == 0 {
			zeroRunningRequestEndpoints = append(zeroRunningRequestEndpoints, e)
		} else {
			nonZeroRunningRequestEndpoints = append(nonZeroRunningRequestEndpoints, e)
		}
	}

	logger.V(logutil.DEBUG).Info("Negative headroom endpoints by running request count",
		"zeroRunningRequests", len(zeroRunningRequestEndpoints),
		"nonZeroRunningRequests", len(nonZeroRunningRequestEndpoints))

	// If we have endpoints with 0 running requests, strictly prefer them
	if len(zeroRunningRequestEndpoints) > 0 {
		logger.V(logutil.DEBUG).Info("Selecting from endpoints with zero running requests")
		return s.selectFromNegativeHeadroomEndpointsInternal(ctx, zeroRunningRequestEndpoints, r)
	}

	// Otherwise, fall back to endpoints with running requests
	logger.V(logutil.DEBUG).Info("No endpoints with zero running requests, selecting from endpoints with running requests")
	return s.selectFromNegativeHeadroomEndpointsInternal(ctx, nonZeroRunningRequestEndpoints, r)
}

// selectFromNegativeHeadroomEndpointsInternal handles the actual selection logic for negative headroom endpoints
func (s *PredictedLatency) selectFromNegativeHeadroomEndpointsInternal(ctx context.Context, negHeadroomEndpoints []endpointPredictionResult, r *rand.Rand) schedulingtypes.Endpoint {
	if len(negHeadroomEndpoints) == 1 {
		return negHeadroomEndpoints[0].Endpoint
	}

	// Apply perfect stickiness (with exploration)
	candidates, sticky := s.epsilonGreedyAffinityGate(ctx, negHeadroomEndpoints, r, "negative", s.config.AffinityGateTau)

	// If perfect stickiness collapsed us to a single endpoint, short-circuit
	if sticky && len(candidates) == 1 {
		return candidates[0].Endpoint
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

	s.handleNegativeHeadroomEndpointsHierarchical(ctx, candidates, &weightedChoices, &total, minWeight)

	// Perform weighted random selection
	return s.performWeightedRandomSelection(weightedChoices, total, candidates, r)
}

// weightEndpointsByBlendedDeficit applies blended weighting using TTFT and ITL deficits.
// Lower blended deficit => higher weight.
func (ps *PredictedLatency) weightEndpointsByBlendedDeficit(
	ctx context.Context,
	endpoints []endpointPredictionResult,
	choices *[]choice,
	total *int,
	minWeight int,
	alpha, beta float64, // weights for TTFT and ITL deficits
	category string,
) {
	logger := log.FromContext(ctx)
	if len(endpoints) == 0 {
		return
	}

	const Wrange = 80
	const eps = 1e-9

	// Compute raw deficits (only when headroom is negative)
	type deficits struct {
		endpoint endpointPredictionResult
		ttftDef  float64
		itlDef   float64
	}
	defs := make([]deficits, 0, len(endpoints))

	minTTFT, maxTTFT := math.MaxFloat64, -math.MaxFloat64
	minITL, maxITL := math.MaxFloat64, -math.MaxFloat64

	for _, e := range endpoints {
		ttftDef := 0.0
		if e.TTFTHeadroom < 0 {
			ttftDef = -e.TTFTHeadroom
		}
		itlDef := 0.0
		if e.Headroom < 0 {
			itlDef = -e.Headroom
		}
		defs = append(defs, deficits{endpoint: e, ttftDef: ttftDef, itlDef: itlDef})

		if ttftDef < minTTFT {
			minTTFT = ttftDef
		}
		if ttftDef > maxTTFT {
			maxTTFT = ttftDef
		}
		if itlDef < minITL {
			minITL = itlDef
		}
		if itlDef > maxITL {
			maxITL = itlDef
		}
	}

	ttftRange := maxTTFT - minTTFT
	itlRange := maxITL - minITL

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
		"minITLDef", minITL, "maxITLDef", maxITL,
		"alphaTTFT", alpha, "betaITL", beta, "endpointCount", len(endpoints))

	for _, d := range defs {
		// Normalize deficits to [0,1] within this bucket (0 = best / least violation)
		nTTFT := 0.0
		if ttftRange > eps {
			nTTFT = (d.ttftDef - minTTFT) / (ttftRange + eps)
		}
		nITL := 0.0
		if itlRange > eps {
			nITL = (d.itlDef - minITL) / (itlRange + eps)
		}

		// Blended "badness": higher = worse violation
		blended := alpha*nTTFT + beta*nITL

		// Convert to selection weight: lower badness -> higher weight
		// Ensure a floor so no endpoint is completely excluded within the bucket.
		w := int((1.0-blended)*float64(Wrange)) + minWeight + 1

		*choices = append(*choices, choice{endpointName: d.endpoint.Endpoint, weight: w})
		*total += w

		logger.V(logutil.TRACE).Info("Negative bucket blended weighting",
			"endpoint", d.endpoint.Endpoint.GetMetadata().String(),
			"ttftDef", d.ttftDef, "itlDef", d.itlDef,
			"normTTFT", nTTFT, "normITL", nITL,
			"blendedBadness", blended, "weight", w)
	}
}

func (s *PredictedLatency) handleNegativeHeadroomEndpointsHierarchical(
	ctx context.Context,
	negHeadroomEndpoints []endpointPredictionResult,
	choices *[]choice,
	total *int,
	minWeightForNegative int,
) {
	logger := log.FromContext(ctx)

	// Categorize endpoints by their headroom status
	var negTTFTNegITL, negTTFTNonNegITL, nonNegTTFTNegITL, nonNegTTFTNonNegITL []endpointPredictionResult

	for _, p := range negHeadroomEndpoints {
		switch {
		case p.TTFTHeadroom < 0 && p.Headroom < 0:
			negTTFTNegITL = append(negTTFTNegITL, p)
		case p.TTFTHeadroom < 0 && p.Headroom >= 0:
			negTTFTNonNegITL = append(negTTFTNonNegITL, p)
		case p.TTFTHeadroom >= 0 && p.Headroom < 0:
			nonNegTTFTNegITL = append(nonNegTTFTNegITL, p)
		default:
			nonNegTTFTNonNegITL = append(nonNegTTFTNonNegITL, p)
		}
	}

	logger.V(logutil.DEBUG).Info("Hierarchical negative headroom endpoint distribution",
		"totalNegative", len(negHeadroomEndpoints),
		"negTTFT_negITL", len(negTTFTNegITL),
		"negTTFT_nonNegITL", len(negTTFTNonNegITL),
		"nonNegTTFT_negITL", len(nonNegTTFTNegITL),
		"nonNegTTFT_nonNegITL", len(nonNegTTFTNonNegITL))

	// Priority 1: both TTFT and ITL negative -> blended deficits (both active)
	alpha := s.config.NegHeadroomTTFTWeight
	beta := s.config.NegHeadroomITLWeight
	if !s.config.StreamingMode {
		alpha = 1
		beta = 0
	}
	if len(negTTFTNegITL) > 0 {
		s.weightEndpointsByBlendedDeficit(ctx, negTTFTNegITL, choices, total, minWeightForNegative,
			alpha, beta, "both_negative")
	}

	// Priority 2: TTFT negative, ITL non-negative -> blended still works (ITL deficit=0)
	if len(negTTFTNonNegITL) > 0 {
		s.weightEndpointsByBlendedDeficit(ctx, negTTFTNonNegITL, choices, total, minWeightForNegative,
			alpha, beta, "ttft_negative")
	}

	// Priority 3: TTFT non-negative, ITL negative -> blended (TTFT deficit=0)
	if len(nonNegTTFTNegITL) > 0 {
		s.weightEndpointsByBlendedDeficit(ctx, nonNegTTFTNegITL, choices, total, minWeightForNegative,
			alpha, beta, "itl_negative")
	}

	// Priority 4: edge-case bucket -> minimal weight
	for _, e := range nonNegTTFTNonNegITL {
		*choices = append(*choices, choice{endpointName: e.Endpoint, weight: minWeightForNegative})
		*total += minWeightForNegative
	}
}

func (s *PredictedLatency) getEndpointMinITLSLO(endpoint schedulingtypes.Endpoint) float64 {
	endpointName := types.NamespacedName{
		Name:      endpoint.GetMetadata().NamespacedName.Name,
		Namespace: endpoint.GetMetadata().NamespacedName.Namespace,
	}
	if runningReqs, ok := s.runningRequestLists[endpointName]; ok && runningReqs.GetSize() > 0 {
		if topReq := runningReqs.Peek(); topReq != nil {
			return topReq.itl
		}
	}
	return 0 // no running requests or no ITL SLOs
}

func (s *PredictedLatency) getEndpointRunningRequestCount(endpoint schedulingtypes.Endpoint) int {
	endpointName := types.NamespacedName{
		Name:      endpoint.GetMetadata().NamespacedName.Name,
		Namespace: endpoint.GetMetadata().NamespacedName.Namespace,
	}
	if runningReqs, ok := s.runningRequestLists[endpointName]; ok {
		return runningReqs.GetSize()
	}
	return 0 // no running requests
}
