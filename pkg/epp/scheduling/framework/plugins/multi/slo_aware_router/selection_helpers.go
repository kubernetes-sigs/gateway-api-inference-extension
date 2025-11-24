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
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

func (s *SLOAwareRouter) calculateHeadroomRanges(candidates []podPredictionResult) (minTPOTH, maxTPOTH, minTTFTH, maxTTFTH float64) {
	minTPOTH, maxTPOTH = math.MaxFloat64, -math.MaxFloat64
	minTTFTH, maxTTFTH = math.MaxFloat64, -math.MaxFloat64

	for _, p := range candidates {
		if p.Headroom < minTPOTH {
			minTPOTH = p.Headroom
		}
		if p.Headroom > maxTPOTH {
			maxTPOTH = p.Headroom
		}
		if p.TTFTHeadroom < minTTFTH {
			minTTFTH = p.TTFTHeadroom
		}
		if p.TTFTHeadroom > maxTTFTH {
			maxTTFTH = p.TTFTHeadroom
		}
	}
	return
}

func (s *SLOAwareRouter) calculateWeightedChoices(
	ctx context.Context,
	candidates []podPredictionResult,
	minTPOTH, maxTPOTH, minTTFTH, maxTTFTH float64,
) ([]choice, int) {
	logger := log.FromContext(ctx)
	tpotRange := maxTPOTH - minTPOTH
	ttftRange := maxTTFTH - minTTFTH

	// Precompute blend weights (renormalize if user sets both to 0)
	alpha := s.config.HeadroomTTFTWeight
	beta := s.config.HeadroomTPOTWeight
	if alpha+beta <= 0 {
		alpha = 1.0
		beta = 0.0
	}
	sum := alpha + beta
	alpha /= sum
	beta /= sum

	logger.V(logutil.DEBUG).Info("Positive headroom normalization ranges",
		"minTPOTHeadroom", minTPOTH, "maxTPOTHeadroom", maxTPOTH,
		"minTTFTHeadroom", minTTFTH, "maxTTFTHeadroom", maxTTFTH,
		"alphaTTFT", alpha, "betaTPOT", beta, "strategy", s.headroomStrategy)

	weightedChoices := make([]choice, 0, len(candidates))
	total := 0

	for _, p := range candidates {
		// Normalize to [0,1] within the cohort
		nTPOTH := 0.5
		if tpotRange > eps {
			nTPOTH = (p.Headroom - minTPOTH) / (tpotRange + eps)
		}
		nTTFTH := 0.5
		if ttftRange > eps {
			nTTFTH = (p.TTFTHeadroom - minTTFTH) / (ttftRange + eps)
		}

		// Blend: larger combined -> "safer"; smaller -> "tighter packing"
		combined := alpha*nTTFTH + beta*nTPOTH

		// Map to integer weights
		var w int
		switch s.headroomStrategy {
		case headroomStrategyLeast:
			// prefer smaller combined headroom (pack closer to limits)
			w = int((1.0-combined)*float64(wMax-minWeight)) + minWeight + 1
		case headroomStrategyMost:
			// prefer larger combined headroom (more conservative / spread)
			w = int(combined*float64(wMax-minWeight)) + minWeight + 1
		default:
			// Fallback to least
			w = int((1.0-combined)*float64(wMax-minWeight)) + minWeight + 1
		}

		weightedChoices = append(weightedChoices, choice{podName: p.Pod, weight: w})
		total += w

		logger.V(logutil.TRACE).Info("Positive headroom blended weight",
			"pod", p.Pod.GetPod().String(),
			"ttftHeadroom", p.TTFTHeadroom, "normTTFTHeadroom", nTTFTH,
			"tpotHeadroom", p.Headroom, "normTPOTHeadroom", nTPOTH,
			"combined", combined, "weight", w)
	}
	return weightedChoices, total
}
