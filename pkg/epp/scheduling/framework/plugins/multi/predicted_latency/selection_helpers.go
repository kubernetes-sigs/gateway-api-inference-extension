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

package predicted_latency

import (
	"context"
	"math"

	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

func (s *PredictedLatency) calculateHeadroomRanges(candidates []endpointPredictionResult) (minITLH, maxITLH, minTTFTH, maxTTFTH float64) {
	minITLH, maxITLH = math.MaxFloat64, -math.MaxFloat64
	minTTFTH, maxTTFTH = math.MaxFloat64, -math.MaxFloat64

	for _, p := range candidates {
		if p.Headroom < minITLH {
			minITLH = p.Headroom
		}
		if p.Headroom > maxITLH {
			maxITLH = p.Headroom
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

func (s *PredictedLatency) calculateWeightedChoices(
	ctx context.Context,
	candidates []endpointPredictionResult,
	minITLH, maxITLH, minTTFTH, maxTTFTH float64,
) ([]choice, int) {
	logger := log.FromContext(ctx)
	itlRange := maxITLH - minITLH
	ttftRange := maxTTFTH - minTTFTH

	// Precompute blend weights (renormalize if user sets both to 0)
	alpha := s.config.HeadroomTTFTWeight
	beta := s.config.HeadroomITLWeight

	if !s.config.StreamingMode {
		alpha = 1
		beta = 0
	}
	if alpha+beta <= 0 {
		alpha = 1.0
		beta = 0.0
	}
	sum := alpha + beta
	alpha /= sum
	beta /= sum

	logger.V(logutil.DEBUG).Info("Positive headroom normalization ranges",
		"minITLHeadroom", minITLH, "maxITLHeadroom", maxITLH,
		"minTTFTHeadroom", minTTFTH, "maxTTFTHeadroom", maxTTFTH,
		"alphaTTFT", alpha, "betaITL", beta, "strategy", s.headroomStrategy)

	weightedChoices := make([]choice, 0, len(candidates))
	total := 0

	for _, e := range candidates {
		// Normalize to [0,1] within the cohort
		nITLH := 0.5
		if itlRange > eps {
			nITLH = (e.Headroom - minITLH) / (itlRange + eps)
		}
		nTTFTH := 0.5
		if ttftRange > eps {
			nTTFTH = (e.TTFTHeadroom - minTTFTH) / (ttftRange + eps)
		}

		// Blend: larger combined -> "safer"; smaller -> "tighter packing"
		combined := alpha*nTTFTH + beta*nITLH

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

		weightedChoices = append(weightedChoices, choice{endpointName: e.Endpoint, weight: w})
		total += w

		logger.V(logutil.TRACE).Info("Positive headroom blended weight",
			"endpoint", e.Endpoint.GetMetadata().String(),
			"ttftHeadroom", e.TTFTHeadroom, "normTTFTHeadroom", nTTFTH,
			"itlHeadroom", e.Headroom, "normITLHeadroom", nITLH,
			"combined", combined, "weight", w)
	}
	return weightedChoices, total
}
