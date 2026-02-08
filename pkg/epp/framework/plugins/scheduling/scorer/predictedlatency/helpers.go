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
	"math/rand"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

func (s *PredictedLatency) selectFromCompositeScores(ctx context.Context, allPreds []endpointPredictionResult, r *rand.Rand, strategy headroomStrategy) schedulingtypes.Endpoint {
	total := 0
	choices := s.buildCompositeChoices(
		ctx, allPreds, s.config.CompositeKVWeight, s.config.CompositeQueueWeight, s.config.CompositePrefixWeight, &total,
	)
	if strategy == headroomStrategyCompositeLeast {
		// Invert weights for "least" strategy
		for i := range choices {
			choices[i].weight = minWeight + wMax - choices[i].weight
		}
	}
	selectedEndpoint := s.performWeightedRandomSelection(choices, total, allPreds, r)
	return selectedEndpoint
}
func (s *PredictedLatency) performWeightedRandomSelection(weightedChoices []choice, total int, candidates []endpointPredictionResult, r *rand.Rand) schedulingtypes.Endpoint {
	if total == 0 {
		return nil
	}
	logger := log.FromContext(context.Background())
	// Check if MAX_SCORE_SELECTION env variable is set
	if s.config.SelectionMode == string(podSelectionMax) {

		logger.V(logutil.DEBUG).Info("Pod selection mode: MAX - selecting pod with highest weight")
		maxWeight := 0
		var selectedEndpoint schedulingtypes.Endpoint
		for _, c := range weightedChoices {
			if c.weight > maxWeight {
				maxWeight = c.weight
				selectedEndpoint = c.endpointName
			}
		}
		if selectedEndpoint != nil {
			return selectedEndpoint
		}
		// Fallback to first pod if no selection made
		return candidates[0].Endpoint
	}

	// Original weighted random selection logic
	logger.V(logutil.DEBUG).Info("Pod selection mode: LINEAR - performing weighted random selection")
	idx := r.Intn(total)
	var selectedEndpoint schedulingtypes.Endpoint

	for _, c := range weightedChoices {
		if idx < c.weight {
			selectedEndpoint = c.endpointName
			break
		}
		idx -= c.weight
	}

	// If no endpoint was selected (shouldn't happen), fallback to first endpoint
	if selectedEndpoint == nil {
		selectedEndpoint = candidates[0].Endpoint
	}

	return selectedEndpoint
}
func (s *PredictedLatency) buildCompositeChoices(
	ctx context.Context,
	candidates []endpointPredictionResult,
	wkv, wq, wpref float64,
	total *int,
) []choice {

	// Normalize weights
	sumw := wkv + wq + wpref
	if sumw <= 0 {
		wkv, wq, wpref = 1, 0, 0
	} else {
		wkv /= sumw
		wq /= sumw
		wpref /= sumw
	}

	// Precompute queue stats
	minQ, maxQ := math.MaxInt32, -1
	queueCounts := make(map[string]int, len(candidates))
	for _, e := range candidates {
		q := e.Endpoint.GetMetrics().WaitingQueueSize
		queueCounts[e.Endpoint.GetMetadata().String()] = q
		if q < minQ {
			minQ = q
		}
		if q > maxQ {
			maxQ = q
		}
	}
	den := float64(maxQ - minQ)

	choices := make([]choice, 0, len(candidates))
	for _, p := range candidates {
		q := queueCounts[p.Endpoint.GetMetadata().String()]
		relQueue := 1.0
		if den > 0 {
			relQueue = (float64(maxQ-q) / den)
		}

		kvUsage := p.Endpoint.GetMetrics().KVCacheUsagePercent
		kvFree := (1.0 - kvUsage)
		prefix := (p.PrefixCacheScore)

		composite := wkv*kvFree + wq*relQueue + wpref*prefix
		w := int(math.Round(float64(minWeight) + (float64(wMax-minWeight) * composite)))
		*total += w
		choices = append(choices, choice{endpointName: p.Endpoint, weight: w})

		log.FromContext(ctx).V(logutil.TRACE).Info("Composite (neg/pos) score",
			"endpoint", p.Endpoint.GetMetadata().String(),
			"kvUsage", kvUsage, "kvFree", kvFree,
			"queue", q, "relQueue", relQueue,
			"prefix", prefix,
			"wkv", wkv, "wq", wq, "wprefix", wpref,
			"composite", composite, "weight", w)
	}
	return choices
}
