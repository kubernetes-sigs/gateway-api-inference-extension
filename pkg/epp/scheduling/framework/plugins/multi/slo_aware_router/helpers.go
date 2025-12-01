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
	"math/rand"

	"sigs.k8s.io/controller-runtime/pkg/log"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

func (s *SLOAwareRouter) selectFromCompositeScores(ctx context.Context, allPreds []podPredictionResult, r *rand.Rand, strategy headroomStrategy) schedulingtypes.Pod {
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
	selectedPod := s.performWeightedRandomSelection(choices, total, allPreds, r)
	return selectedPod
}
func (s *SLOAwareRouter) performWeightedRandomSelection(weightedChoices []choice, total int, candidates []podPredictionResult, r *rand.Rand) schedulingtypes.Pod {
	if total == 0 {
		return nil
	}
	logger := log.FromContext(context.Background())
	// Check if MAX_SCORE_SELECTION env variable is set
	if s.config.SelectionMode == string(podSelectionMax) {

		logger.V(logutil.DEBUG).Info("Pod selection mode: MAX - selecting pod with highest weight")
		maxWeight := 0
		var selectedPod schedulingtypes.Pod
		for _, c := range weightedChoices {
			if c.weight > maxWeight {
				maxWeight = c.weight
				selectedPod = c.podName
			}
		}
		if selectedPod != nil {
			return selectedPod
		}
		// Fallback to first pod if no selection made
		return candidates[0].Pod
	}

	// Original weighted random selection logic
	logger.V(logutil.DEBUG).Info("Pod selection mode: LINEAR - performing weighted random selection")
	idx := r.Intn(total)
	var selectedPod schedulingtypes.Pod

	for _, c := range weightedChoices {
		if idx < c.weight {
			selectedPod = c.podName
			break
		}
		idx -= c.weight
	}

	// If no pod was selected (shouldn't happen), fallback to first pod
	if selectedPod == nil {
		selectedPod = candidates[0].Pod
	}

	return selectedPod
}
func (s *SLOAwareRouter) buildCompositeChoices(
	ctx context.Context,
	candidates []podPredictionResult,
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
	for _, p := range candidates {
		q := p.Pod.GetMetrics().WaitingQueueSize
		queueCounts[p.Pod.GetPod().String()] = q
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
		q := queueCounts[p.Pod.GetPod().String()]
		relQueue := 1.0
		if den > 0 {
			relQueue = (float64(maxQ-q) / den)
		}

		kvUsage := p.Pod.GetMetrics().KVCacheUsagePercent
		kvFree := (1.0 - kvUsage)
		prefix := (p.PrefixCacheScore)

		composite := wkv*kvFree + wq*relQueue + wpref*prefix
		w := int(math.Round(float64(minWeight) + (float64(wMax-minWeight) * composite)))
		*total += w
		choices = append(choices, choice{podName: p.Pod, weight: w})

		log.FromContext(ctx).V(logutil.TRACE).Info("Composite (neg/pos) score",
			"pod", p.Pod.GetPod().String(),
			"kvUsage", kvUsage, "kvFree", kvFree,
			"queue", q, "relQueue", relQueue,
			"prefix", prefix,
			"wkv", wkv, "wq", wq, "wprefix", wpref,
			"composite", composite, "weight", w)
	}
	return choices
}
