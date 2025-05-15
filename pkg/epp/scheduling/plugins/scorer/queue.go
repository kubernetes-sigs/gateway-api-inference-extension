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

package scorer

import (
	"math"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

const (
	DefaultQueueScorerWeight = 1
)

type QueueScorer struct{}

func (q *QueueScorer) Name() string {
	return "queue"
}

func (q *QueueScorer) Score(ctx *types.SchedulingContext, pods []types.Pod) map[types.Pod]float64 {
	minQueueSize := math.MaxInt
	maxQueueSize := math.MinInt

	// Iterate through the remaining pods to find min and max
	for _, pod := range pods {
		queueSize := pod.GetMetrics().WaitingQueueSize
		if queueSize < minQueueSize {
			minQueueSize = queueSize
		}
		if queueSize > maxQueueSize {
			maxQueueSize = queueSize
		}
	}

	// podScoreFunc calculates the score based on the queue size of each pod. Longer queue gets a lower score.
	podScoreFunc := func(pod types.Pod) float64 {
		if maxQueueSize == minQueueSize {
			// If all pods have the same queue size, return a neutral score
			return 1.0
		}
		return float64(maxQueueSize-pod.GetMetrics().WaitingQueueSize) / float64(maxQueueSize-minQueueSize)
	}

	// Create a map to hold the scores for each pod
	scores := make(map[types.Pod]float64, len(pods))
	for _, pod := range pods {
		scores[pod] = podScoreFunc(pod)
	}
	return scores
}
