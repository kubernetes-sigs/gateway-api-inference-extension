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
	"context"
	"encoding/json"
	"math"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

const (
	RunningRequestsSizeScorerType = "running-requests-size-scorer"
)

// compile-time type assertion
var _ framework.Scorer = &RunningRequestsSizeScorer{}

// RunningRequestsSizeScorerFactory defines the factory function for RunningRequestsSizeScorer.
func RunningRequestsSizeScorerFactory(name string, _ json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	return NewRunningRequestsSizeScorer().WithName(name), nil
}

// NewRunningRequestsSizeScorer initializes a new RunningRequestsSizeScorer and returns its pointer.
func NewRunningRequestsSizeScorer() *RunningRequestsSizeScorer {
	return &RunningRequestsSizeScorer{
		typedName: plugins.TypedName{Type: RunningRequestsSizeScorerType, Name: RunningRequestsSizeScorerType},
	}
}

// RunningRequestsSizeScorer scores list of candidate pods based on the pod's running request size.
// the less running request size the pod has, the higher score it will get (since it's more available to serve new request).
type RunningRequestsSizeScorer struct {
	typedName plugins.TypedName
}

// TypedName returns the type and name tuple of this plugin instance.
func (s *RunningRequestsSizeScorer) TypedName() plugins.TypedName {
	return s.typedName
}

// Consumes returns the list of data that is consumed by the plugin.
func (s *RunningRequestsSizeScorer) Consumes() map[string]any {
	return map[string]any{
		metrics.RunningRequestsSizeKey: int(0),
	}
}

// WithName sets the name of the scorer.
func (s *RunningRequestsSizeScorer) WithName(name string) *RunningRequestsSizeScorer {
	s.typedName.Name = name
	return s
}

// Score returns the scoring result for the given list of pods based on context.
func (s *RunningRequestsSizeScorer) Score(_ context.Context, _ *types.CycleState, _ *types.LLMRequest, pods []types.Pod) map[types.Pod]float64 {
	minQueueSize := math.MaxInt
	maxQueueSize := math.MinInt

	// Iterate through the remaining pods to find min and max
	for _, pod := range pods {
		queueSize := pod.GetMetrics().RunningRequestsSize
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
		return float64(maxQueueSize-pod.GetMetrics().RunningRequestsSize) / float64(maxQueueSize-minQueueSize)
	}

	// Create a map to hold the scores for each pod
	scores := make(map[types.Pod]float64, len(pods))
	for _, pod := range pods {
		scores[pod] = podScoreFunc(pod)
	}
	return scores
}
