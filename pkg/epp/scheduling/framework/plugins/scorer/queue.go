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

	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/scheduling"
	types "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
)

const (
	QueueScorerType = "queue-scorer"
)

// compile-time type assertion
var _ framework.Scorer = &QueueScorer{}

// QueueScorerFactory defines the factory function for QueueScorer.
func QueueScorerFactory(name string, _ json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	return NewQueueScorer().WithName(name), nil
}

// NewQueueScorer initializes a new QueueScorer and returns its pointer.
func NewQueueScorer() *QueueScorer {
	return &QueueScorer{
		typedName: plugins.TypedName{Type: QueueScorerType, Name: QueueScorerType},
	}
}

// QueueScorer scores list of candidate pods based on the pod's waiting queue size.
// the less waiting queue size the pod has, the higher score it will get (since it's more available to serve new request).
type QueueScorer struct {
	typedName plugins.TypedName
}

// TypedName returns the type and name tuple of this plugin instance.
func (s *QueueScorer) TypedName() plugins.TypedName {
	return s.typedName
}

// Category returns the preference the scorer applies when scoring candidate endpoints.
func (s *QueueScorer) Category() framework.ScorerCategory {
	return framework.Distribution
}

// Consumes returns the list of data that is consumed by the plugin.
func (s *QueueScorer) Consumes() map[string]any {
	return map[string]any{
		metrics.WaitingQueueSizeKey: int(0),
	}
}

// WithName sets the name of the scorer.
func (s *QueueScorer) WithName(name string) *QueueScorer {
	s.typedName.Name = name
	return s
}

// Score returns the scoring result for the given list of endpoints based on context.
func (s *QueueScorer) Score(_ context.Context, _ *types.CycleState, _ *types.LLMRequest, endpoints []types.Endpoint) map[types.Endpoint]float64 {
	minQueueSize := math.MaxInt
	maxQueueSize := math.MinInt

	// Iterate through the remaining endpoints to find min and max
	for _, endpoint := range endpoints {
		queueSize := endpoint.GetMetrics().WaitingQueueSize
		if queueSize < minQueueSize {
			minQueueSize = queueSize
		}
		if queueSize > maxQueueSize {
			maxQueueSize = queueSize
		}
	}

	// endpointScoreFunc calculates the score based on the queue size of each endpoint. Longer queue gets a lower score.
	endpointScoreFunc := func(endpoint types.Endpoint) float64 {
		if maxQueueSize == minQueueSize {
			// If all pods have the same queue size, return a neutral score
			return 1.0
		}
		return float64(maxQueueSize-endpoint.GetMetrics().WaitingQueueSize) / float64(maxQueueSize-minQueueSize)
	}

	// Create a map to hold the scores for each endpoint
	scores := make(map[types.Endpoint]float64, len(endpoints))
	for _, endpoint := range endpoints {
		scores[endpoint] = endpointScoreFunc(endpoint)
	}
	return scores
}
