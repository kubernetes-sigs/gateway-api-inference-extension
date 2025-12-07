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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

const (
	LoraAffinityScorerType = "lora-affinity-scorer"
)

// compile-time type assertion
var _ framework.Scorer = &LoraAffinityScorer{}

// LoraAffinityScorerFactory defines the factory function for LoraAffinityScorer.
func LoraAffinityScorerFactory(name string, _ json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	return NewLoraAffinityScorer().WithName(name), nil
}

// NewLoraAffinityScorer initializes a new LoraAffinityScorer and returns its pointer.
func NewLoraAffinityScorer() *LoraAffinityScorer {
	return &LoraAffinityScorer{
		typedName: plugins.TypedName{Type: LoraAffinityScorerType, Name: LoraAffinityScorerType},
	}
}

// LoraAffinityScorer scores list of candidate pods based on Lora affinity and availability.
type LoraAffinityScorer struct {
	typedName plugins.TypedName
}

// TypedName returns the type and name tuple of this plugin instance.
func (s *LoraAffinityScorer) TypedName() plugins.TypedName {
	return s.typedName
}

// Consumes returns the list of data that is consumed by the plugin.
func (s *LoraAffinityScorer) Consumes() map[string]any {
	return map[string]any{
		metrics.ActiveModelsKey:  map[string]int{},
		metrics.WaitingModelsKey: map[string]int{},
	}
}

// WithName sets the name of the scorer.
func (s *LoraAffinityScorer) WithName(name string) *LoraAffinityScorer {
	s.typedName.Name = name
	return s
}

func (s *LoraAffinityScorer) Score(_ context.Context, _ *types.CycleState, request *types.LLMRequest, pods []types.Pod) map[types.Pod]float64 {
	scores := make(map[types.Pod]float64, len(pods))

	// Assign a score to each pod for loading the target adapter.
	for _, pod := range pods {
		_, active := pod.GetMetrics().ActiveModels[request.TargetModel]
		_, waiting := pod.GetMetrics().WaitingModels[request.TargetModel]

		// Determine the model server's suitability score based on adapter load status and capacity.
		switch {
		// Ideal: The adapter is already active on this model server.
		case active:
			scores[pod] = 1.0
		// Good: The model server has capacity to load at least one more adapter.
		case len(pod.GetMetrics().ActiveModels)+len(pod.GetMetrics().WaitingModels) < pod.GetMetrics().MaxActiveModels:
			scores[pod] = 0.8
		// Moderate: The adapter is already in the queue to be loaded on this model server.
		case waiting:
			scores[pod] = 0.6
		// Unsuitable: The model server has reached its maximum capacity and cannot load the adapter.
		default:
			scores[pod] = 0.0
		}
	}

	return scores
}
