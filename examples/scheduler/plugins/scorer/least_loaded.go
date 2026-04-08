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

// Package scorer demonstrates how to implement a custom scheduling Scorer plugin.
//
// A Scorer assigns a score in [0, 1] to each candidate endpoint. Higher is better.
// After all Scorers run, the framework multiplies each score by the scorer's weight
// and sums them to produce a final weighted score per endpoint.
//
// To implement your own Scorer:
//
//  1. Define a struct for any scorer state / config.
//  2. Implement plugin.Plugin       — return a TypedName.
//  3. Implement scheduling.Scorer:
//     - Category() — return Affinity, Distribution, or Balance.
//     - Score()    — return map[Endpoint]float64 with scores in [0, 1].
//  4. Provide a FactoryFunc and register it in RegisterAllPlugins() (see plugins/register.go).
package scorer

import (
	"context"
	"encoding/json"
	"fmt"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwksched "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

const PluginType = "least-loaded-scorer"

// compile-time interface check
var _ fwksched.Scorer = &LeastLoadedScorer{}

// LeastLoadedScorer scores each endpoint by how much spare LoRA adapter capacity
// it has:
//
//	score = 1 - len(ActiveModels) / MaxActiveModels
//
// ActiveModels tracks the LoRA adapters currently loaded on an endpoint, not
// fully fledged models. An endpoint running 1 of 4 possible LoRA adapters
// scores 0.75 (plenty of room), while one running 4 of 4 scores 0.0 (fully
// loaded). This encourages the scheduler to spread new LoRA adapter loads to
// less busy pods.
type LeastLoadedScorer struct {
	typedName fwkplugin.TypedName
}

// New creates a LeastLoadedScorer with default type and name.
func New() *LeastLoadedScorer {
	return &LeastLoadedScorer{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: PluginType},
	}
}

func (s *LeastLoadedScorer) TypedName() fwkplugin.TypedName { return s.typedName }

// Category returns Distribution because this scorer prefers spreading load
// across endpoints rather than concentrating requests on a few.
func (s *LeastLoadedScorer) Category() fwksched.ScorerCategory {
	return fwksched.Distribution
}

// Score returns a score in [0, 1] for each endpoint.
func (s *LeastLoadedScorer) Score(
	_ context.Context,
	_ *fwksched.CycleState,
	_ *fwksched.LLMRequest,
	endpoints []fwksched.Endpoint,
) map[fwksched.Endpoint]float64 {
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	for _, ep := range endpoints {
		m := ep.GetMetrics()
		if m.MaxActiveModels <= 0 {
			scores[ep] = 0.5
			continue
		}
		score := 1.0 - float64(len(m.ActiveModels))/float64(m.MaxActiveModels)
		scores[ep] = score
		fmt.Printf("  [scorer/%s] %-12s  active=%d  max=%d  → score=%.2f\n",
			PluginType, ep.GetMetadata().NamespacedName.Name,
			len(m.ActiveModels), m.MaxActiveModels, score)
	}
	return scores
}

// Factory creates a LeastLoadedScorer from a YAML config entry.
func Factory(name string, _ json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	s := New()
	s.typedName.Name = name
	return s, nil
}
