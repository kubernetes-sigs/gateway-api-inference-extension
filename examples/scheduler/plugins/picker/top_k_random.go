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

// Package picker demonstrates how to implement a custom scheduling Picker plugin.
//
// A Picker receives a list of scored endpoints (after Filters and Scorers have
// run) and selects the final endpoint(s) to route the request to.
//
// To implement your own Picker:
//
//  1. Define a struct for any picker state / config.
//  2. Implement plugin.Plugin       — return a TypedName.
//  3. Implement scheduling.Picker   — Pick() returns a ProfileRunResult.
//  4. Provide a FactoryFunc and register it in RegisterAllPlugins() (see plugins/register.go).
//
// The built-in MaxScorePicker always picks the single highest-scored endpoint.
// This TopKRandomPicker instead takes the top K candidates by score and then
// randomly selects one, which helps spread load when multiple endpoints have
// similar scores and avoids always hitting the same pod.
package picker

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand/v2"
	"slices"
	"time"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwksched "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

const (
	PluginType = "top-k-random-picker"
	DefaultK   = 2
)

// compile-time interface check
var _ fwksched.Picker = &TopKRandomPicker{}

// TopKRandomPicker sorts candidates by score descending, keeps the top K,
// and randomly picks one from them. This balances quality (only top candidates
// are considered) with load distribution (random selection among them).
type TopKRandomPicker struct {
	typedName fwkplugin.TypedName
	k         int
}

// New creates a TopKRandomPicker with the given K value.
func New(k int) *TopKRandomPicker {
	if k <= 0 {
		k = DefaultK
	}
	return &TopKRandomPicker{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: PluginType},
		k:         k,
	}
}

func (p *TopKRandomPicker) TypedName() fwkplugin.TypedName { return p.typedName }

// Pick selects one endpoint from the top K highest-scored candidates.
func (p *TopKRandomPicker) Pick(
	_ context.Context,
	_ *fwksched.CycleState,
	scoredEndpoints []*fwksched.ScoredEndpoint,
) *fwksched.ProfileRunResult {
	// Sort by score descending.
	slices.SortStableFunc(scoredEndpoints, func(a, b *fwksched.ScoredEndpoint) int {
		if a.Score > b.Score {
			return -1
		}
		if a.Score < b.Score {
			return 1
		}
		return 0
	})

	// Keep top K.
	topK := scoredEndpoints
	if p.k < len(topK) {
		topK = topK[:p.k]
	}

	// Randomly pick one from top K.
	rng := rand.New(rand.NewPCG(uint64(time.Now().UnixNano()), 0))
	winner := topK[rng.IntN(len(topK))]

	fmt.Printf("  [picker/%s] top-%d from %d candidates → %s (score=%.4f)\n",
		PluginType, p.k, len(scoredEndpoints),
		winner.GetMetadata().NamespacedName.Name, winner.Score)

	return &fwksched.ProfileRunResult{
		TargetEndpoints: []fwksched.Endpoint{winner},
	}
}

// topKRandomParameters holds the YAML-configurable parameters.
type topKRandomParameters struct {
	K int `json:"k"`
}

// Factory creates a TopKRandomPicker from a YAML config entry.
// Example YAML parameters: {"k": 3}
func Factory(name string, rawParameters json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := topKRandomParameters{K: DefaultK}
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &params); err != nil {
			return nil, fmt.Errorf("failed to parse parameters of '%s' picker: %w", PluginType, err)
		}
	}
	p := New(params.K)
	p.typedName.Name = name
	return p, nil
}
