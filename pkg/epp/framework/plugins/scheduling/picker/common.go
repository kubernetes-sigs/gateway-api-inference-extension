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

// Package picker provides the standard implementations of the Picker interface for the scheduling
// framework. These plugins represent the final phase of the scheduling cycle, selecting target
// endpoints from the scored candidates.
//
// For detailed descriptions of the available pickers and their configuration, see the package
// README.md.
package picker

import (
	"math/rand/v2"
	"sync"
	"time"

	types "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

const (
	// DefaultMaxNumOfEndpoints is the fallback maximum number of endpoints to pick if not specified
	// in the configuration.
	DefaultMaxNumOfEndpoints = 1
)

// PickerParameters defines the common parameters for all pickers
type PickerParameters struct {
	MaxNumOfEndpoints int `json:"maxNumOfEndpoints"`
}

type lockedRand struct {
	mu   sync.Mutex
	rand *rand.Rand
}

func newLockedRand() *lockedRand {
	seed := uint64(time.Now().UnixNano())

	return &lockedRand{
		rand: rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15)),
	}
}

func (r *lockedRand) Float64() float64 {
	r.mu.Lock()
	defer r.mu.Unlock()

	return r.rand.Float64()
}

func (r *lockedRand) Shuffle(n int, swap func(i, j int)) {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.rand.Shuffle(n, swap)
}

// PickerRand is a thread-safe random number generator shared by all pickers to avoid seeding
// overhead and ensure controlled randomization.
var PickerRand = newLockedRand()

// ShuffleScoredEndpoints randomizes the order of the given scored candidates in-place.
func ShuffleScoredEndpoints(scoredEndpoints []*types.ScoredEndpoint) {
	// Shuffle in-place
	PickerRand.Shuffle(len(scoredEndpoints), func(i, j int) {
		scoredEndpoints[i], scoredEndpoints[j] = scoredEndpoints[j], scoredEndpoints[i]
	})
}
