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

package picker

import (
	"math/rand/v2"
	"sync"
	"time"
)

const (
	DefaultMaxNumOfEndpoints = 1 // common default to all pickers
)

// pickerParameters defines the common parameters for all pickers
type pickerParameters struct {
	MaxNumOfEndpoints int `json:"maxNumOfEndpoints"`
}

// safeRand is a thread-safe wrapper around rand.Rand
// to ensure that random operations are safe to use in concurrent environments.
type safeRand struct {
	r   *rand.Rand
	mut sync.Mutex
}

// NewSafeRand initializes a new safeRand.
func NewSafeRand(r *rand.Rand) *safeRand {
	if r == nil {
		seed := time.Now().UnixNano()
		r = rand.New(rand.NewPCG(uint64(seed), uint64(seed))) // default source if nil.
	}

	return &safeRand{r: r}
}

// Uint64 is a thread-safe method to get a random number.
func (r *safeRand) Uint64() uint64 {
	r.mut.Lock()
	defer r.mut.Unlock()

	return r.r.Uint64()
}

// Shuffle is a thread-safe method to shuffle a slice.
func (r *safeRand) Shuffle(n int, swap func(i int, j int)) {
	r.mut.Lock()
	defer r.mut.Unlock()

	r.r.Shuffle(n, swap)
}
