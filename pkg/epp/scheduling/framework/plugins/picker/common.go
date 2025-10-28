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
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

const (
	DefaultMaxNumOfEndpoints = 1 // common default to all pickers
)

// pickerParameters defines the common parameters for all pickers
type pickerParameters struct {
	MaxNumOfEndpoints int `json:"maxNumOfEndpoints"`
}

func shuffleScoredPods(scoredPods []*types.ScoredPod) {
	// Rand package is not safe for concurrent use, so we create a new instance.
	// Source: https://pkg.go.dev/math/rand/v2#pkg-overview
	randomGenerator := rand.New(rand.NewPCG(uint64(time.Now().UnixNano()), 0))

	// Shuffle in-place
	randomGenerator.Shuffle(len(scoredPods), func(i, j int) {
		scoredPods[i], scoredPods[j] = scoredPods[j], scoredPods[i]
	})
}
