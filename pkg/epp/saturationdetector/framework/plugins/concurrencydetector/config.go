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

package concurrencydetector

// Config holds the configuration for the Concurrency Detector.
type Config struct {
	// MaxConcurrency defines the saturation threshold for a backend.
	//
	// This limit serves as the "ideal" capacity for a backend. When the number of active requests on a replica reaches
	// this value, that specific backend is considered "full" for the purpose of global saturation detection. If all
	// available replicas are full (>= MaxConcurrency), the Detector signals saturation to the Flow Controller, which will
	// trigger backpressure.
	//
	// Defaults to 100 if unset.
	MaxConcurrency int64 `json:"maxConcurrency"`

	// Headroom defines the allowed burst capacity above MaxConcurrency for specific pod scheduling, expressed as a
	// fraction in [0.0, 1.0].
	//
	// While IsSaturated uses MaxConcurrency as the "ideal" capacity limit per pod to determine if the pool is full
	// (rejecting admission if ALL pods are >= MaxConcurrency), the Filter logic uses (MaxConcurrency * (1 + Headroom))
	// to determine if a specific pod is capable of accepting more work.
	//
	// Example: MaxConcurrency=100 (per pod), Headroom=0.2.
	// - Global Saturation: Triggered when all pods have >= 100 active requests.
	//   This stops new requests from entering the scheduling loop, enforcing pool-average concurrency.
	// - Pod Filter Limit: 120 active requests.
	//   The scheduler can still assign a request to a pod with 110 requests to satisfy affinity rules, as long as it
	//   stays under 120.
	//
	// This allows the system to maintain a target average load (100) while giving the scheduler flexibility to handle
	// hot-spots or affinity (up to 120).
	//
	// Defaults to 0.0 (no burst allowed).
	Headroom float64 `json:"headroom"`

	// ConcurrencyMode defines the mode of concurrency detection.
	//
	// Valid values are:
	// - "requests": use request count for concurrency detection.
	// - "tokens": use token count for concurrency detection.
	// TODO: Should there be an option to check for both?
	//
	// Defaults to "requests" if unset.
	ConcurrencyMode string `json:"concurrencyMode"`

	// MaxTokenConcurrency defines the maximum number of tokens allowed for a request.
	//
	// This limit is used to prevent requests from consuming too many tokens.
	//
	// Defaults to 1000000 if unset.
	MaxTokenConcurrency int64 `json:"maxTokenConcurrency"`

}

const (
	// DefaultMaxConcurrency is the safe baseline for many LLM serving engines.
	DefaultMaxConcurrency = 100
	// DefaultHeadroom is the default burst allowance (0%).
	DefaultHeadroom = 0.0
	// DefaultConcurrencyMode is the default mode to check for concurency. We default to "requests"to retain existing behvaiour.
	DefaultConcurrencyMode = "requests"
	// DefaultMaxTokenConcurrency is the maximum number of tokens allowed for a request.
	DefaultMaxTokenConcurrency = 1000000
)
