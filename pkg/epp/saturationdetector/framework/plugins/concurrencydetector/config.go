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

import (
	"errors"
	"fmt"

	"k8s.io/utils/ptr"
)

// apiConfig represents the external configuration schema for the concurrency detector.
// It dictates how the plugin calculates pool-level saturation (to trigger backpressure) and
// endpoint-level limits (to filter out overloaded candidates during routing).
//
// It is designed to be deserialized from JSON via the plugin's raw parameters.
type apiConfig struct {
	// MaxConcurrency defines the request-based saturation threshold for an endpoint.
	//
	// This limit serves as the "ideal" request capacity for a single endpoint. The plugin aggregates
	// the active requests across all endpoints and compares them against the aggregate pool capacity
	// (Total Endpoints * MaxConcurrency) to compute PoolSaturation. When the pool approaches or
	// exceeds this aggregate capacity, the Flow Controller triggers backpressure to buffer new
	// traffic until capacity becomes available.
	//
	// Defaults to 100 if unset.
	MaxConcurrency *int64 `json:"maxConcurrency,omitempty"`

	// Headroom defines the allowed burst capacity above the ideal threshold (MaxConcurrency or
	// MaxTokenConcurrency), expressed as a multiplier (e.g., 0.2 for 20%).
	//
	// This parameter decouples pool-level backpressure from individual endpoint routing. While
	// PoolSaturation uses the strict ideal capacity to manage overall pool load, the Filter logic
	// uses EndpointLimit (Capacity * (1 + Headroom)) to determine if a specific endpoint can accept
	// more work.
	//
	// Example: MaxConcurrency=100, Headroom=0.2.
	//
	//   - Backpressure: An endpoint with 110 requests is considered 110% full, pushing pool averages up.
	//   - Routing: The endpoint's hard filter limit is 120. The scheduling layer can still send it
	//     requests (e.g., to satisfy high affinity) until it hits 120.
	//
	// Defaults to 0.0 (no burst allowed) if unset.
	Headroom *float64 `json:"headroom,omitempty"`

	// ConcurrencyMode defines the mode of concurrency detection.
	//
	// Valid values are:
	// - "requests": use discrete request counts for capacity accounting.
	// - "tokens": use estimated token counts for capacity accounting.
	//
	// Defaults to "requests" if unset.
	ConcurrencyMode *concurrencyMode `json:"concurrencyMode,omitempty"`

	// MaxTokenConcurrency defines the token-based saturation threshold for an endpoint.
	//
	// This is the "tokens" mode equivalent of MaxConcurrency. It represents the "ideal" token
	// capacity per endpoint. It drives both the pool saturation calculation (for backpressure) and,
	// combined with Headroom, the per-endpoint filtering limits (for routing).
	//
	// Defaults to 1000000 if unset.
	MaxTokenConcurrency *int64 `json:"maxTokenConcurrency,omitempty"`
}

// concurrencyMode is the concurrency detection mode.
type concurrencyMode string

const (
	// modeRequests uses request count for concurrency detection.
	modeRequests concurrencyMode = "requests"
	// modeTokens uses token count for concurrency detection.
	modeTokens concurrencyMode = "tokens"
)

const (
	// defaultMaxConcurrency is the safe baseline for many LLM serving engines.
	defaultMaxConcurrency int64 = 100
	// defaultHeadroom is the default burst allowance (0%).
	defaultHeadroom float64 = 0.0
	// defaultConcurrencyMode is used when ConcurrencyMode is unset.
	defaultConcurrencyMode = modeRequests
	// defaultMaxTokenConcurrency is the default maximum number of tokens allowed per endpoint.
	defaultMaxTokenConcurrency int64 = 1000000
)

// config is the internal, fully-validated configuration used by the detector.
type config struct {
	maxConcurrency      int64
	headroom            float64
	mode                concurrencyMode
	maxTokenConcurrency int64
}

// buildConfig applies the configuration lifecycle (defaulting and validation) and translates the
// external schema into the internal domain model.
// The provided apiConfig is copied to prevent mutation side-effects.
func buildConfig(apiCfg *apiConfig) (*config, error) {
	var safeCfg apiConfig
	if apiCfg != nil {
		safeCfg = *apiCfg
	}

	applyDefaults(&safeCfg)

	if err := validateConfig(&safeCfg); err != nil {
		return nil, fmt.Errorf("invalid concurrency detector configuration: %w", err)
	}

	return &config{
		maxConcurrency:      *safeCfg.MaxConcurrency,
		headroom:            *safeCfg.Headroom,
		mode:                *safeCfg.ConcurrencyMode,
		maxTokenConcurrency: *safeCfg.MaxTokenConcurrency,
	}, nil
}

// applyDefaults populates unset fields in the external configuration with their standard defaults.
func applyDefaults(cfg *apiConfig) {
	if cfg.MaxConcurrency == nil {
		cfg.MaxConcurrency = ptr.To(defaultMaxConcurrency)
	}
	if cfg.Headroom == nil {
		cfg.Headroom = ptr.To(defaultHeadroom)
	}
	if cfg.ConcurrencyMode == nil {
		cfg.ConcurrencyMode = ptr.To(defaultConcurrencyMode)
	}
	if cfg.MaxTokenConcurrency == nil {
		cfg.MaxTokenConcurrency = ptr.To(defaultMaxTokenConcurrency)
	}
}

// validateConfig checks the constraints of the fully defaulted configuration.
// It aggregates all validation failures.
func validateConfig(cfg *apiConfig) error {
	var errs []error

	if cfg.MaxConcurrency != nil && *cfg.MaxConcurrency <= 0 {
		errs = append(errs, fmt.Errorf("maxConcurrency must be strictly positive, got %d", *cfg.MaxConcurrency))
	}
	if cfg.Headroom != nil && *cfg.Headroom < 0.0 {
		errs = append(errs, fmt.Errorf("headroom must be a non-negative value, got %f", *cfg.Headroom))
	}
	if cfg.MaxTokenConcurrency != nil && *cfg.MaxTokenConcurrency <= 0 {
		errs = append(errs, fmt.Errorf("maxTokenConcurrency must be strictly positive, got %d", *cfg.MaxTokenConcurrency))
	}

	if cfg.ConcurrencyMode != nil {
		switch *cfg.ConcurrencyMode {
		case modeRequests, modeTokens:
			// Valid
		default:
			errs = append(errs, fmt.Errorf("unsupported concurrencyMode: %q", *cfg.ConcurrencyMode))
		}
	}

	return errors.Join(errs...)
}
