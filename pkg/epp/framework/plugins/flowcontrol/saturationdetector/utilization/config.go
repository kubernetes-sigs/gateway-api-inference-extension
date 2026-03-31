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

package utilization

import (
	"errors"
	"fmt"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
)

// Default configuration values
const (
	// DefaultQueueDepthThreshold is the default waiting queue size threshold.
	DefaultQueueDepthThreshold int = 5
	// DefaultKVCacheUtilThreshold is the default KV cache utilization (0.0 to 1.0) threshold.
	DefaultKVCacheUtilThreshold float64 = 0.8
	// DefaultMetricsStalenessThreshold defines how old metrics can be before they are considered
	// stale.
	DefaultMetricsStalenessThreshold time.Duration = 200 * time.Millisecond
	// defaultHeadroom is the default burst allowance (0%).
	defaultHeadroom float64 = 0.0
)

// apiConfig represents the external configuration schema for the utilization detector.
// It dictates how the plugin calculates pool-level saturation (to trigger backpressure) and
// endpoint-level limits (to filter out overloaded candidates during routing).
//
// It is designed to be deserialized from JSON via the plugin's raw parameters.
type apiConfig struct {
	// QueueDepthThreshold defines the saturation threshold for an endpoint's waiting queue depth.
	//
	// This limit serves as the "ideal" queue capacity for a single endpoint.
	// When combined with the KV Cache threshold, it defines PoolSaturation. When the pool
	// approaches or exceeds this aggregate capacity, the Flow Controller triggers backpressure
	// to buffer new traffic until capacity becomes available.
	//
	// Defaults to 5 if unset.
	QueueDepthThreshold *int `json:"queueDepthThreshold,omitempty"`

	// KVCacheUtilThreshold defines the saturation threshold for an endpoint's KV cache memory
	// utilization, expressed as a fraction in [0.0, 1.0].
	//
	// This limit serves as the "ideal" memory capacity for a single endpoint. The detector uses
	// a roofline model, taking the maximum of either the QueueDepth ratio or the KVCache ratio to
	// determine the endpoint's true saturation score.
	//
	// Defaults to 0.8 (80%) if unset.
	KVCacheUtilThreshold *float64 `json:"kvCacheUtilThreshold,omitempty"`

	// MetricsStalenessThreshold defines how old an endpoint's metrics can be before they are
	// considered stale. Stale endpoints are treated as 100% saturated to prevent routing traffic into
	// unobservable black holes.
	//
	// WARNING: In continuous batching architectures, the KV cache grows rapidly during the decode
	// phase. A high staleness threshold combined with a high KV cache utilization limit can lead to
	// routing traffic to an endpoint whose memory is silently expanding within the telemetry blind
	// spot.
	//
	// Defaults to 200ms if unset.
	MetricsStalenessThreshold *metav1.Duration `json:"metricsStalenessThreshold,omitempty"`

	// Headroom defines the allowed burst capacity above the ideal thresholds, expressed as a
	// multiplier (e.g., 0.2 for 20%).
	//
	// This parameter decouples pool-level backpressure from individual endpoint routing. While
	// PoolSaturation uses the strict ideal capacity to manage overall pool load, the Filter logic
	// evaluates relaxed limits (Threshold * (1 + Headroom)) to determine if a specific endpoint can
	// accept more work.
	//
	// Example: QueueDepthThreshold=5, Headroom=0.2.
	//
	//   - Backpressure: An endpoint with 6 queued requests is considered 120% full, pushing pool
	//     averages up.
	//   - Routing: The endpoint's hard filter limit is 6. The router can still send it requests
	//     (e.g., to satisfy high affinity) until it hits 6.
	//
	// Defaults to 0.0 (no burst allowed) if unset.
	Headroom *float64 `json:"headroom,omitempty"`
}

// Config is the internal, fully-validated configuration used by the detector.
type Config struct {
	QueueDepthThreshold       int
	KVCacheUtilThreshold      float64
	MetricsStalenessThreshold time.Duration
	Headroom                  float64
}

// buildConfig applies the configuration lifecycle (defaulting and validation) and translates the
// external schema into the internal domain model.
// The provided apiConfig is copied to prevent mutation side-effects.
func buildConfig(apiCfg *apiConfig) (*Config, error) {
	var safeCfg apiConfig
	if apiCfg != nil {
		safeCfg = *apiCfg
	}

	applyDefaults(&safeCfg)

	if err := validateConfig(&safeCfg); err != nil {
		return nil, fmt.Errorf("invalid utilization detector configuration: %w", err)
	}

	return &Config{
		QueueDepthThreshold:       *safeCfg.QueueDepthThreshold,
		KVCacheUtilThreshold:      *safeCfg.KVCacheUtilThreshold,
		MetricsStalenessThreshold: safeCfg.MetricsStalenessThreshold.Duration,
		Headroom:                  *safeCfg.Headroom,
	}, nil
}

// applyDefaults populates unset fields in the external configuration with their standard defaults.
func applyDefaults(cfg *apiConfig) {
	if cfg.QueueDepthThreshold == nil {
		cfg.QueueDepthThreshold = ptr.To(DefaultQueueDepthThreshold)
	}
	if cfg.KVCacheUtilThreshold == nil {
		cfg.KVCacheUtilThreshold = ptr.To(DefaultKVCacheUtilThreshold)
	}
	if cfg.MetricsStalenessThreshold == nil {
		cfg.MetricsStalenessThreshold = &metav1.Duration{Duration: DefaultMetricsStalenessThreshold}
	}
	if cfg.Headroom == nil {
		cfg.Headroom = ptr.To(defaultHeadroom)
	}
}

// validateConfig checks the constraints of the fully defaulted configuration.
// It aggregates all validation failures rather than failing on the first error.
func validateConfig(cfg *apiConfig) error {
	var errs []error

	if cfg.QueueDepthThreshold != nil && *cfg.QueueDepthThreshold <= 0 {
		errs = append(errs, fmt.Errorf("queueDepthThreshold must be strictly positive, got %d", *cfg.QueueDepthThreshold))
	}
	if cfg.KVCacheUtilThreshold != nil && (*cfg.KVCacheUtilThreshold <= 0.0 || *cfg.KVCacheUtilThreshold > 1.0) {
		errs = append(errs, fmt.Errorf("kvCacheUtilThreshold must be in (0.0, 1.0], got %f", *cfg.KVCacheUtilThreshold))
	}
	if cfg.MetricsStalenessThreshold != nil && cfg.MetricsStalenessThreshold.Duration <= 0 {
		errs = append(errs, fmt.Errorf("metricsStalenessThreshold must be strictly positive, got %v",
			cfg.MetricsStalenessThreshold.Duration))
	}
	if cfg.Headroom != nil && *cfg.Headroom < 0.0 {
		errs = append(errs, fmt.Errorf("headroom must be a non-negative value, got %f", *cfg.Headroom))
	}

	return errors.Join(errs...)
}
