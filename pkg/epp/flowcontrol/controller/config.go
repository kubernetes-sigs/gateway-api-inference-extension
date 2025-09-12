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

package controller

import (
	"fmt"
	"time"
)

const (
	// defaultExpiryCleanupInterval is the default frequency for scanning for expired items.
	defaultExpiryCleanupInterval = 1 * time.Second
	// defaultProcessorReconciliationInterval is the default frequency for the supervisor loop.
	defaultProcessorReconciliationInterval = 5 * time.Second
	// defaultEnqueueChannelBufferSize is the default size of a worker's incoming request buffer.
	defaultEnqueueChannelBufferSize = 100
)

// Config holds the configuration for the `FlowController`.
type Config struct {
	// DefaultRequestTTL is the default Time-To-Live applied to requests that do not
	// specify their own TTL hint.
	// Optional: If zero, no TTL is applied by default and we rely solely on request context cancellation.
	DefaultRequestTTL time.Duration

	// ExpiryCleanupInterval is the interval at which each shard processor scans its queues for expired items.
	// Optional: Defaults to `defaultExpiryCleanupInterval` (1 second).
	ExpiryCleanupInterval time.Duration

	// ProcessorReconciliationInterval is the frequency at which the `FlowController`'s supervisor loop garbage collects
	// stale workers.
	// Optional: Defaults to `defaultProcessorReconciliationInterval` (5 seconds).
	ProcessorReconciliationInterval time.Duration

	// EnqueueChannelBufferSize is the size of the buffered channel that accepts incoming requests for each shard
	// processor. This buffer acts as a shock absorber, decoupling the high-frequency distributor from the processor's
	// serial execution loop and allowing the system to handle short bursts of traffic without blocking.
	// Optional: Defaults to `defaultEnqueueChannelBufferSize` (100).
	EnqueueChannelBufferSize int
}

// newConfig performs validation and initialization, returning a guaranteed-valid `Config` object.
// This is the required constructor for creating a new configuration.
// It does not mutate the input `cfg`.
func newConfig(cfg Config) (*Config, error) {
	newCfg := cfg.deepCopy()
	if err := newCfg.validateAndApplyDefaults(); err != nil {
		return nil, err
	}
	return newCfg, nil
}

// validateAndApplyDefaults checks the global configuration for validity and then mutates the receiver to populate any
// empty fields with system defaults.
func (c *Config) validateAndApplyDefaults() error {
	// --- Validation ---
	if c.DefaultRequestTTL < 0 {
		return fmt.Errorf("DefaultRequestTTL cannot be negative, but got %v", c.DefaultRequestTTL)
	}
	if c.ExpiryCleanupInterval < 0 {
		return fmt.Errorf("ExpiryCleanupInterval cannot be negative, but got %v", c.ExpiryCleanupInterval)
	}
	if c.ProcessorReconciliationInterval < 0 {
		return fmt.Errorf("ProcessorReconciliationInterval cannot be negative, but got %v",
			c.ProcessorReconciliationInterval)
	}
	if c.EnqueueChannelBufferSize < 0 {
		return fmt.Errorf("EnqueueChannelBufferSize cannot be negative, but got %d", c.EnqueueChannelBufferSize)
	}

	// --- Defaulting ---
	if c.ExpiryCleanupInterval == 0 {
		c.ExpiryCleanupInterval = defaultExpiryCleanupInterval
	}
	if c.ProcessorReconciliationInterval == 0 {
		c.ProcessorReconciliationInterval = defaultProcessorReconciliationInterval
	}
	if c.EnqueueChannelBufferSize == 0 {
		c.EnqueueChannelBufferSize = defaultEnqueueChannelBufferSize
	}
	return nil
}

// deepCopy creates a deep copy of the `Config` object.
func (c *Config) deepCopy() *Config {
	if c == nil {
		return nil
	}
	newCfg := &Config{
		DefaultRequestTTL:               c.DefaultRequestTTL,
		ExpiryCleanupInterval:           c.ExpiryCleanupInterval,
		ProcessorReconciliationInterval: c.ProcessorReconciliationInterval,
		EnqueueChannelBufferSize:        c.EnqueueChannelBufferSize,
	}
	return newCfg
}
