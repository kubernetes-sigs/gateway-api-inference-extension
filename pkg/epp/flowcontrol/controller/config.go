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

	configapi "sigs.k8s.io/gateway-api-inference-extension/apix/config/v1alpha1"
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

// ConfigOption is a functional option for configuring the FlowController.
type ConfigOption func(*Config)

// NewConfigFromAPI creates a new Config from the API configuration.
func NewConfigFromAPI(apiConfig *configapi.FlowControlConfig) (*Config, error) {
	opts := make([]ConfigOption, 0, 2)
	if apiConfig != nil {
		if apiConfig.DefaultRequestTTL != nil {
			opts = append(opts, WithDefaultRequestTTL(apiConfig.DefaultRequestTTL.Duration))
		}
		if apiConfig.ExpiryCleanupInterval != nil {
			opts = append(opts, WithExpiryCleanupInterval(apiConfig.ExpiryCleanupInterval.Duration))
		}
	}
	return NewConfig(opts...)
}

// NewConfig creates a new Config with the given options, applying defaults and validation.
func NewConfig(opts ...ConfigOption) (*Config, error) {
	c := &Config{
		ExpiryCleanupInterval:           defaultExpiryCleanupInterval,
		ProcessorReconciliationInterval: defaultProcessorReconciliationInterval,
		EnqueueChannelBufferSize:        defaultEnqueueChannelBufferSize,
	}

	for _, opt := range opts {
		opt(c)
	}

	if err := c.validate(); err != nil {
		return nil, err
	}
	return c, nil
}

// WithDefaultRequestTTL sets the default request TTL.
func WithDefaultRequestTTL(d time.Duration) ConfigOption {
	return func(c *Config) {
		c.DefaultRequestTTL = d
	}
}

// WithExpiryCleanupInterval sets the expiry cleanup interval.
func WithExpiryCleanupInterval(d time.Duration) ConfigOption {
	return func(c *Config) {
		c.ExpiryCleanupInterval = d
	}
}

// WithProcessorReconciliationInterval sets the processor reconciliation interval.
func WithProcessorReconciliationInterval(d time.Duration) ConfigOption {
	return func(c *Config) {
		c.ProcessorReconciliationInterval = d
	}
}

// WithEnqueueChannelBufferSize sets the size of the enqueue channel buffer.
func WithEnqueueChannelBufferSize(size int) ConfigOption {
	return func(c *Config) {
		c.EnqueueChannelBufferSize = size
	}
}

// validate checks the configuration for validity.
func (c *Config) validate() error {
	if c.DefaultRequestTTL < 0 {
		return fmt.Errorf("DefaultRequestTTL cannot be negative, but got %v", c.DefaultRequestTTL)
	}
	if c.ExpiryCleanupInterval <= 0 {
		return fmt.Errorf("ExpiryCleanupInterval must be positive, but got %v", c.ExpiryCleanupInterval)
	}
	if c.ProcessorReconciliationInterval <= 0 {
		return fmt.Errorf("ProcessorReconciliationInterval must be positive, but got %v", c.ProcessorReconciliationInterval)
	}
	if c.EnqueueChannelBufferSize < 0 {
		return fmt.Errorf("EnqueueChannelBufferSize cannot be negative, but got %d", c.EnqueueChannelBufferSize)
	}
	return nil
}
