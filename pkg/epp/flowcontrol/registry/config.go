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

package registry

import (
	"errors"
	"fmt"
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/interflow"
	intra "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/intraflow/dispatch"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/intraflow/dispatch/fcfs"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue"
)

// =============================================================================
// Default Values
// =============================================================================

const (
	// defaultPriorityBandMaxBytes is the default global capacity for a priority band if not explicitly configured.
	// It is set to 1 GB.
	defaultPriorityBandMaxBytes uint64 = 1_000_000_000
	// defaultIntraFlowDispatchPolicy is the default policy for selecting items within a single flow's queue.
	defaultIntraFlowDispatchPolicy intra.RegisteredPolicyName = fcfs.FCFSPolicyName
	// defaultInterFlowDispatchPolicy is the default policy for selecting which flow's queue to service next.
	defaultInterFlowDispatchPolicy interflow.RegisteredPolicyName = interflow.BestHeadPolicyName
	// defaultQueue is the default queue implementation for flows.
	defaultQueue queue.RegisteredQueueName = queue.ListQueueName
	// defaultInitialShardCount is the default number of parallel shards to create when the registry is initialized.
	defaultInitialShardCount int = 1
	// defaultFlowGCTimeout is the default duration of inactivity after which an idle flow is garbage collected.
	// This also serves as the interval for the periodic garbage collection scan.
	defaultFlowGCTimeout time.Duration = 5 * time.Minute
	// defaultEventChannelBufferSize is the default size of the buffered channel for control plane events.
	defaultEventChannelBufferSize int = 4096
)

// =============================================================================
// Global Configuration (FlowRegistry)
// =============================================================================

// Config holds the master configuration for the entire `FlowRegistry`.
// It serves as the top-level blueprint, defining global settings and the templates for its priority bands.
//
// This configuration is validated and defaulted once at startup. It is then used to generate partitioned `ShardConfig`
// instances for each internal `registryShard`.
type Config struct {
	// MaxBytes defines an optional, global maximum total byte size limit aggregated across all priority bands and shards.
	// The `controller.FlowController` enforces this limit in addition to per-band capacity limits.
	// A value of 0 signifies that this global limit is ignored, and only per-band limits apply.
	// Optional: Defaults to 0.
	MaxBytes uint64

	// PriorityBands defines the set of priority band templates managed by the `FlowRegistry`.
	// The configuration for each band, including its default policies and queue types, is specified here.
	// These templates are used to generate partitioned `ShardPriorityBandConfig`s.
	// Required: At least one `PriorityBandConfig` must be provided for a functional registry.
	PriorityBands []PriorityBandConfig

	// InitialShardCount specifies the number of parallel shards to create when the registry is initialized.
	// This value must be greater than zero.
	// Optional: Defaults to `defaultInitialShardCount` (1).
	InitialShardCount int

	// FlowGCTimeout defines the interval at which the registry scans for and garbage collects idle flows. A flow is
	// collected if it has been observed to be Idle for at least one full scan interval.
	// Optional: Defaults to `defaultFlowGCTimeout` (5 minutes).
	FlowGCTimeout time.Duration

	// EventChannelBufferSize defines the size of the buffered channel used for internal control plane events.
	// A larger buffer can absorb larger bursts of events (e.g., from many queues becoming non-empty simultaneously)
	// without blocking the data path, but consumes more memory.
	// This value must be greater than zero.
	// Optional: Defaults to `defaultEventChannelBufferSize` (4096).
	EventChannelBufferSize int

	// priorityBandMap provides O(1) lookups of `PriorityBandConfig` by priority level.
	// Crucially, this map serves as a correctness mechanism. It ensures that accessors return a safe, stable pointer to
	// the correct element within this specific configuration instance, preventing common "pointer-to-loop-variable"
	// errors, especially across deep copies or partitioning.
	// It is populated during validation and when the config is copied or partitioned.
	priorityBandMap map[int]*PriorityBandConfig

	// Factory functions used for plugin instantiation during configuration validation.
	// These enable dependency injection for unit testing the validation logic.
	interFlowDispatchPolicyFactory interFlowDispatchPolicyFactory
	intraFlowDispatchPolicyFactory intraFlowDispatchPolicyFactory
	queueFactory                   queueFactory
}

// PriorityBandConfig defines the configuration template for a single priority band.
// It establishes the default behaviors (such as queueing and dispatch policies) and total capacity limits for all flows
// that operate at this priority level.
type PriorityBandConfig struct {
	// Priority is the unique numerical priority level for this band.
	// Convention: Highest numeric value corresponds to highest priority (centered on 0).
	// Required.
	Priority int

	// PriorityName is a human-readable name for this priority band (e.g., "Critical", "Standard").
	// It must be unique across all priority bands in the configuration.
	// Required.
	PriorityName string

	// IntraFlowDispatchPolicy specifies the default name of the policy used to select a request from within a single
	// flow's queue in this band.
	// Optional: Defaults to `defaultIntraFlowDispatchPolicy` ("FCFS").
	IntraFlowDispatchPolicy intra.RegisteredPolicyName

	// InterFlowDispatchPolicy specifies the name of the policy used to select which flow's queue to service next from
	// this band.
	// Optional: Defaults to `defaultInterFlowDispatchPolicy` ("BestHead").
	InterFlowDispatchPolicy interflow.RegisteredPolicyName

	// Queue specifies the default name of the `framework.SafeQueue` implementation for flow queues in this band.
	// Optional: Defaults to `defaultQueue` ("ListQueue").
	Queue queue.RegisteredQueueName

	// MaxBytes defines the maximum total byte size for this priority band, aggregated across all shards.
	// Optional: Defaults to `defaultPriorityBandMaxBytes` (1 GB).
	MaxBytes uint64
}

// =============================================================================
// Shard-Level Configuration
// =============================================================================

// ShardConfig holds the partitioned configuration for a single `registryShard`.
// It contains only the data relevant to that shard and is derived from the global `Config`.
type ShardConfig struct {
	// MaxBytes is this shard's partitioned portion of the global `Config.MaxBytes` limit.
	MaxBytes uint64

	// PriorityBands holds the partitioned configuration for each priority band managed by this shard.
	PriorityBands []ShardPriorityBandConfig

	// priorityBandMap provides O(1) lookups of `ShardPriorityBandConfig` by priority level.
	// It serves as a correctness mechanism, ensuring that accessors return a safe, stable pointer to the correct element
	// within this specific shard configuration instance.
	priorityBandMap map[int]*ShardPriorityBandConfig
}

// ShardPriorityBandConfig holds the partitioned configuration for a single priority band within a single shard.
type ShardPriorityBandConfig struct {
	// Priority is the unique numerical priority level for this band.
	Priority int
	// PriorityName is a unique human-readable name for this priority band.
	PriorityName string
	// IntraFlowDispatchPolicy is the name of the policy for dispatch within a flow's queue.
	IntraFlowDispatchPolicy intra.RegisteredPolicyName
	// InterFlowDispatchPolicy is the name of the policy for dispatch between flow queues.
	InterFlowDispatchPolicy interflow.RegisteredPolicyName
	// Queue is the name of the queue implementation to use.
	Queue queue.RegisteredQueueName
	// MaxBytes is this shard's partitioned portion of this band's global capacity limit.
	// The `controller.FlowController` enforces this limit for this specific shard.
	MaxBytes uint64
}

// getBandConfig finds and returns the shard-level configuration for a specific priority level.
// Returns an error wrapping `contracts.ErrPriorityBandNotFound` if the priority is not configured.
func (sc *ShardConfig) getBandConfig(priority int) (*ShardPriorityBandConfig, error) {
	if band, ok := sc.priorityBandMap[priority]; ok {
		return band, nil
	}
	return nil, fmt.Errorf("shard config for priority %d not found: %w", priority, contracts.ErrPriorityBandNotFound)
}

// =============================================================================
// Internal Methods and Helpers
// =============================================================================

// --- Validation and Defaulting ---

// ValidateAndApplyDefaults checks the global configuration for validity and then creates a new `Config` object,
// populating any empty fields with system defaults.
// It does not mutate the receiver.
func (c *Config) ValidateAndApplyDefaults() (*Config, error) {
	cfg := c.deepCopy()

	// Apply defaults to top-level fields.
	if cfg.InitialShardCount <= 0 {
		cfg.InitialShardCount = defaultInitialShardCount
	}
	if cfg.FlowGCTimeout <= 0 {
		cfg.FlowGCTimeout = defaultFlowGCTimeout
	}
	if cfg.EventChannelBufferSize <= 0 {
		cfg.EventChannelBufferSize = defaultEventChannelBufferSize
	}

	// Ensure the DI factories are initialized for production use if `NewConfig` was called without options.
	if cfg.interFlowDispatchPolicyFactory == nil {
		cfg.interFlowDispatchPolicyFactory = interflow.NewPolicyFromName
	}
	if cfg.intraFlowDispatchPolicyFactory == nil {
		cfg.intraFlowDispatchPolicyFactory = intra.NewPolicyFromName
	}
	if cfg.queueFactory == nil {
		cfg.queueFactory = queue.NewQueueFromName
	}

	if len(cfg.PriorityBands) == 0 {
		return nil, errors.New("config validation failed: at least one priority band must be defined")
	}

	// Validate and default each priority band.
	priorities := make(map[int]struct{})
	priorityNames := make(map[string]struct{})
	cfg.priorityBandMap = make(map[int]*PriorityBandConfig, len(cfg.PriorityBands))

	for i := range cfg.PriorityBands {
		band := &cfg.PriorityBands[i]
		if _, exists := priorities[band.Priority]; exists {
			return nil, fmt.Errorf("config validation failed: duplicate priority level %d found", band.Priority)
		}
		priorities[band.Priority] = struct{}{}

		if band.PriorityName == "" {
			return nil, fmt.Errorf("config validation failed: PriorityName is required for priority band %d", band.Priority)
		}
		if _, exists := priorityNames[band.PriorityName]; exists {
			return nil, fmt.Errorf("config validation failed: duplicate priority name %q found", band.PriorityName)
		}
		priorityNames[band.PriorityName] = struct{}{}

		if band.IntraFlowDispatchPolicy == "" {
			band.IntraFlowDispatchPolicy = defaultIntraFlowDispatchPolicy
		}
		if band.InterFlowDispatchPolicy == "" {
			band.InterFlowDispatchPolicy = defaultInterFlowDispatchPolicy
		}
		if band.Queue == "" {
			band.Queue = defaultQueue
		}
		if band.MaxBytes == 0 {
			band.MaxBytes = defaultPriorityBandMaxBytes
		}

		if err := cfg.validateBandCompatibility(*band); err != nil {
			return nil, err
		}
		cfg.priorityBandMap[band.Priority] = band
	}
	return cfg, nil
}

// validateBandCompatibility verifies that a band's configured queue type has the necessary capabilities.
func (c *Config) validateBandCompatibility(band PriorityBandConfig) error {
	policy, err := c.intraFlowDispatchPolicyFactory(band.IntraFlowDispatchPolicy)
	if err != nil {
		return fmt.Errorf("failed to validate policy %q for priority band %d: %w",
			band.IntraFlowDispatchPolicy, band.Priority, err)
	}

	requiredCapabilities := policy.RequiredQueueCapabilities()
	if len(requiredCapabilities) == 0 {
		return nil // The policy is compatible with any queue type.
	}

	// Create a temporary queue instance to inspect its capabilities.
	tempQueue, err := c.queueFactory(band.Queue, nil)
	if err != nil {
		return fmt.Errorf("failed to instantiate queue type %q for capability inspection (priority band %d): %w",
			band.Queue, band.Priority, err)
	}
	queueCapabilities := tempQueue.Capabilities()

	capabilitySet := make(map[framework.QueueCapability]struct{}, len(queueCapabilities))
	for _, cap := range queueCapabilities {
		capabilitySet[cap] = struct{}{}
	}

	for _, req := range requiredCapabilities {
		if _, ok := capabilitySet[req]; !ok {
			return fmt.Errorf(
				"policy %q is not compatible with queue %q for priority band %d (%s): missing capability %q: %w",
				policy.Name(),
				tempQueue.Name(),
				band.Priority,
				band.PriorityName,
				req,
				contracts.ErrPolicyQueueIncompatible,
			)
		}
	}
	return nil
}

// --- Partitioning Logic ---

// partition derives a `ShardConfig` from the master `Config` for a specific shard index.
// It calculates the capacity distribution, ensuring that the total global capacity is distributed completely and
// evenly.
func (c *Config) partition(shardIndex, totalShards int) *ShardConfig {
	shardCfg := &ShardConfig{
		MaxBytes:        partitionUint64(c.MaxBytes, shardIndex, totalShards),
		PriorityBands:   make([]ShardPriorityBandConfig, len(c.PriorityBands)),
		priorityBandMap: make(map[int]*ShardPriorityBandConfig, len(c.PriorityBands)),
	}

	for i, template := range c.PriorityBands {
		shardBandCfg := ShardPriorityBandConfig{
			Priority:                template.Priority,
			PriorityName:            template.PriorityName,
			IntraFlowDispatchPolicy: template.IntraFlowDispatchPolicy,
			InterFlowDispatchPolicy: template.InterFlowDispatchPolicy,
			Queue:                   template.Queue,
			MaxBytes:                partitionUint64(template.MaxBytes, shardIndex, totalShards),
		}
		shardCfg.PriorityBands[i] = shardBandCfg

		// Crucial: We must take the address of the element within the new slice (`shardCfg.PriorityBands`) to ensure the
		// map pointers are correct for the newly created ShardConfig instance.
		shardCfg.priorityBandMap[shardBandCfg.Priority] = &shardCfg.PriorityBands[i]
	}
	return shardCfg
}

// partitionUint64 distributes a total uint64 value across a number of partitions.
// It distributes the remainder of the division one by one to the first few partitions.
func partitionUint64(total uint64, partitionIndex, totalPartitions int) uint64 {
	if totalPartitions <= 0 {
		panic(fmt.Sprintf("invariant violation: cannot partition into %d partitions", totalPartitions))
	}
	if total == 0 {
		return 0
	}
	base := total / uint64(totalPartitions)
	remainder := total % uint64(totalPartitions)
	// Distribute the remainder. The first `remainder` partitions get one extra from the total.
	// For example, if total=10 and partitions=3, base=3, remainder=1. Partition 0 gets 3+1=4, partitions 1 and 2 get 3.
	if uint64(partitionIndex) < remainder {
		base++
	}
	return base
}

// --- Test-Only Dependency Injection ---

// configOption is a function that applies a configuration change to a `Config` object.
// This is used exclusively for dependency injection in tests and should not be used in production code.
type configOption func(*Config)

// interFlowDispatchPolicyFactory defines the signature for a function that creates an
// `framework.InterFlowDispatchPolicy` instance from its registered name.
// It serves as an abstraction over the concrete `interflow.NewPolicyFromName` factory, enabling dependency injection for
// testing validation logic.
type interFlowDispatchPolicyFactory func(name interflow.RegisteredPolicyName) (framework.InterFlowDispatchPolicy, error)

// intraFlowDispatchPolicyFactory defines the signature for a function that creates an
// `framework.IntraFlowDispatchPolicy` instance from its registered name.
// It serves as an abstraction over the concrete `intra.NewPolicyFromName` factory, enabling dependency injection for
// testing validation logic.
type intraFlowDispatchPolicyFactory func(name intra.RegisteredPolicyName) (framework.IntraFlowDispatchPolicy, error)

// queueFactory defines the signature for a function that creates a `framework.SafeQueue` instance from its registered
// name and a given `framework.ItemComparator`.
// It serves as an abstraction over the concrete `queue.NewQueueFromName` factory, enabling dependency injection for
// testing validation logic.
type queueFactory func(
	name queue.RegisteredQueueName,
	comparator framework.ItemComparator,
) (framework.SafeQueue, error)

// withInterFlowDispatchPolicyFactory returns a test-only `configOption` that overrides the factory function for
// creating `framework.InterFlowDispatchPolicy` instances.
// This is used exclusively for testing validation logic.
// test-only
func withInterFlowDispatchPolicyFactory(factory interFlowDispatchPolicyFactory) configOption {
	return func(c *Config) {
		c.interFlowDispatchPolicyFactory = factory
	}
}

// withIntraFlowDispatchPolicyFactory returns a test-only `configOption` that overrides the factory function for
// creating `framework.IntraFlowDispatchPolicy` instances.
// This is used exclusively for testing validation logic.
// test-only
func withIntraFlowDispatchPolicyFactory(factory intraFlowDispatchPolicyFactory) configOption {
	return func(c *Config) {
		c.intraFlowDispatchPolicyFactory = factory
	}
}

// withQueueFactory returns a test-only `configOption` that overrides the factory function for creating
// `framework.SafeQueue` instances.
// This is used exclusively for testing validation logic.
// test-only
func withQueueFactory(factory queueFactory) configOption {
	return func(c *Config) {
		c.queueFactory = factory
	}
}

// newConfig creates a new validated and defaulted `Config` object.
// It applies provided test-only functional options before validation and defaulting.
// It does not mutate the input `cfg`.
// test-only
func newConfig(cfg Config, opts ...configOption) (*Config, error) {
	newCfg := cfg.deepCopy()
	for _, opt := range opts {
		opt(newCfg)
	}
	return newCfg.ValidateAndApplyDefaults()
}

// --- Internal Utilities ---

// deepCopy creates a deep copy of the `Config` object.
func (c *Config) deepCopy() *Config {
	if c == nil {
		return nil
	}
	newCfg := &Config{
		MaxBytes:                       c.MaxBytes,
		InitialShardCount:              c.InitialShardCount,
		FlowGCTimeout:                  c.FlowGCTimeout,
		EventChannelBufferSize:         c.EventChannelBufferSize,
		PriorityBands:                  make([]PriorityBandConfig, len(c.PriorityBands)),
		interFlowDispatchPolicyFactory: c.interFlowDispatchPolicyFactory,
		intraFlowDispatchPolicyFactory: c.intraFlowDispatchPolicyFactory,
		queueFactory:                   c.queueFactory,
	}

	// PriorityBandConfig contains only value types, so a slice copy is sufficient for a deep copy.
	copy(newCfg.PriorityBands, c.PriorityBands)

	if c.priorityBandMap != nil {
		newCfg.priorityBandMap = make(map[int]*PriorityBandConfig, len(c.PriorityBands))
		// Crucial: We must rebuild the map and take the address of the elements within the new slice (`newCfg.PriorityBands`)
		// to ensure the map pointers are correct for the newly created `Config` instance.
		for i := range newCfg.PriorityBands {
			band := &newCfg.PriorityBands[i]
			newCfg.priorityBandMap[band.Priority] = band
		}
	}
	return newCfg
}

// getBandConfig finds and returns the global configuration template for a specific priority level.
// Returns an error wrapping `contracts.ErrPriorityBandNotFound` if the priority is not configured.
func (c *Config) getBandConfig(priority int) (*PriorityBandConfig, error) {
	if band, ok := c.priorityBandMap[priority]; ok {
		return band, nil
	}
	return nil, fmt.Errorf("global config for priority %d not found: %w", priority, contracts.ErrPriorityBandNotFound)
}
