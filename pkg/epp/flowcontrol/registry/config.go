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
	inter "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/interflow/dispatch"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/interflow/dispatch/besthead"
	intra "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/intraflow/dispatch"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/intraflow/dispatch/fcfs"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue/listqueue"
)

const (
	// defaultPriorityBandMaxBytes is the default capacity for a priority band if not explicitly configured.
	// It is set to 1 GB.
	defaultPriorityBandMaxBytes = 1_000_000_000
	// defaultIntraFlowDispatchPolicy is the default policy for selecting items within a single flow's queue.
	defaultIntraFlowDispatchPolicy = fcfs.FCFSPolicyName
	// defaultInterFlowDispatchPolicy is the default policy for selecting which flow's queue to service next.
	defaultInterFlowDispatchPolicy = besthead.BestHeadPolicyName
	// defaultQueue is the default queue implementation for flows.
	defaultQueue = listqueue.ListQueueName
	// defaultFlowGCTimeout is the default duration of inactivity after which an idle flow is garbage collected.
	defaultFlowGCTimeout = 5 * time.Minute
	// defaultEventChannelBufferSize is the default size of the buffered channel for control plane events.
	defaultEventChannelBufferSize = 4096
)

// Config holds the master configuration for the entire `FlowRegistry`. It serves as the top-level blueprint, defining
// global capacity limits and the structure of its priority bands.
//
// This master configuration is validated and defaulted once at startup. It is then partitioned and distributed to each
// internal `registryShard`, ensuring a consistent and predictable state across the system.
type Config struct {
	// MaxBytes defines an optional, global maximum total byte size limit aggregated across all priority bands and shards.
	// The `controller.FlowController` enforces this limit in addition to per-band capacity limits.
	//
	// Optional: Defaults to 0, which signifies that the global limit is ignored.
	MaxBytes uint64

	// PriorityBands defines the set of priority bands managed by the `FlowRegistry`. The configuration for each band,
	// including its default policies and queue types, is specified here.
	//
	// Required: At least one `PriorityBandConfig` must be provided for a functional registry.
	PriorityBands []PriorityBandConfig

	// FlowGCTimeout defines the duration of inactivity after which an idle flow is automatically unregistered and
	// garbage collected. A flow is considered inactive if its queues have been empty across all shards for this
	// duration.
	//
	// Optional: Defaults to 5 minutes.
	FlowGCTimeout time.Duration

	// EventChannelBufferSize defines the size of the buffered channel used for internal control plane events.
	// A larger buffer can absorb larger bursts of events (e.g., from many queues becoming non-empty simultaneously)
	// without blocking the data path, but consumes more memory.
	//
	// Optional: Defaults to 4096.
	EventChannelBufferSize uint32

	// priorityBandMap is a cache for O(1) lookups of PriorityBandConfig by priority level.
	// It is populated during validateAndApplyDefaults and when the config is partitioned or copied.
	priorityBandMap map[uint]*PriorityBandConfig
}

// partition calculates and returns a new `Config` with capacity values partitioned for a specific shard. This method
// ensures that the total capacity is distributed as evenly as possible across all shards by distributing the remainder
// of the division one by one to the first few shards.
func (c *Config) partition(shardIndex, totalShards int) (*Config, error) {
	if totalShards <= 0 || shardIndex < 0 || shardIndex >= totalShards {
		return nil, fmt.Errorf("invalid shard partitioning arguments: shardIndex=%d, totalShards=%d",
			shardIndex, totalShards)
	}

	newCfg := &Config{
		MaxBytes:               partitionUint64(c.MaxBytes, shardIndex, totalShards),
		FlowGCTimeout:          c.FlowGCTimeout,
		EventChannelBufferSize: c.EventChannelBufferSize,
		PriorityBands:          make([]PriorityBandConfig, len(c.PriorityBands)),
		priorityBandMap:        make(map[uint]*PriorityBandConfig, len(c.PriorityBands)),
	}

	for i, band := range c.PriorityBands {
		newBand := band                                                            // Copy the original config
		newBand.MaxBytes = partitionUint64(band.MaxBytes, shardIndex, totalShards) // Overwrite with the partitioned value
		newCfg.PriorityBands[i] = newBand
	}

	// Populate the map for the new partitioned config.
	for i := range newCfg.PriorityBands {
		band := &newCfg.PriorityBands[i]
		newCfg.priorityBandMap[band.Priority] = band
	}

	return newCfg, nil
}

// partitionUint64 distributes a total uint64 value across a number of partitions.
// It distributes the remainder of the division one by one to the first few partitions.
func partitionUint64(total uint64, partitionIndex, totalPartitions int) uint64 {
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

// validateAndApplyDefaults checks the configuration for validity and mutates it to populate any empty fields with
// system defaults. It ensures that all priority bands are well-formed, have unique priority levels and names, and that
// their chosen plugins are compatible. This method should be called once by the registry before it initializes any
// shards.
func (c *Config) validateAndApplyDefaults() error {
	if c.FlowGCTimeout <= 0 {
		c.FlowGCTimeout = defaultFlowGCTimeout
	}
	if c.EventChannelBufferSize <= 0 {
		c.EventChannelBufferSize = defaultEventChannelBufferSize
	}

	if len(c.PriorityBands) == 0 {
		return errors.New("config validation failed: at least one priority band must be defined")
	}

	priorities := make(map[uint]struct{})
	priorityNames := make(map[string]struct{})
	c.priorityBandMap = make(map[uint]*PriorityBandConfig, len(c.PriorityBands))

	for i := range c.PriorityBands {
		band := &c.PriorityBands[i]
		if _, exists := priorities[band.Priority]; exists {
			return fmt.Errorf("config validation failed: duplicate priority level %d found", band.Priority)
		}
		priorities[band.Priority] = struct{}{}

		if band.PriorityName == "" {
			return fmt.Errorf("config validation failed: PriorityName is required for priority band %d", band.Priority)
		}
		if _, exists := priorityNames[band.PriorityName]; exists {
			return fmt.Errorf("config validation failed: duplicate priority name %q found", band.PriorityName)
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

		// After defaulting, validate that the chosen plugins are compatible.
		if err := validateBandCompatibility(*band); err != nil {
			return err
		}
		// Populate the lookup map.
		c.priorityBandMap[band.Priority] = band
	}
	return nil
}

// validateBandCompatibility verifies that a band's configured queue type has the necessary capabilities to support its
// configured intra-flow dispatch policy. For example, a priority-based policy requires a queue that supports priority
// ordering.
func validateBandCompatibility(band PriorityBandConfig) error {
	policy, err := intra.NewPolicyFromName(band.IntraFlowDispatchPolicy)
	if err != nil {
		return fmt.Errorf("failed to validate policy %q for priority band %d: %w",
			band.IntraFlowDispatchPolicy, band.Priority, err)
	}

	requiredCapabilities := policy.RequiredQueueCapabilities()
	if len(requiredCapabilities) == 0 {
		return nil // Policy has no specific requirements.
	}

	// Create a temporary queue instance to inspect its capabilities.
	tempQueue, err := queue.NewQueueFromName(band.Queue, nil)
	if err != nil {
		return fmt.Errorf("failed to inspect queue type %q for priority band %d: %w", band.Queue, band.Priority, err)
	}
	queueCapabilities := tempQueue.Capabilities()

	// Build a set of the queue's capabilities for efficient lookup.
	capabilitySet := make(map[framework.QueueCapability]struct{}, len(queueCapabilities))
	for _, cap := range queueCapabilities {
		capabilitySet[cap] = struct{}{}
	}

	// Check if all required capabilities are present.
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

// PriorityBandConfig defines the configuration for a single priority band within the `FlowRegistry`. It establishes the
// default behaviors (such as queueing and dispatch policies) and capacity limits for all flows that operate at this
// priority level.
type PriorityBandConfig struct {
	// Priority is the numerical priority level for this band.
	// Convention: Lower numerical values indicate higher priority (e.g., 0 is highest).
	//
	// Required.
	Priority uint

	// PriorityName is a human-readable name for this priority band (e.g., "Critical", "Standard", "Sheddable").
	//
	// Required.
	PriorityName string

	// IntraFlowDispatchPolicy specifies the default name of the registered policy used to select a specific request to
	// dispatch next from within a single flow's queue in this band. This default can be overridden on a per-flow basis.
	//
	// Optional: If empty, a system default ("FCFS") is used.
	IntraFlowDispatchPolicy intra.RegisteredPolicyName

	// InterFlowDispatchPolicy specifies the name of the registered policy used to select which flow's queue to service
	// next from this band.
	//
	// Optional: If empty, a system default ("BestHead") is used.
	InterFlowDispatchPolicy inter.RegisteredPolicyName

	// Queue specifies the default name of the registered SafeQueue implementation to be used for flow queues within this
	// band.
	//
	// Optional: If empty, a system default ("ListQueue") is used.
	Queue queue.RegisteredQueueName

	// MaxBytes defines the maximum total byte size for this specific priority band, aggregated across all shards.
	//
	// Optional: If not set, a system default (e.g., 1 GB) is applied.
	MaxBytes uint64
}

// getBandConfig finds and returns the configuration for a specific priority level using the O(1) lookup map.
func (c *Config) getBandConfig(priority uint) (*PriorityBandConfig, error) {
	if band, ok := c.priorityBandMap[priority]; ok {
		return band, nil
	}
	return nil, fmt.Errorf("config for priority %d not found: %w", priority, contracts.ErrPriorityBandNotFound)
}

// deepCopy creates a deep copy of the Config object.
func (c *Config) deepCopy() *Config {
	if c == nil {
		return nil
	}
	newCfg := &Config{
		MaxBytes:               c.MaxBytes,
		FlowGCTimeout:          c.FlowGCTimeout,
		EventChannelBufferSize: c.EventChannelBufferSize,
		PriorityBands:          make([]PriorityBandConfig, len(c.PriorityBands)),
		priorityBandMap:        make(map[uint]*PriorityBandConfig, len(c.PriorityBands)),
	}
	// PriorityBandConfig is a struct of value types, so a direct copy of the struct
	// is sufficient for a deep copy. The `copy` built-in creates a new slice and
	// copies the struct values from the original slice into it.
	copy(newCfg.PriorityBands, c.PriorityBands)

	// Rebuild the map so pointers refer to the new slice elements.
	for i := range newCfg.PriorityBands {
		band := &newCfg.PriorityBands[i]
		newCfg.priorityBandMap[band.Priority] = band
	}
	return newCfg
}
