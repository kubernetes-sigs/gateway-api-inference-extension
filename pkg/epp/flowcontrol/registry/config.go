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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	inter "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/interflow/dispatch"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/interflow/dispatch/besthead"
	intra "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/intraflow/dispatch"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/policies/intraflow/dispatch/fcfs"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue/listqueue"
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
}

// partition calculates and returns a new `Config` with capacity values partitioned for a specific shard.
// This method ensures that the total capacity is distributed as evenly as possible across all shards.
func (c *Config) partition(shardIndex, totalShards int) (*Config, error) {
	if totalShards <= 0 || shardIndex < 0 || shardIndex >= totalShards {
		return nil, fmt.Errorf("invalid shard partitioning arguments: shardIndex=%d, totalShards=%d",
			shardIndex, totalShards)
	}

	partitionValue := func(total uint64) uint64 {
		if total == 0 {
			return 0
		}
		base := total / uint64(totalShards)
		remainder := total % uint64(totalShards)
		if uint64(shardIndex) < remainder {
			return base + 1
		}
		return base
	}

	newCfg := &Config{
		MaxBytes:      partitionValue(c.MaxBytes),
		PriorityBands: make([]PriorityBandConfig, len(c.PriorityBands)),
	}

	for i, band := range c.PriorityBands {
		newBand := band                                  // Copy the original config
		newBand.MaxBytes = partitionValue(band.MaxBytes) // Overwrite with the partitioned value
		newCfg.PriorityBands[i] = newBand
	}

	return newCfg, nil
}

// validateAndApplyDefaults checks the configuration for validity and populates any empty fields with system defaults.
// This method should be called once by the registry before it initializes any shards.
func (c *Config) validateAndApplyDefaults() error {
	if len(c.PriorityBands) == 0 {
		return errors.New("config validation failed: at least one priority band must be defined")
	}

	priorities := make(map[uint]struct{}) // Keep track of seen priorities

	for i := range c.PriorityBands {
		band := &c.PriorityBands[i]
		if _, exists := priorities[band.Priority]; exists {
			return fmt.Errorf("config validation failed: duplicate priority level %d found", band.Priority)
		}
		priorities[band.Priority] = struct{}{}

		if band.PriorityName == "" {
			return errors.New("config validation failed: PriorityName is required for all priority bands")
		}
		if band.IntraFlowDispatchPolicy == "" {
			band.IntraFlowDispatchPolicy = fcfs.FCFSPolicyName
		}
		if band.InterFlowDispatchPolicy == "" {
			band.InterFlowDispatchPolicy = besthead.BestHeadPolicyName
		}
		if band.Queue == "" {
			band.Queue = listqueue.ListQueueName
		}

		// After defaulting, validate that the chosen plugins are compatible.
		if err := validateBandCompatibility(*band); err != nil {
			return err
		}
	}
	return nil
}

// validateBandCompatibility verifies that a band's default policy is compatible with its default queue type.
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
	// Optional: If empty, a system default (e.g., "FCFS") is used.
	IntraFlowDispatchPolicy intra.RegisteredPolicyName

	// InterFlowDispatchPolicy specifies the name of the registered policy used to select which flow's queue to service
	// next from this band.
	//
	// Optional: If empty, a system default (e.g., "BestHead") is used.
	InterFlowDispatchPolicy inter.RegisteredPolicyName

	// Queue specifies the default name of the registered SafeQueue implementation to be used for flow queues within this
	// band.
	//
	// Optional: If empty, a system default (e.g., "ListQueue") is used.
	Queue queue.RegisteredQueueName

	// MaxBytes defines the maximum total byte size for this specific priority band, aggregated across all shards.
	//
	// Optional: If not set, a system default (e.g., 1 GB) is applied.
	MaxBytes uint64
}
