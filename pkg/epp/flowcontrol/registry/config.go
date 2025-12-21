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

	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/interflow"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/intraflow"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/queue"
)

// --- Defaults ---

const (
	// defaultPriorityBandMaxBytes is the default global capacity for a priority band if not explicitly configured.
	// It is set to 1 GB.
	defaultPriorityBandMaxBytes uint64 = 1_000_000_000
	// defaultIntraFlowDispatchPolicy is the default policy for selecting items within a single flow's queue.
	defaultIntraFlowDispatchPolicy intraflow.RegisteredPolicyName = intraflow.FCFSPolicyName
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

// --- Capability Checking ---

// capabilityChecker abstracts the logic required to validate if a policy is compatible with a queue.
type capabilityChecker interface {
	CheckCompatibility(p intraflow.RegisteredPolicyName, q queue.RegisteredQueueName) error
}

// runtimeCapabilityChecker is the default implementation used in production.
// It instantiates the actual plugins to inspect their required and provided capabilities.
type runtimeCapabilityChecker struct{}

func (r *runtimeCapabilityChecker) CheckCompatibility(p intraflow.RegisteredPolicyName, q queue.RegisteredQueueName) error {
	tempPolicy, err := intraflow.NewPolicyFromName(p)
	if err != nil {
		return fmt.Errorf("failed to validate policy %q: %w", p, err)
	}

	requiredCapabilities := tempPolicy.RequiredQueueCapabilities()

	// We pass nil for the comparator as we only need to inspect static capabilities here.
	tempQueue, err := queue.NewQueueFromName(q, nil)
	if err != nil {
		return fmt.Errorf("failed to instantiate queue type %q: %w", q, err)
	}

	if len(requiredCapabilities) == 0 {
		return nil // The policy is compatible with any queue type.
	}

	queueCapabilities := tempQueue.Capabilities()
	capabilitySet := make(map[framework.QueueCapability]struct{}, len(queueCapabilities))
	for _, cap := range queueCapabilities {
		capabilitySet[cap] = struct{}{}
	}

	for _, req := range requiredCapabilities {
		if _, ok := capabilitySet[req]; !ok {
			return fmt.Errorf(
				"policy %q is not compatible with queue %q: missing capability %q: %w",
				tempPolicy.Name(),
				tempQueue.Name(),
				req,
				contracts.ErrPolicyQueueIncompatible,
			)
		}
	}
	return nil
}

// --- Configuration ---

// Config holds the master configuration for the entire FlowRegistry.
// It serves as the top-level blueprint, defining global settings and the templates for its priority bands.
type Config struct {
	// MaxBytes defines an optional, global maximum total byte size limit aggregated across all priority bands and shards.
	// The `controller.FlowController` enforces this limit in addition to per-band capacity limits.
	// A value of 0 signifies that this global limit is ignored, and only per-band limits apply.
	// Optional: Defaults to 0.
	MaxBytes uint64

	// PriorityBands defines the set of priority band templates managed by the `FlowRegistry`.
	// It is a map keyed by Priority level, providing O(1) access and ensuring priority uniqueness by definition.
	PriorityBands map[int]*PriorityBandConfig

	// DefaultPriorityBand serves as a template for dynamically provisioning priority bands when a request arrives with a
	// priority level that was not explicitly configured.
	// If nil, it is automatically populated with system defaults during NewConfig.
	DefaultPriorityBand *PriorityBandConfig

	// InitialShardCount specifies the number of parallel shards to create when the registry is initialized.
	// This value must be greater than zero.
	// Optional: Defaults to `defaultInitialShardCount` (1).
	InitialShardCount int

	// FlowGCTimeout defines the interval at which the registry scans for and garbage collects idle flows.
	// A flow is collected if it has been observed to be Idle for at least one full scan interval.
	// Optional: Defaults to `defaultFlowGCTimeout` (5 minutes).
	FlowGCTimeout time.Duration

	// EventChannelBufferSize defines the size of the buffered channel used for internal control plane events.
	// A larger buffer can absorb larger bursts of events (e.g., from many queues becoming non-empty simultaneously)
	// without blocking the data path, but consumes more memory.
	// This value must be greater than zero.
	// Optional: Defaults to `defaultEventChannelBufferSize` (4096).
	EventChannelBufferSize int
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
	// Optional: Defaults to defaultIntraFlowDispatchPolicy ("FCFS").
	IntraFlowDispatchPolicy intraflow.RegisteredPolicyName

	// InterFlowDispatchPolicy specifies the name of the policy used to select which flow's queue to service next from
	// this band.
	// Optional: Defaults to defaultInterFlowDispatchPolicy ("BestHead").
	InterFlowDispatchPolicy interflow.RegisteredPolicyName

	// Queue specifies the default name of the `framework.SafeQueue` implementation for flow queues in this band.
	// Optional: Defaults to defaultQueue ("ListQueue").
	Queue queue.RegisteredQueueName

	// MaxBytes defines the maximum total byte size for this priority band, aggregated across all shards.
	// Optional: Defaults to defaultPriorityBandMaxBytes (1 GB).
	MaxBytes uint64
}

// --- Config Functional Options ---

// configBuilder holds the intermediate state during NewConfig.
// It allows us to manage build-time dependencies (like capabilityChecker) without polluting the final Config struct.
type configBuilder struct {
	config  *Config
	checker capabilityChecker
}

// ConfigOption defines a functional option for configuring the registry.
type ConfigOption func(*configBuilder) error

// WithMaxBytes sets the global maximum total byte size limit.
func WithMaxBytes(maxBytes uint64) ConfigOption {
	return func(b *configBuilder) error {
		b.config.MaxBytes = maxBytes
		return nil
	}
}

// WithInitialShardCount sets the number of shards to create on startup.
func WithInitialShardCount(count int) ConfigOption {
	return func(b *configBuilder) error {
		if count <= 0 {
			return errors.New("initialShardCount must be greater than 0")
		}
		b.config.InitialShardCount = count
		return nil
	}
}

// WithFlowGCTimeout sets the idle flow garbage collection interval.
func WithFlowGCTimeout(d time.Duration) ConfigOption {
	return func(b *configBuilder) error {
		if d <= 0 {
			return errors.New("flowGCTimeout must be positive")
		}
		b.config.FlowGCTimeout = d
		return nil
	}
}

// WithPriorityBand adds a priority band configuration.
// If a band with the same Priority already exists, it returns an error.
func WithPriorityBand(band *PriorityBandConfig) ConfigOption {
	return func(b *configBuilder) error {
		if band == nil {
			return errors.New("cannot add nil PriorityBandConfig")
		}
		if _, exists := b.config.PriorityBands[band.Priority]; exists {
			return fmt.Errorf("duplicate priority level %d", band.Priority)
		}
		b.config.PriorityBands[band.Priority] = band
		return nil
	}
}

// WithDefaultPriorityBand sets the template configuration used for dynamically provisioning priority bands.
func WithDefaultPriorityBand(band *PriorityBandConfig) ConfigOption {
	return func(b *configBuilder) error {
		b.config.DefaultPriorityBand = band
		return nil
	}
}

// withCapabilityChecker overrides the compatibility checker used during validation.
// It is intended for use only in internal unit tests.
// test-only
func withCapabilityChecker(checker capabilityChecker) ConfigOption {
	return func(b *configBuilder) error {
		if checker == nil {
			return errors.New("cannot set nil CapabilityChecker")
		}
		b.checker = checker
		return nil
	}
}

// --- PriorityBandConfig Functional Options ---

// PriorityBandConfigOption defines a functional option for configuring a single PriorityBandConfig.
type PriorityBandConfigOption func(*PriorityBandConfig) error

// WithIntraFlowPolicy sets the intraflow-flow dispatch policy (e.g., "FCFS").
func WithIntraFlowPolicy(name intraflow.RegisteredPolicyName) PriorityBandConfigOption {
	return func(p *PriorityBandConfig) error {
		if name == "" {
			return errors.New("IntraFlowDispatchPolicy cannot be empty")
		}
		p.IntraFlowDispatchPolicy = name
		return nil
	}
}

// WithInterFlowPolicy sets the inter-flow dispatch policy (e.g., "RoundRobin").
func WithInterFlowPolicy(name interflow.RegisteredPolicyName) PriorityBandConfigOption {
	return func(p *PriorityBandConfig) error {
		if name == "" {
			return errors.New("InterFlowDispatchPolicy cannot be empty")
		}
		p.InterFlowDispatchPolicy = name
		return nil
	}
}

// WithQueue sets the queue implementation (e.g., "ListQueue") for flows in this band.
func WithQueue(name queue.RegisteredQueueName) PriorityBandConfigOption {
	return func(p *PriorityBandConfig) error {
		if name == "" {
			return errors.New("Queue cannot be empty")
		}
		p.Queue = name
		return nil
	}
}

// WithBandMaxBytes sets the capacity limit for this specific priority band.
func WithBandMaxBytes(maxBytes uint64) PriorityBandConfigOption {
	return func(p *PriorityBandConfig) error {
		p.MaxBytes = maxBytes
		return nil
	}
}

// --- Constructors ---

// NewConfig creates a new Config populated with system defaults, applies the provided options, and enforces strict
// validation.
func NewConfig(opts ...ConfigOption) (*Config, error) {
	builder := &configBuilder{
		config: &Config{
			MaxBytes:               0, // no limit enforced
			InitialShardCount:      defaultInitialShardCount,
			FlowGCTimeout:          defaultFlowGCTimeout,
			EventChannelBufferSize: defaultEventChannelBufferSize,
			PriorityBands:          make(map[int]*PriorityBandConfig),
		},
		checker: &runtimeCapabilityChecker{},
	}

	for _, opt := range opts {
		if err := opt(builder); err != nil {
			return nil, err
		}
	}

	// Initialize DefaultPriorityBand if missing.
	// This ensures we always have a template for dynamic provisioning.
	if builder.config.DefaultPriorityBand == nil {
		builder.config.DefaultPriorityBand = &PriorityBandConfig{}
	}

	// Apply defaults to the template.
	builder.config.DefaultPriorityBand.applyDefaults()
	if builder.config.DefaultPriorityBand.PriorityName == "" {
		builder.config.DefaultPriorityBand.PriorityName = "Dynamic-Default"
	}

	// Apply defaults to all explicitly configured bands.
	for _, band := range builder.config.PriorityBands {
		band.applyDefaults()
	}

	if err := builder.config.validate(builder.checker); err != nil {
		return nil, fmt.Errorf("invalid registry config: %w", err)
	}
	return builder.config, nil
}

// NewPriorityBandConfig creates a new band configuration with the required fields.
// It applies system defaults first, then applies any provided options to override those defaults.
func NewPriorityBandConfig(priority int, name string, opts ...PriorityBandConfigOption) (*PriorityBandConfig, error) {
	pb := &PriorityBandConfig{
		Priority:     priority,
		PriorityName: name,
	}

	pb.applyDefaults()

	for _, opt := range opts {
		if err := opt(pb); err != nil {
			return nil, err
		}
	}

	return pb, nil
}

// --- Validation & Defaults ---

func (p *PriorityBandConfig) applyDefaults() {
	if p.IntraFlowDispatchPolicy == "" {
		p.IntraFlowDispatchPolicy = defaultIntraFlowDispatchPolicy
	}
	if p.InterFlowDispatchPolicy == "" {
		p.InterFlowDispatchPolicy = defaultInterFlowDispatchPolicy
	}
	if p.Queue == "" {
		p.Queue = defaultQueue
	}
	if p.MaxBytes == 0 {
		p.MaxBytes = defaultPriorityBandMaxBytes
	}
}

// validate checks the integrity of a single band's configuration.
func (p *PriorityBandConfig) validate(checker capabilityChecker) error {
	if p.PriorityName == "" {
		return fmt.Errorf("PriorityName is required for priority band %d", p.Priority)
	}
	if p.IntraFlowDispatchPolicy == "" {
		return fmt.Errorf("IntraFlowDispatchPolicy required for priority band %d", p.Priority)
	}
	if p.InterFlowDispatchPolicy == "" {
		return fmt.Errorf("InterFlowDispatchPolicy required for priority band %d", p.Priority)
	}
	if p.Queue == "" {
		return fmt.Errorf("Queue required for priority band %d", p.Priority)
	}
	if checker != nil {
		if err := checker.CheckCompatibility(p.IntraFlowDispatchPolicy, p.Queue); err != nil {
			return fmt.Errorf("priority band %d (%s) configuration error: %w",
				p.Priority, p.PriorityName, err)
		}
	}
	return nil
}

// validate checks global constraints and delegates band validation.
func (c *Config) validate(checker capabilityChecker) error {
	if c.InitialShardCount <= 0 {
		return errors.New("initialShardCount must be greater than 0")
	}
	if c.FlowGCTimeout <= 0 {
		return errors.New("flowGCTimeout must be positive")
	}
	if c.EventChannelBufferSize <= 0 {
		return errors.New("eventChannelBufferSize must be greater than 0")
	}

	// Validate the dynamic template.
	// We use a dummy priority since the template itself doesn't have a fixed priority.
	templateValidationCopy := *c.DefaultPriorityBand
	templateValidationCopy.Priority = 0
	if err := templateValidationCopy.validate(checker); err != nil {
		return fmt.Errorf("invalid DefaultPriorityBand configuration: %w", err)
	}

	// Validate statically configured bands.
	names := sets.New[string]()
	for _, band := range c.PriorityBands {
		if names.Has(band.PriorityName) {
			return fmt.Errorf("duplicate priority name %q found", band.PriorityName)
		}
		names.Insert(band.PriorityName)

		if err := band.validate(checker); err != nil {
			return err
		}
	}
	return nil
}

// --- Sharding & Partitioning ---

// ShardConfig holds the partitioned configuration for a single registryShard.
type ShardConfig struct {
	MaxBytes      uint64
	PriorityBands map[int]*PriorityBandConfig
}

// partition derives a `ShardConfig` from the master `Config` for a specific shard index.
// It calculates the capacity distribution, ensuring that the total global capacity is distributed completely and
// evenly.
func (c *Config) partition(shardIndex, totalShards int) *ShardConfig {
	shardCfg := &ShardConfig{
		MaxBytes:      partitionUint64(c.MaxBytes, shardIndex, totalShards),
		PriorityBands: make(map[int]*PriorityBandConfig, len(c.PriorityBands)),
	}

	for _, template := range c.PriorityBands {
		shardBand := &PriorityBandConfig{
			Priority:                template.Priority,
			PriorityName:            template.PriorityName,
			IntraFlowDispatchPolicy: template.IntraFlowDispatchPolicy,
			InterFlowDispatchPolicy: template.InterFlowDispatchPolicy,
			Queue:                   template.Queue,
			MaxBytes:                partitionUint64(template.MaxBytes, shardIndex, totalShards),
		}

		shardCfg.PriorityBands[shardBand.Priority] = shardBand
	}
	return shardCfg
}

// partitionUint64 distributes a total uint64 value across a number of partitions.
// It distributes the remainder of the division one by one to the first few partitions.
func partitionUint64(total uint64, partitionIndex, totalPartitions int) uint64 {
	if total == 0 {
		return 0
	}

	t := uint64(totalPartitions)
	base := total / t
	remainder := total % t

	if uint64(partitionIndex) < remainder {
		return base + 1
	}
	return base
}

// Clone creates a deep copy of the Config.
// It ensures the new Config has its own independent map and PriorityBandConfig instances.
func (c *Config) Clone() *Config {
	if c == nil {
		return nil
	}

	clone := *c

	if c.DefaultPriorityBand != nil {
		val := *c.DefaultPriorityBand
		clone.DefaultPriorityBand = &val
	}

	if c.PriorityBands != nil {
		clone.PriorityBands = make(map[int]*PriorityBandConfig, len(c.PriorityBands))
		for prio, band := range c.PriorityBands {
			// Dereference the pointer to copy the struct value, then take the address of the new value.
			// This ensures 'clone' points to a new memory address.
			b := *band
			clone.PriorityBands[prio] = &b
		}
	}
	return &clone
}
