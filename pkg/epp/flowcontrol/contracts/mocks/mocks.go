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

// Package mocks provides mocks for the interfaces defined in the `contracts` package.
//
// # Testing Philosophy: High-Fidelity Mocks
//
// The components that consume these contracts, particularly the `controller.ShardProcessor`, are complex, concurrent
// orchestrators. Testing them reliably requires more than simple stubs. It requires high-fidelity mocks that allow for
// the deterministic simulation of race conditions and specific failure modes.
//
// For this reason, mocks like `MockManagedQueue` are deliberately stateful and thread-safe. They provide a reliable,
// in-memory simulation of the real component's behavior, while also providing function-based overrides
// (e.g., `AddFunc`) that allow tests to inject specific errors or pause execution at critical moments. This strategy is
// essential for creating the robust, non-flaky tests needed to verify the correctness of the system's concurrent logic.
// For a more detailed defense of this strategy, see the comment at the top of `controller/internal/processor_test.go`.
package mocks

import (
	"context"
	"fmt"
	"sync"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	typesmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types/mocks"
)

// --- RegistryShard Mocks ---

// MockRegistryShard is a simple "stub-style" mock for testing.
// Its methods are implemented as function fields (e.g., `IDFunc`). A test can inject behavior by setting the desired
// function field in the test setup. If a func is nil, the method will return a zero value.
type MockRegistryShard struct {
	IDFunc                       func() string
	IsActiveFunc                 func() bool
	ManagedQueueFunc             func(key types.FlowKey) (contracts.ManagedQueue, error)
	IntraFlowDispatchPolicyFunc  func(key types.FlowKey) (framework.IntraFlowDispatchPolicy, error)
	InterFlowDispatchPolicyFunc  func(priority int) (framework.InterFlowDispatchPolicy, error)
	PriorityBandAccessorFunc     func(priority int) (framework.PriorityBandAccessor, error)
	AllOrderedPriorityLevelsFunc func() []int
	StatsFunc                    func() contracts.ShardStats
}

func (m *MockRegistryShard) ID() string {
	if m.IDFunc != nil {
		return m.IDFunc()
	}
	return ""
}

func (m *MockRegistryShard) IsActive() bool {
	if m.IsActiveFunc != nil {
		return m.IsActiveFunc()
	}
	return false
}

func (m *MockRegistryShard) ManagedQueue(key types.FlowKey) (contracts.ManagedQueue, error) {
	if m.ManagedQueueFunc != nil {
		return m.ManagedQueueFunc(key)
	}
	return nil, nil
}

func (m *MockRegistryShard) IntraFlowDispatchPolicy(key types.FlowKey) (framework.IntraFlowDispatchPolicy, error) {
	if m.IntraFlowDispatchPolicyFunc != nil {
		return m.IntraFlowDispatchPolicyFunc(key)
	}
	return nil, nil
}

func (m *MockRegistryShard) InterFlowDispatchPolicy(priority int) (framework.InterFlowDispatchPolicy, error) {
	if m.InterFlowDispatchPolicyFunc != nil {
		return m.InterFlowDispatchPolicyFunc(priority)
	}
	return nil, nil
}

func (m *MockRegistryShard) PriorityBandAccessor(priority int) (framework.PriorityBandAccessor, error) {
	if m.PriorityBandAccessorFunc != nil {
		return m.PriorityBandAccessorFunc(priority)
	}
	return nil, nil
}

func (m *MockRegistryShard) AllOrderedPriorityLevels() []int {
	if m.AllOrderedPriorityLevelsFunc != nil {
		return m.AllOrderedPriorityLevelsFunc()
	}
	return nil
}

func (m *MockRegistryShard) Stats() contracts.ShardStats {
	if m.StatsFunc != nil {
		return m.StatsFunc()
	}
	return contracts.ShardStats{}
}

// --- Dependency Mocks ---

// MockSaturationDetector is a simple "stub-style" mock for testing.
type MockSaturationDetector struct {
	IsSaturatedFunc func(ctx context.Context, candidatePods []metrics.PodMetrics) bool
}

func (m *MockSaturationDetector) IsSaturated(ctx context.Context, candidatePods []metrics.PodMetrics) bool {
	if m.IsSaturatedFunc != nil {
		return m.IsSaturatedFunc(ctx, candidatePods)
	}
	return false
}

// MockPodLocator provides a mock implementation of the contracts.PodLocator interface.
// It allows tests to control the exact set of pods returned for a given request.
type MockPodLocator struct {
	// LocateFunc allows injecting custom logic.
	LocateFunc func(ctx context.Context, requestMetadata map[string]any) []metrics.PodMetrics
	// Pods is a static return value used if LocateFunc is nil.
	Pods []metrics.PodMetrics
}

func (m *MockPodLocator) Locate(ctx context.Context, requestMetadata map[string]any) []metrics.PodMetrics {
	if m.LocateFunc != nil {
		return m.LocateFunc(ctx, requestMetadata)
	}
	// Return copy to be safe
	if m.Pods == nil {
		return nil
	}
	result := make([]metrics.PodMetrics, len(m.Pods))
	copy(result, m.Pods)
	return result
}

// --- ManagedQueue Mock ---

// MockManagedQueue is a high-fidelity, thread-safe mock of the `contracts.ManagedQueue` interface, designed
// specifically for testing the concurrent `controller/internal.ShardProcessor`.
//
// This mock is essential for creating deterministic and focused unit tests. It allows for precise control over queue
// behavior and enables the testing of critical edge cases (e.g., empty queues, dispatch failures) in complete
// isolation, which would be difficult and unreliable to achieve with the concrete `registry.managedQueue`
// implementation.
//
// ### Design Philosophy
//
//  1. **Stateful**: The mock maintains an internal map of items to accurately reflect a real queue's state. Its `Len()`
//     and `ByteSize()` methods are derived directly from this state.
//  2. **Deadlock-Safe Overrides**: Test-specific logic (e.g., `AddFunc`) is executed instead of the default
//     implementation. The override function is fully responsible for its own logic and synchronization, as the mock's
//     internal mutex will *not* be held during its execution.
//  3. **Self-Wiring**: The `FlowQueueAccessor()` method returns the mock itself, ensuring the accessor is always
//     correctly connected to the queue's state without manual wiring in tests.
type MockManagedQueue struct {
	// FlowKeyV defines the flow specification for this mock queue. It should be set by the test.
	FlowKeyV types.FlowKey

	// AddFunc allows a test to completely override the default Add behavior.
	AddFunc func(item types.QueueItemAccessor) error
	// RemoveFunc allows a test to completely override the default Remove behavior.
	RemoveFunc func(handle types.QueueItemHandle) (types.QueueItemAccessor, error)
	// CleanupFunc allows a test to completely override the default Cleanup behavior.
	CleanupFunc func(predicate framework.PredicateFunc) []types.QueueItemAccessor
	// DrainFunc allows a test to completely override the default Drain behavior.
	DrainFunc func() []types.QueueItemAccessor

	// mu protects access to the internal `items` map.
	mu       sync.Mutex
	initOnce sync.Once
	items    map[types.QueueItemHandle]types.QueueItemAccessor
}

func (m *MockManagedQueue) init() {
	m.initOnce.Do(func() {
		m.items = make(map[types.QueueItemHandle]types.QueueItemAccessor)
	})
}

// FlowQueueAccessor returns the mock itself, as it fully implements the `framework.FlowQueueAccessor` interface.
func (m *MockManagedQueue) FlowQueueAccessor() framework.FlowQueueAccessor {
	return m
}

// Add adds an item to the queue.
// It checks for a test override before locking. If no override is present, it executes the default stateful logic,
// which includes fulfilling the `SafeQueue.Add` contract.
func (m *MockManagedQueue) Add(item types.QueueItemAccessor) error {
	// If an override is provided, it is responsible for the full contract, including setting the handle.
	if m.AddFunc != nil {
		return m.AddFunc(item)
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	m.init()

	// Fulfill the `SafeQueue.Add` contract: the queue is responsible for setting the handle.
	if item.Handle() == nil {
		item.SetHandle(&typesmocks.MockQueueItemHandle{})
	}

	m.items[item.Handle()] = item
	return nil
}

// Remove removes an item from the queue. It checks for a test override before locking.
func (m *MockManagedQueue) Remove(handle types.QueueItemHandle) (types.QueueItemAccessor, error) {
	if m.RemoveFunc != nil {
		return m.RemoveFunc(handle)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.init()
	item, ok := m.items[handle]
	if !ok {
		return nil, fmt.Errorf("item with handle %v not found", handle)
	}
	delete(m.items, handle)
	return item, nil
}

// Cleanup removes items matching a predicate. It checks for a test override before locking.
func (m *MockManagedQueue) Cleanup(predicate framework.PredicateFunc) []types.QueueItemAccessor {
	if m.CleanupFunc != nil {
		return m.CleanupFunc(predicate)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.init()
	var removed []types.QueueItemAccessor
	for handle, item := range m.items {
		if predicate(item) {
			removed = append(removed, item)
			delete(m.items, handle)
		}
	}
	return removed
}

// Drain removes all items from the queue. It checks for a test override before locking.
func (m *MockManagedQueue) Drain() []types.QueueItemAccessor {
	if m.DrainFunc != nil {
		return m.DrainFunc()
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.init()
	drained := make([]types.QueueItemAccessor, 0, len(m.items))
	for _, item := range m.items {
		drained = append(drained, item)
	}
	m.items = make(map[types.QueueItemHandle]types.QueueItemAccessor)
	return drained
}

func (m *MockManagedQueue) FlowKey() types.FlowKey                    { return m.FlowKeyV }
func (m *MockManagedQueue) Name() string                              { return "" }
func (m *MockManagedQueue) Capabilities() []framework.QueueCapability { return nil }
func (m *MockManagedQueue) Comparator() framework.ItemComparator      { return nil }

// Len returns the actual number of items currently in the mock queue.
func (m *MockManagedQueue) Len() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.init()
	return len(m.items)
}

// ByteSize returns the actual total byte size of all items in the mock queue.
func (m *MockManagedQueue) ByteSize() uint64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.init()
	var size uint64
	for _, item := range m.items {
		size += item.OriginalRequest().ByteSize()
	}
	return size
}

// PeekHead returns the first item found in the mock queue. Note: map iteration order is not guaranteed.
func (m *MockManagedQueue) PeekHead() types.QueueItemAccessor {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.init()
	for _, item := range m.items {
		return item // Return first item found
	}
	return nil // Queue is empty
}

// PeekTail is not implemented for this mock.
func (m *MockManagedQueue) PeekTail() types.QueueItemAccessor {
	return nil
}
