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

package mocks

import (
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
)

// MockItemComparator is a simple stub mock for the `framework.ItemComparator` interface.
type MockItemComparator struct {
	FuncV      framework.ItemComparatorFunc
	ScoreTypeV string
}

func (m *MockItemComparator) Func() framework.ItemComparatorFunc { return m.FuncV }
func (m *MockItemComparator) ScoreType() string                  { return m.ScoreTypeV }

// MockFlowQueueAccessor is a simple stub mock for the `framework.FlowQueueAccessor` interface.
// It is used for tests that require static, predictable return values from a queue accessor.
// For complex, stateful queue behavior, use the mock in `contracts/mocks.MockManagedQueue`.
type MockFlowQueueAccessor struct {
	NameV         string
	LenV          int
	ByteSizeV     uint64
	PeekHeadV     types.QueueItemAccessor
	PeekTailV     types.QueueItemAccessor
	FlowKeyV      types.FlowKey
	ComparatorV   framework.ItemComparator
	CapabilitiesV []framework.QueueCapability
}

func (m *MockFlowQueueAccessor) Name() string                              { return m.NameV }
func (m *MockFlowQueueAccessor) Len() int                                  { return m.LenV }
func (m *MockFlowQueueAccessor) ByteSize() uint64                          { return m.ByteSizeV }
func (m *MockFlowQueueAccessor) Comparator() framework.ItemComparator      { return m.ComparatorV }
func (m *MockFlowQueueAccessor) FlowKey() types.FlowKey                    { return m.FlowKeyV }
func (m *MockFlowQueueAccessor) Capabilities() []framework.QueueCapability { return m.CapabilitiesV }

func (m *MockFlowQueueAccessor) PeekHead() types.QueueItemAccessor {
	return m.PeekHeadV
}

func (m *MockFlowQueueAccessor) PeekTail() types.QueueItemAccessor {
	return m.PeekTailV
}

var _ framework.FlowQueueAccessor = &MockFlowQueueAccessor{}

// MockPriorityBandAccessor is a behavioral mock for the `framework.MockPriorityBandAccessor` interface.
// Simple accessors are configured with public value fields (e.g., `PriorityV`).
// Complex methods with logic are configured with function fields (e.g., `IterateQueuesFunc`).
type MockPriorityBandAccessor struct {
	PriorityV         int
	PriorityNameV     string
	FlowKeysFunc      func() []types.FlowKey
	QueueFunc         func(flowID string) framework.FlowQueueAccessor
	IterateQueuesFunc func(callback func(queue framework.FlowQueueAccessor) (keepIterating bool))
}

func (m *MockPriorityBandAccessor) Priority() int        { return m.PriorityV }
func (m *MockPriorityBandAccessor) PriorityName() string { return m.PriorityNameV }

func (m *MockPriorityBandAccessor) FlowKeys() []types.FlowKey {
	if m.FlowKeysFunc != nil {
		return m.FlowKeysFunc()
	}
	return nil
}

func (m *MockPriorityBandAccessor) Queue(id string) framework.FlowQueueAccessor {
	if m.QueueFunc != nil {
		return m.QueueFunc(id)
	}
	return nil
}

func (m *MockPriorityBandAccessor) IterateQueues(callback func(queue framework.FlowQueueAccessor) bool) {
	if m.IterateQueuesFunc != nil {
		m.IterateQueuesFunc(callback)
	}
}

var _ framework.PriorityBandAccessor = &MockPriorityBandAccessor{}

// MockSafeQueue is a simple stub mock for the `framework.SafeQueue` interface.
// It is used for tests that need to control the exact return values of a queue's methods without simulating the queue's
// internal logic or state.
type MockSafeQueue struct {
	NameV         string
	CapabilitiesV []framework.QueueCapability
	LenV          int
	ByteSizeV     uint64
	PeekHeadV     types.QueueItemAccessor
	PeekTailV     types.QueueItemAccessor
	AddFunc       func(item types.QueueItemAccessor)
	RemoveFunc    func(handle types.QueueItemHandle) (types.QueueItemAccessor, error)
	CleanupFunc   func(predicate framework.PredicateFunc) []types.QueueItemAccessor
	DrainFunc     func() []types.QueueItemAccessor
}

func (m *MockSafeQueue) Name() string                              { return m.NameV }
func (m *MockSafeQueue) Capabilities() []framework.QueueCapability { return m.CapabilitiesV }
func (m *MockSafeQueue) Len() int                                  { return m.LenV }
func (m *MockSafeQueue) ByteSize() uint64                          { return m.ByteSizeV }

func (m *MockSafeQueue) PeekHead() types.QueueItemAccessor {
	return m.PeekHeadV
}

func (m *MockSafeQueue) PeekTail() types.QueueItemAccessor {
	return m.PeekTailV
}

func (m *MockSafeQueue) Add(item types.QueueItemAccessor) {
	if m.AddFunc != nil {
		m.AddFunc(item)
	}
}

func (m *MockSafeQueue) Remove(handle types.QueueItemHandle) (types.QueueItemAccessor, error) {
	if m.RemoveFunc != nil {
		return m.RemoveFunc(handle)
	}
	return nil, nil
}

func (m *MockSafeQueue) Cleanup(predicate framework.PredicateFunc) []types.QueueItemAccessor {
	if m.CleanupFunc != nil {
		return m.CleanupFunc(predicate)
	}
	return nil
}

func (m *MockSafeQueue) Drain() []types.QueueItemAccessor {
	if m.DrainFunc != nil {
		return m.DrainFunc()
	}
	return nil
}

var _ framework.SafeQueue = &MockSafeQueue{}

// MockIntraFlowDispatchPolicy is a behavioral mock for the `framework.IntraFlowDispatchPolicy` interface.
// Simple accessors are configured with public value fields (e.g., `NameV`).
// Complex methods with logic are configured with function fields (e.g., `SelectItemFunc`).
type MockIntraFlowDispatchPolicy struct {
	NameV                      string
	ComparatorV                framework.ItemComparator
	RequiredQueueCapabilitiesV []framework.QueueCapability
	SelectItemFunc             func(queue framework.FlowQueueAccessor) (types.QueueItemAccessor, error)
}

func (m *MockIntraFlowDispatchPolicy) Name() string                         { return m.NameV }
func (m *MockIntraFlowDispatchPolicy) Comparator() framework.ItemComparator { return m.ComparatorV }
func (m *MockIntraFlowDispatchPolicy) RequiredQueueCapabilities() []framework.QueueCapability {
	return m.RequiredQueueCapabilitiesV
}

func (m *MockIntraFlowDispatchPolicy) SelectItem(queue framework.FlowQueueAccessor) (types.QueueItemAccessor, error) {
	if m.SelectItemFunc != nil {
		return m.SelectItemFunc(queue)
	}
	return nil, nil
}

var _ framework.IntraFlowDispatchPolicy = &MockIntraFlowDispatchPolicy{}

// MockInterFlowDispatchPolicy is a behavioral mock for the `framework.InterFlowDispatchPolicy` interface.
// Simple accessors are configured with public value fields (e.g., `NameV`).
// Complex methods with logic are configured with function fields (e.g., `SelectQueueFunc`).
type MockInterFlowDispatchPolicy struct {
	NameV           string
	SelectQueueFunc func(band framework.PriorityBandAccessor) (framework.FlowQueueAccessor, error)
}

func (m *MockInterFlowDispatchPolicy) Name() string {
	return m.NameV
}

func (m *MockInterFlowDispatchPolicy) SelectQueue(band framework.PriorityBandAccessor) (framework.FlowQueueAccessor, error) {
	if m.SelectQueueFunc != nil {
		return m.SelectQueueFunc(band)
	}
	return nil, nil
}

var _ framework.InterFlowDispatchPolicy = &MockInterFlowDispatchPolicy{}
