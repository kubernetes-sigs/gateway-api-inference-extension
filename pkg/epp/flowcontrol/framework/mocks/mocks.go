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

// MockItemComparator provides a mock implementation of the `framework.ItemComparator` interface.
type MockItemComparator struct {
	FuncV      framework.ItemComparatorFunc
	ScoreTypeV string
}

func (m *MockItemComparator) Func() framework.ItemComparatorFunc { return m.FuncV }
func (m *MockItemComparator) ScoreType() string                  { return m.ScoreTypeV }

var _ framework.ItemComparator = &MockItemComparator{}

// MockFlowQueueAccessor is a mock implementation of the `framework.FlowQueueAccessor` interface.
type MockFlowQueueAccessor struct {
	NameV         string
	CapabilitiesV []framework.QueueCapability
	LenV          int
	ByteSizeV     uint64
	PeekHeadV     types.QueueItemAccessor
	PeekHeadErrV  error
	PeekTailV     types.QueueItemAccessor
	PeekTailErrV  error
	FlowSpecV     types.FlowSpecification
	ComparatorV   framework.ItemComparator
}

func (m *MockFlowQueueAccessor) Name() string                              { return m.NameV }
func (m *MockFlowQueueAccessor) Capabilities() []framework.QueueCapability { return m.CapabilitiesV }
func (m *MockFlowQueueAccessor) Len() int                                  { return m.LenV }
func (m *MockFlowQueueAccessor) ByteSize() uint64                          { return m.ByteSizeV }

func (m *MockFlowQueueAccessor) PeekHead() (types.QueueItemAccessor, error) {
	return m.PeekHeadV, m.PeekHeadErrV
}

func (m *MockFlowQueueAccessor) PeekTail() (types.QueueItemAccessor, error) {
	return m.PeekTailV, m.PeekTailErrV
}

func (m *MockFlowQueueAccessor) Comparator() framework.ItemComparator { return m.ComparatorV }
func (m *MockFlowQueueAccessor) FlowSpec() types.FlowSpecification    { return m.FlowSpecV }

var _ framework.FlowQueueAccessor = &MockFlowQueueAccessor{}

// MockPriorityBandAccessor is a mock implementation of the `framework.PriorityBandAccessor` interface.
type MockPriorityBandAccessor struct {
	PriorityV      uint
	PriorityNameV  string
	FlowIDsV       []string
	QueueV         framework.FlowQueueAccessor // Value to return for any Queue(flowID) call
	QueueFuncV     func(flowID string) framework.FlowQueueAccessor
	IterateQueuesV func(callback func(queue framework.FlowQueueAccessor) bool)
}

func (m *MockPriorityBandAccessor) Priority() uint       { return m.PriorityV }
func (m *MockPriorityBandAccessor) PriorityName() string { return m.PriorityNameV }
func (m *MockPriorityBandAccessor) FlowIDs() []string    { return m.FlowIDsV }

func (m *MockPriorityBandAccessor) Queue(flowID string) framework.FlowQueueAccessor {
	if m.QueueFuncV != nil {
		return m.QueueFuncV(flowID)
	}
	return m.QueueV
}

func (m *MockPriorityBandAccessor) IterateQueues(callback func(queue framework.FlowQueueAccessor) bool) {
	if m.IterateQueuesV != nil {
		m.IterateQueuesV(callback)
	} else {
		// Default behavior: iterate based on FlowIDsV and QueueV/QueueFuncV
		for _, id := range m.FlowIDsV {
			q := m.Queue(id)
			if q != nil { // Only call callback if queue exists
				if !callback(q) {
					return
				}
			}
		}
	}
}

var _ framework.PriorityBandAccessor = &MockPriorityBandAccessor{}

// MockSafeQueue is a mock implementation of the `framework.SafeQueue` interface.
type MockSafeQueue struct {
	NameV         string
	CapabilitiesV []framework.QueueCapability
	LenV          int
	ByteSizeV     uint64
	PeekHeadV     types.QueueItemAccessor
	PeekHeadErrV  error
	PeekTailV     types.QueueItemAccessor
	PeekTailErrV  error
	AddFunc       func(item types.QueueItemAccessor) error
	RemoveFunc    func(handle types.QueueItemHandle) (types.QueueItemAccessor, error)
	CleanupFunc   func(predicate framework.PredicateFunc) ([]types.QueueItemAccessor, error)
	DrainFunc     func() ([]types.QueueItemAccessor, error)
}

func (m *MockSafeQueue) Name() string                              { return m.NameV }
func (m *MockSafeQueue) Capabilities() []framework.QueueCapability { return m.CapabilitiesV }
func (m *MockSafeQueue) Len() int                                  { return m.LenV }
func (m *MockSafeQueue) ByteSize() uint64                          { return m.ByteSizeV }

func (m *MockSafeQueue) PeekHead() (types.QueueItemAccessor, error) {
	return m.PeekHeadV, m.PeekHeadErrV
}

func (m *MockSafeQueue) PeekTail() (types.QueueItemAccessor, error) {
	return m.PeekTailV, m.PeekTailErrV
}

func (m *MockSafeQueue) Add(item types.QueueItemAccessor) error {
	if m.AddFunc != nil {
		return m.AddFunc(item)
	}
	return nil
}

func (m *MockSafeQueue) Remove(handle types.QueueItemHandle) (types.QueueItemAccessor, error) {
	if m.RemoveFunc != nil {
		return m.RemoveFunc(handle)
	}
	return nil, nil
}

func (m *MockSafeQueue) Cleanup(predicate framework.PredicateFunc) ([]types.QueueItemAccessor, error) {
	if m.CleanupFunc != nil {
		return m.CleanupFunc(predicate)
	}
	return nil, nil
}

func (m *MockSafeQueue) Drain() ([]types.QueueItemAccessor, error) {
	if m.DrainFunc != nil {
		return m.DrainFunc()
	}
	return nil, nil
}

var _ framework.SafeQueue = &MockSafeQueue{}

// MockIntraFlowDispatchPolicy is a mock implementation of the `framework.IntraFlowDispatchPolicy` interface.
type MockIntraFlowDispatchPolicy struct {
	NameV                      string
	SelectItemV                types.QueueItemAccessor
	SelectItemErrV             error
	ComparatorV                framework.ItemComparator
	RequiredQueueCapabilitiesV []framework.QueueCapability
}

func (m *MockIntraFlowDispatchPolicy) Name() string                         { return m.NameV }
func (m *MockIntraFlowDispatchPolicy) Comparator() framework.ItemComparator { return m.ComparatorV }

func (m *MockIntraFlowDispatchPolicy) SelectItem(queue framework.FlowQueueAccessor) (types.QueueItemAccessor, error) {
	return m.SelectItemV, m.SelectItemErrV
}

func (m *MockIntraFlowDispatchPolicy) RequiredQueueCapabilities() []framework.QueueCapability {
	return m.RequiredQueueCapabilitiesV
}

var _ framework.IntraFlowDispatchPolicy = &MockIntraFlowDispatchPolicy{}
