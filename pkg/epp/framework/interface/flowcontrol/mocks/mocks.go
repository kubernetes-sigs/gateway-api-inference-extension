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
	"context"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// MockFlowQueueAccessor is a simple stub mock for the FlowQueueAccessor interface.
// It is used for tests that require static, predictable return values from a queue accessor.
// For complex, stateful queue behavior, use the mock in ../../contracts/mocks.MockManagedQueue.
type MockFlowQueueAccessor struct {
	NameV           string
	LenV            int
	ByteSizeV       uint64
	PeekHeadV       types.QueueItemAccessor
	PeekTailV       types.QueueItemAccessor
	FlowKeyV        types.FlowKey
	OrderingPolicyV flowcontrol.OrderingPolicy
	CapabilitiesV   []flowcontrol.QueueCapability
}

func (m *MockFlowQueueAccessor) Name() string                                { return m.NameV }
func (m *MockFlowQueueAccessor) Len() int                                    { return m.LenV }
func (m *MockFlowQueueAccessor) ByteSize() uint64                            { return m.ByteSizeV }
func (m *MockFlowQueueAccessor) OrderingPolicy() flowcontrol.OrderingPolicy  { return m.OrderingPolicyV }
func (m *MockFlowQueueAccessor) FlowKey() types.FlowKey                      { return m.FlowKeyV }
func (m *MockFlowQueueAccessor) Capabilities() []flowcontrol.QueueCapability { return m.CapabilitiesV }

func (m *MockFlowQueueAccessor) PeekHead() types.QueueItemAccessor {
	return m.PeekHeadV
}

func (m *MockFlowQueueAccessor) PeekTail() types.QueueItemAccessor {
	return m.PeekTailV
}

var _ flowcontrol.FlowQueueAccessor = &MockFlowQueueAccessor{}

// MockPriorityBandAccessor is a behavioral mock for the PriorityBandAccessor interface.
// Simple accessors are configured with public value fields (e.g., PriorityV).
// Complex methods with logic are configured with function fields (e.g., IterateQueuesFunc).
//
// Convention: Fields suffixed with 'V' (e.g., PriorityV) are static Value return fields.
// This avoids collision with the interface method of the same name.
type MockPriorityBandAccessor struct {
	PriorityV         int
	PriorityNameV     string
	PolicyStateV      any
	FlowKeysFunc      func() []types.FlowKey
	QueueFunc         func(flowID string) flowcontrol.FlowQueueAccessor
	IterateQueuesFunc func(callback func(flow flowcontrol.FlowQueueAccessor) (keepIterating bool))
}

func (m *MockPriorityBandAccessor) Priority() int        { return m.PriorityV }
func (m *MockPriorityBandAccessor) PriorityName() string { return m.PriorityNameV }
func (m *MockPriorityBandAccessor) PolicyState() any     { return m.PolicyStateV }

func (m *MockPriorityBandAccessor) FlowKeys() []types.FlowKey {
	if m.FlowKeysFunc != nil {
		return m.FlowKeysFunc()
	}
	return nil
}

func (m *MockPriorityBandAccessor) Queue(id string) flowcontrol.FlowQueueAccessor {
	if m.QueueFunc != nil {
		return m.QueueFunc(id)
	}
	return nil
}

func (m *MockPriorityBandAccessor) IterateQueues(callback func(flow flowcontrol.FlowQueueAccessor) bool) {
	if m.IterateQueuesFunc != nil {
		m.IterateQueuesFunc(callback)
	}
}

var _ flowcontrol.PriorityBandAccessor = &MockPriorityBandAccessor{}

// MockSafeQueue is a simple stub mock for the SafeQueue interface.
// It is used for tests that need to control the exact return values of a queue's methods without simulating the queue's
// internal logic or state.
type MockSafeQueue struct {
	NameV         string
	CapabilitiesV []flowcontrol.QueueCapability
	LenV          int
	ByteSizeV     uint64
	PeekHeadV     types.QueueItemAccessor
	PeekTailV     types.QueueItemAccessor
	AddFunc       func(item types.QueueItemAccessor)
	RemoveFunc    func(handle types.QueueItemHandle) (types.QueueItemAccessor, error)
	CleanupFunc   func(predicate flowcontrol.PredicateFunc) []types.QueueItemAccessor
	DrainFunc     func() []types.QueueItemAccessor
}

func (m *MockSafeQueue) Name() string                                { return m.NameV }
func (m *MockSafeQueue) Capabilities() []flowcontrol.QueueCapability { return m.CapabilitiesV }
func (m *MockSafeQueue) Len() int                                    { return m.LenV }
func (m *MockSafeQueue) ByteSize() uint64                            { return m.ByteSizeV }

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

func (m *MockSafeQueue) Cleanup(predicate flowcontrol.PredicateFunc) []types.QueueItemAccessor {
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

var _ flowcontrol.SafeQueue = &MockSafeQueue{}

// MockOrderingPolicy is a behavioral mock for the OrderingPolicy interface.
// Simple accessors are configured with public value fields (e.g., TypedNameV).
// Complex methods with logic are configured with function fields (e.g., LessFunc).
type MockOrderingPolicy struct {
	TypedNameV                 plugin.TypedName
	LessFunc                   func(a, b types.QueueItemAccessor) bool
	RequiredQueueCapabilitiesV []flowcontrol.QueueCapability
}

func (m *MockOrderingPolicy) TypedName() plugin.TypedName { return m.TypedNameV }

func (m *MockOrderingPolicy) Less(a, b types.QueueItemAccessor) bool {
	if m.LessFunc != nil {
		return m.LessFunc(a, b)
	}
	return false
}

func (m *MockOrderingPolicy) RequiredQueueCapabilities() []flowcontrol.QueueCapability {
	return m.RequiredQueueCapabilitiesV
}

var _ flowcontrol.OrderingPolicy = &MockOrderingPolicy{}

// MockFairnessPolicy is a behavioral mock for the FairnessPolicy interface.
// Simple accessors are configured with public value fields (e.g., NameV).
// Complex methods with logic are configured with function fields (e.g., PickFunc).
type MockFairnessPolicy struct {
	TypedNameV   plugin.TypedName
	NewStateFunc func(ctx context.Context) any
	PickFunc     func(ctx context.Context, flowGroup flowcontrol.PriorityBandAccessor) (flowcontrol.FlowQueueAccessor, error)
}

func (m *MockFairnessPolicy) TypedName() plugin.TypedName { return m.TypedNameV }

func (m *MockFairnessPolicy) NewState(ctx context.Context) any {
	if m.NewStateFunc != nil {
		return m.NewStateFunc(ctx)
	}
	return nil
}

func (m *MockFairnessPolicy) Pick(ctx context.Context, flowGroup flowcontrol.PriorityBandAccessor) (flowcontrol.FlowQueueAccessor, error) {
	if m.PickFunc != nil {
		return m.PickFunc(ctx, flowGroup)
	}
	return nil, nil
}

var _ flowcontrol.FairnessPolicy = &MockFairnessPolicy{}
