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
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// MockFlowControlRequest provides a mock implementation of the FlowControlRequest interface.
type MockFlowControlRequest struct {
	FlowKeyV             flowcontrol.FlowKey
	ByteSizeV            uint64
	InitialEffectiveTTLV time.Duration
	IDV                  string
	MetadataV            map[string]any
	InferencePoolNameV   string
	ModelNameV           string
	TargetModelNameV     string
}

// MockRequestOption is a functional option for configuring a MockFlowControlRequest.
type MockRequestOption func(*MockFlowControlRequest)

// WithInferencePoolName sets the InferencePoolName for the mock request.
func WithInferencePoolName(name string) MockRequestOption {
	return func(m *MockFlowControlRequest) {
		m.InferencePoolNameV = name
	}
}

// WithModelName sets the ModelName for the mock request.
func WithModelName(name string) MockRequestOption {
	return func(m *MockFlowControlRequest) {
		m.ModelNameV = name
	}
}

// WithTargetModelName sets the TargetModelName for the mock request.
func WithTargetModelName(name string) MockRequestOption {
	return func(m *MockFlowControlRequest) {
		m.TargetModelNameV = name
	}
}

// NewMockFlowControlRequest creates a new MockFlowControlRequest instance with optional configuration.
func NewMockFlowControlRequest(
	byteSize uint64,
	id string,
	key flowcontrol.FlowKey,
	opts ...MockRequestOption,
) *MockFlowControlRequest {
	m := &MockFlowControlRequest{
		ByteSizeV: byteSize,
		IDV:       id,
		FlowKeyV:  key,
		MetadataV: make(map[string]any),
	}

	for _, opt := range opts {
		opt(m)
	}

	return m
}

func (m *MockFlowControlRequest) FlowKey() flowcontrol.FlowKey       { return m.FlowKeyV }
func (m *MockFlowControlRequest) ByteSize() uint64                   { return m.ByteSizeV }
func (m *MockFlowControlRequest) InitialEffectiveTTL() time.Duration { return m.InitialEffectiveTTLV }
func (m *MockFlowControlRequest) ID() string                         { return m.IDV }
func (m *MockFlowControlRequest) GetMetadata() map[string]any        { return m.MetadataV }
func (m *MockFlowControlRequest) InferencePoolName() string          { return m.InferencePoolNameV }
func (m *MockFlowControlRequest) ModelName() string                  { return m.ModelNameV }
func (m *MockFlowControlRequest) TargetModelName() string            { return m.TargetModelNameV }

var _ flowcontrol.FlowControlRequest = &MockFlowControlRequest{}

// MockQueueItemHandle provides a mock implementation of the QueueItemHandle interface.
type MockQueueItemHandle struct {
	RawHandle      any
	IsInvalidatedV bool
}

func (m *MockQueueItemHandle) Handle() any         { return m.RawHandle }
func (m *MockQueueItemHandle) Invalidate()         { m.IsInvalidatedV = true }
func (m *MockQueueItemHandle) IsInvalidated() bool { return m.IsInvalidatedV }

var _ flowcontrol.QueueItemHandle = &MockQueueItemHandle{}

// MockQueueItemAccessor provides a mock implementation of the QueueItemAccessor interface.
type MockQueueItemAccessor struct {
	EnqueueTimeV     time.Time
	EffectiveTTLV    time.Duration
	OriginalRequestV flowcontrol.FlowControlRequest
	HandleV          flowcontrol.QueueItemHandle
}

func (m *MockQueueItemAccessor) EnqueueTime() time.Time      { return m.EnqueueTimeV }
func (m *MockQueueItemAccessor) EffectiveTTL() time.Duration { return m.EffectiveTTLV }

func (m *MockQueueItemAccessor) OriginalRequest() flowcontrol.FlowControlRequest {
	if m.OriginalRequestV == nil {
		return &MockFlowControlRequest{}
	}
	return m.OriginalRequestV
}

func (m *MockQueueItemAccessor) Handle() flowcontrol.QueueItemHandle          { return m.HandleV }
func (m *MockQueueItemAccessor) SetHandle(handle flowcontrol.QueueItemHandle) { m.HandleV = handle }

var _ flowcontrol.QueueItemAccessor = &MockQueueItemAccessor{}

// NewMockQueueItemAccessor is a constructor for MockQueueItemAccessor that initializes the mock with a default
// MockFlowControlRequest and MockQueueItemHandle to prevent nil pointer dereferences in tests.
// It accepts MockRequestOptions to configure the underlying request.
func NewMockQueueItemAccessor(
	byteSize uint64,
	reqID string,
	key flowcontrol.FlowKey,
	opts ...MockRequestOption,
) *MockQueueItemAccessor {
	return &MockQueueItemAccessor{
		EnqueueTimeV: time.Now(),
		OriginalRequestV: NewMockFlowControlRequest(
			byteSize,
			reqID,
			key,
			opts...,
		),
		HandleV: &MockQueueItemHandle{},
	}
}

// MockFlowQueueAccessor is a simple stub mock for the FlowQueueAccessor interface.
// It is used for tests that require static, predictable return values from a queue accessor.
// For complex, stateful queue behavior, use the mock in ../../contracts/mocks.MockManagedQueue.
type MockFlowQueueAccessor struct {
	NameV           string
	LenV            int
	ByteSizeV       uint64
	PeekHeadV       flowcontrol.QueueItemAccessor
	PeekTailV       flowcontrol.QueueItemAccessor
	FlowKeyV        flowcontrol.FlowKey
	OrderingPolicyV flowcontrol.OrderingPolicy
	CapabilitiesV   []flowcontrol.QueueCapability
}

func (m *MockFlowQueueAccessor) Name() string                                { return m.NameV }
func (m *MockFlowQueueAccessor) Len() int                                    { return m.LenV }
func (m *MockFlowQueueAccessor) ByteSize() uint64                            { return m.ByteSizeV }
func (m *MockFlowQueueAccessor) OrderingPolicy() flowcontrol.OrderingPolicy  { return m.OrderingPolicyV }
func (m *MockFlowQueueAccessor) FlowKey() flowcontrol.FlowKey                { return m.FlowKeyV }
func (m *MockFlowQueueAccessor) Capabilities() []flowcontrol.QueueCapability { return m.CapabilitiesV }

func (m *MockFlowQueueAccessor) PeekHead() flowcontrol.QueueItemAccessor {
	return m.PeekHeadV
}

func (m *MockFlowQueueAccessor) PeekTail() flowcontrol.QueueItemAccessor {
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
	FlowKeysFunc      func() []flowcontrol.FlowKey
	QueueFunc         func(flowID string) flowcontrol.FlowQueueAccessor
	IterateQueuesFunc func(callback func(flow flowcontrol.FlowQueueAccessor) (keepIterating bool))
}

func (m *MockPriorityBandAccessor) Priority() int        { return m.PriorityV }
func (m *MockPriorityBandAccessor) PriorityName() string { return m.PriorityNameV }
func (m *MockPriorityBandAccessor) PolicyState() any     { return m.PolicyStateV }

func (m *MockPriorityBandAccessor) FlowKeys() []flowcontrol.FlowKey {
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

// MockOrderingPolicy is a behavioral mock for the OrderingPolicy interface.
// Simple accessors are configured with public value fields (e.g., TypedNameV).
// Complex methods with logic are configured with function fields (e.g., LessFunc).
type MockOrderingPolicy struct {
	TypedNameV                 plugin.TypedName
	LessFunc                   func(a, b flowcontrol.QueueItemAccessor) bool
	RequiredQueueCapabilitiesV []flowcontrol.QueueCapability
}

func (m *MockOrderingPolicy) TypedName() plugin.TypedName { return m.TypedNameV }

func (m *MockOrderingPolicy) Less(a, b flowcontrol.QueueItemAccessor) bool {
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
