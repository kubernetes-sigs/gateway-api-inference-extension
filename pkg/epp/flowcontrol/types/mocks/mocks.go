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

// Package mocks provides simple, configurable mock implementations of the core flow control types, intended for use in
// unit and integration tests.
package mocks

import (
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
)

// MockFlowControlRequest provides a mock implementation of the types.FlowControlRequest interface.
type MockFlowControlRequest struct {
	FlowKeyV             types.FlowKey
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
	key types.FlowKey,
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

func (m *MockFlowControlRequest) FlowKey() types.FlowKey             { return m.FlowKeyV }
func (m *MockFlowControlRequest) ByteSize() uint64                   { return m.ByteSizeV }
func (m *MockFlowControlRequest) InitialEffectiveTTL() time.Duration { return m.InitialEffectiveTTLV }
func (m *MockFlowControlRequest) ID() string                         { return m.IDV }
func (m *MockFlowControlRequest) GetMetadata() map[string]any        { return m.MetadataV }
func (m *MockFlowControlRequest) InferencePoolName() string          { return m.InferencePoolNameV }
func (m *MockFlowControlRequest) ModelName() string                  { return m.ModelNameV }
func (m *MockFlowControlRequest) TargetModelName() string            { return m.TargetModelNameV }

var _ types.FlowControlRequest = &MockFlowControlRequest{}

// MockQueueItemHandle provides a mock implementation of the `types.QueueItemHandle` interface.
type MockQueueItemHandle struct {
	RawHandle      any
	IsInvalidatedV bool
}

func (m *MockQueueItemHandle) Handle() any         { return m.RawHandle }
func (m *MockQueueItemHandle) Invalidate()         { m.IsInvalidatedV = true }
func (m *MockQueueItemHandle) IsInvalidated() bool { return m.IsInvalidatedV }

var _ types.QueueItemHandle = &MockQueueItemHandle{}

// MockQueueItemAccessor provides a mock implementation of the `types.QueueItemAccessor` interface.
type MockQueueItemAccessor struct {
	EnqueueTimeV     time.Time
	EffectiveTTLV    time.Duration
	OriginalRequestV types.FlowControlRequest
	HandleV          types.QueueItemHandle
}

func (m *MockQueueItemAccessor) EnqueueTime() time.Time      { return m.EnqueueTimeV }
func (m *MockQueueItemAccessor) EffectiveTTL() time.Duration { return m.EffectiveTTLV }

func (m *MockQueueItemAccessor) OriginalRequest() types.FlowControlRequest {
	if m.OriginalRequestV == nil {
		return &MockFlowControlRequest{}
	}
	return m.OriginalRequestV
}

func (m *MockQueueItemAccessor) Handle() types.QueueItemHandle          { return m.HandleV }
func (m *MockQueueItemAccessor) SetHandle(handle types.QueueItemHandle) { m.HandleV = handle }

var _ types.QueueItemAccessor = &MockQueueItemAccessor{}

// NewMockQueueItemAccessor is a constructor for `MockQueueItemAccessor` that initializes the mock with a default
// `MockFlowControlRequest` and `MockQueueItemHandle` to prevent nil pointer dereferences in tests.
// It accepts MockRequestOptions to configure the underlying request.
func NewMockQueueItemAccessor(
	byteSize uint64,
	reqID string,
	key types.FlowKey,
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
