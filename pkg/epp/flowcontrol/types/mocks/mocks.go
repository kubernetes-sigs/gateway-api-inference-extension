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
}

// NewMockFlowControlRequest creates a new MockFlowControlRequest instance.
func NewMockFlowControlRequest(
	byteSize uint64,
	id string,
	key types.FlowKey,
) *MockFlowControlRequest {
	return &MockFlowControlRequest{
		ByteSizeV: byteSize,
		IDV:       id,
		FlowKeyV:  key,
		MetadataV: make(map[string]any),
	}
}

func (m *MockFlowControlRequest) FlowKey() types.FlowKey             { return m.FlowKeyV }
func (m *MockFlowControlRequest) ByteSize() uint64                   { return m.ByteSizeV }
func (m *MockFlowControlRequest) InitialEffectiveTTL() time.Duration { return m.InitialEffectiveTTLV }
func (m *MockFlowControlRequest) ID() string                         { return m.IDV }
func (m *MockFlowControlRequest) GetMetadata() map[string]any        { return m.MetadataV }

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
func NewMockQueueItemAccessor(byteSize uint64, reqID string, key types.FlowKey) *MockQueueItemAccessor {
	return &MockQueueItemAccessor{
		EnqueueTimeV: time.Now(),
		OriginalRequestV: NewMockFlowControlRequest(
			byteSize,
			reqID,
			key,
		),
		HandleV: &MockQueueItemHandle{},
	}
}
