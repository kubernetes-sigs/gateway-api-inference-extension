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
