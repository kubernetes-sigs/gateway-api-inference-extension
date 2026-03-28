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

package ordering

import (
	"testing"
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol/mocks"
)

func TestSLODeadlinePolicy_Name(t *testing.T) {
	t.Parallel()
	policy := newSLODeadlinePolicy()
	assert.Equal(t, SLODeadlineOrderingPolicyType, policy.Name())
}

func TestSLODeadlinePolicy_WithName(t *testing.T) {
	t.Parallel()
	policy := newSLODeadlinePolicy().withName("test-name")
	assert.Equal(t, "test-name", policy.Name())
}

func TestSLODeadlinePolicy_RequiredQueueCapabilities(t *testing.T) {
	t.Parallel()
	policy := newSLODeadlinePolicy()
	caps := policy.RequiredQueueCapabilities()
	require.Len(t, caps, 1)
	assert.Equal(t, flowcontrol.CapabilityPriorityConfigurable, caps[0])
}

// makeSLOItem builds a QueueItemAccessor with the given SLO header and received time.
func makeSLOItem(id string, received time.Time, sloTTFTMs string) flowcontrol.QueueItemAccessor {
	req := mocks.NewMockFlowControlRequest(10, id, testFlowKey)
	req.ReceivedTimestampV = received
	req.InferenceRequestV = &requesthandling.InferenceRequest{LLM: &requesthandling.LLMRequest{Headers: map[string]string{sloTtftHeader: sloTTFTMs}}}
	return &mocks.MockQueueItemAccessor{
		EffectiveTTLV:    0,
		OriginalRequestV: req,
	}
}

func TestSLODeadline_Less(t *testing.T) {
	t.Parallel()
	policy := newSLODeadlinePolicy()

	now := time.Now()

	// A: received now, 100ms SLO → deadline now+100ms
	itemA := makeSLOItem("a", now, "100")
	// B: received now, 50ms SLO → deadline now+50ms (earlier)
	itemB := makeSLOItem("b", now, "50")
	// C: received now+20ms, 50ms SLO → deadline now+20ms+50ms = now+70ms (after B but earlier than A)
	itemC := makeSLOItem("c", now.Add(20*time.Millisecond), "50")
	// D: no header → far-future deadline
	reqD := mocks.NewMockFlowControlRequest(10, "d", testFlowKey)
	reqD.ReceivedTimestampV = now
	reqD.InferenceRequestV = &requesthandling.InferenceRequest{LLM: &requesthandling.LLMRequest{Headers: map[string]string{}}}
	itemD := &mocks.MockQueueItemAccessor{EffectiveTTLV: 0, OriginalRequestV: reqD}
	// E: same deadline as B (received 1s earlier + 1050ms SLO = now+50ms), earlier ReceivedTimestamp → wins tie-breaker
	itemE := makeSLOItem("e", now.Add(-time.Second), "1050")

	testCases := []struct {
		name     string
		a        flowcontrol.QueueItemAccessor
		b        flowcontrol.QueueItemAccessor
		expected bool
	}{
		{"earlier SLO deadline first (B before A)", itemB, itemA, true},
		{"later SLO deadline after (A after B)", itemA, itemB, false},
		{"received later but earlier deadline (C before A)", itemC, itemA, true},
		{"SLO-bound before no-header (A before D)", itemA, itemD, true},
		{"no-header after SLO-bound (D after A)", itemD, itemA, false},
		{"same deadline: earlier ReceivedTimestamp first (E before B)", itemE, itemB, true},
		{"same deadline: later ReceivedTimestamp after (B after E)", itemB, itemE, false},
		{"a is nil → b wins", nil, itemA, false},
		{"b is nil → a wins", itemA, nil, true},
		{"both nil → false", nil, nil, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tc.expected, policy.Less(tc.a, tc.b))
		})
	}
}

func TestCalculateSLODeadline(t *testing.T) {
	t.Parallel()

	now := time.Now()

	// Valid header
	reqValid := mocks.NewMockFlowControlRequest(1, "valid", testFlowKey)
	reqValid.ReceivedTimestampV = now
	reqValid.InferenceRequestV = &requesthandling.InferenceRequest{LLM: &requesthandling.LLMRequest{Headers: map[string]string{sloTtftHeader: "200"}}}
	accValid := &mocks.MockQueueItemAccessor{OriginalRequestV: reqValid}
	deadline := calculateSLODeadline(accValid)
	assert.Equal(t, now.Add(200*time.Millisecond), deadline)

	// Missing header
	reqNoHeader := mocks.NewMockFlowControlRequest(2, "no", testFlowKey)
	reqNoHeader.InferenceRequestV = &requesthandling.InferenceRequest{LLM: &requesthandling.LLMRequest{Headers: map[string]string{}}}
	accNoHeader := &mocks.MockQueueItemAccessor{OriginalRequestV: reqNoHeader}
	assert.Equal(t, sloMaxDeadlineTime, calculateSLODeadline(accNoHeader))

	reqNoHeader.InferenceRequestV = &requesthandling.InferenceRequest{LLM: &requesthandling.LLMRequest{Headers: map[string]string{"x-some-header": "200"}}}
	accNoHeader = &mocks.MockQueueItemAccessor{OriginalRequestV: reqNoHeader}
	assert.Equal(t, sloMaxDeadlineTime, calculateSLODeadline(accNoHeader))

	// Invalid value
	reqInvalid := mocks.NewMockFlowControlRequest(3, "inv", testFlowKey)
	reqInvalid.InferenceRequestV = &requesthandling.InferenceRequest{LLM: &requesthandling.LLMRequest{Headers: map[string]string{sloTtftHeader: "x"}}}
	accInvalid := &mocks.MockQueueItemAccessor{OriginalRequestV: reqInvalid}
	assert.Equal(t, sloMaxDeadlineTime, calculateSLODeadline(accInvalid))

	// Nil OriginalRequest
	accNilReq := &mocks.MockQueueItemAccessor{OriginalRequestV: nil}
	assert.Equal(t, sloMaxDeadlineTime, calculateSLODeadline(accNilReq))

	// Nil InferenceRequest
	reqNilInfReq := mocks.NewMockFlowControlRequest(4, "no-inf-req", testFlowKey)
	accNoInfReq := &mocks.MockQueueItemAccessor{OriginalRequestV: reqNilInfReq}
	assert.Equal(t, sloMaxDeadlineTime, calculateSLODeadline(accNoInfReq))
}
