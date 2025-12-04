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

package requestcontrol

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts/mocks"
	fctypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/handlers"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// --- Mocks ---

type mockFlowController struct {
	outcome fctypes.QueueOutcome
	err     error
	called  bool
}

func (m *mockFlowController) EnqueueAndWait(
	_ context.Context,
	_ fctypes.FlowControlRequest,
) (fctypes.QueueOutcome, error) {
	m.called = true
	return m.outcome, m.err
}

// --- Legacy Controller Tests ---

func TestLegacyAdmissionController_Admit(t *testing.T) {
	t.Parallel()
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	reqCtx := &handlers.RequestContext{
		SchedulingRequest: &schedulingtypes.LLMRequest{RequestId: "test-req"},
		Request: &handlers.Request{
			Metadata: map[string]any{},
		},
	}

	mockPods := []backendmetrics.PodMetrics{&backendmetrics.FakePodMetrics{}}

	testCases := []struct {
		name            string
		priority        int
		isSaturated     bool
		locatorPods     []backendmetrics.PodMetrics
		expectErr       bool
		expectErrCode   string
		expectErrSubstr string
	}{
		{
			name:        "non_sheddable_saturated_admit",
			priority:    0,
			isSaturated: true,
			locatorPods: mockPods,
			expectErr:   false,
		},
		{
			name:        "sheddable_not_saturated_admit",
			priority:    -1,
			isSaturated: false,
			locatorPods: mockPods,
			expectErr:   false,
		},
		{
			name:            "sheddable_saturated_reject",
			priority:        -1,
			isSaturated:     true,
			locatorPods:     mockPods,
			expectErr:       true,
			expectErrCode:   errutil.InferencePoolResourceExhausted,
			expectErrSubstr: "system saturated, sheddable request dropped",
		},
		{
			name:            "sheddable_no_pods_reject",
			priority:        -1,
			isSaturated:     true,
			locatorPods:     []backendmetrics.PodMetrics{},
			expectErr:       true,
			expectErrCode:   errutil.InferencePoolResourceExhausted,
			expectErrSubstr: "system saturated, sheddable request dropped",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			saturationDetector := &mocks.MockSaturationDetector{
				IsSaturatedFunc: func(context.Context, []backendmetrics.PodMetrics) bool { return tc.isSaturated },
			}
			locator := &mocks.MockPodLocator{Pods: tc.locatorPods}
			ac := NewLegacyAdmissionController(saturationDetector, locator)

			err := ac.Admit(ctx, reqCtx, tc.priority)

			if !tc.expectErr {
				assert.NoError(t, err, "Admit() should not have returned an error for scenario: %s", tc.name)
			} else {
				require.Error(t, err, "Admit() should have returned an error for scenario: %s", tc.name)
				var e errutil.Error
				if assert.ErrorAs(t, err, &e, "error should be of type errutil.Error") {
					assert.Equal(t, tc.expectErrCode, e.Code, "incorrect error code for scenario: %s", tc.name)
					assert.Contains(t, e.Msg, tc.expectErrSubstr, "incorrect error message substring for scenario: %s", tc.name)
				}
			}
		})
	}
}

// --- Flow Control Controller Tests ---

func TestFlowControlRequestAdapter(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name            string
		requestID       string
		fairnessID      string
		priority        int
		requestByteSize uint64
		expectFlowKey   fctypes.FlowKey
	}{
		{
			name:            "simple",
			requestID:       "req-1",
			fairnessID:      "flow-1",
			priority:        10,
			requestByteSize: 1024,
			expectFlowKey:   fctypes.FlowKey{ID: "flow-1", Priority: 10},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			fcReq := &flowControlRequest{
				requestID:       tc.requestID,
				fairnessID:      tc.fairnessID,
				priority:        tc.priority,
				requestByteSize: tc.requestByteSize,
			}

			assert.Equal(t, tc.requestID, fcReq.ID(), "ID() mismatch")
			assert.Equal(t, tc.requestByteSize, fcReq.ByteSize(), "ByteSize() mismatch")
			assert.Equal(t, tc.expectFlowKey, fcReq.FlowKey(), "FlowKey() mismatch")
			assert.Zero(t, fcReq.InitialEffectiveTTL(), "InitialEffectiveTTL() should be zero")
		})
	}
}

func TestFlowControlAdmissionController_Admit(t *testing.T) {
	t.Parallel()
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	reqCtx := &handlers.RequestContext{
		SchedulingRequest: &schedulingtypes.LLMRequest{RequestId: "test-req"},
		Request: &handlers.Request{
			Metadata: map[string]any{},
		},
	}

	testCases := []struct {
		name            string
		priority        int
		fcOutcome       fctypes.QueueOutcome
		fcErr           error
		expectErr       bool
		expectErrCode   string
		expectErrSubstr string
	}{
		{
			name:      "sheddable_dispatched",
			priority:  -1,
			fcOutcome: fctypes.QueueOutcomeDispatched,
			expectErr: false,
		},
		{
			name:      "non_sheddable_dispatched",
			priority:  0,
			fcOutcome: fctypes.QueueOutcomeDispatched,
			expectErr: false,
		},
		{
			name:            "fc_reject_capacity",
			priority:        0,
			fcOutcome:       fctypes.QueueOutcomeRejectedCapacity,
			expectErr:       true,
			expectErrCode:   errutil.InferencePoolResourceExhausted,
			expectErrSubstr: "request rejected by flow control",
		},
		{
			name:            "fc_evict_ttl",
			priority:        0,
			fcOutcome:       fctypes.QueueOutcomeEvictedTTL,
			fcErr:           errors.New("timeout"),
			expectErr:       true,
			expectErrCode:   errutil.ServiceUnavailable,
			expectErrSubstr: "request timed out in queue: timeout",
		},
		{
			name:            "fc_evict_context_cancelled",
			priority:        0,
			fcOutcome:       fctypes.QueueOutcomeEvictedContextCancelled,
			expectErr:       true,
			expectErrCode:   errutil.ServiceUnavailable,
			expectErrSubstr: "client disconnected",
		},
		{
			name:            "fc_reject_other",
			priority:        0,
			fcOutcome:       fctypes.QueueOutcomeRejectedOther,
			expectErr:       true,
			expectErrCode:   errutil.Internal,
			expectErrSubstr: "internal flow control error",
		},
		{
			name:            "fc_evict_other",
			priority:        0,
			fcOutcome:       fctypes.QueueOutcomeEvictedOther,
			fcErr:           errors.New("internal error"),
			expectErr:       true,
			expectErrCode:   errutil.Internal,
			expectErrSubstr: "internal flow control error: internal error",
		},
		{
			name:            "fc_unhandled_outcome",
			priority:        0,
			fcOutcome:       fctypes.QueueOutcomeNotYetFinalized,
			expectErr:       true,
			expectErrCode:   errutil.Internal,
			expectErrSubstr: "unhandled flow control outcome",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			fc := &mockFlowController{outcome: tc.fcOutcome, err: tc.fcErr}
			ac := NewFlowControlAdmissionController(fc)

			err := ac.Admit(ctx, reqCtx, tc.priority)

			assert.True(t, fc.called, "FlowController should have been called for scenario: %s", tc.name)

			if !tc.expectErr {
				assert.NoError(t, err, "Admit() returned an unexpected error for scenario: %s", tc.name)
			} else {
				require.Error(t, err, "Admit() should have returned an error for scenario: %s", tc.name)
				var e errutil.Error
				if assert.ErrorAs(t, err, &e, "error should be of type errutil.Error") {
					assert.Equal(t, tc.expectErrCode, e.Code, "incorrect error code for scenario: %s", tc.name)
					assert.Contains(t, e.Msg, tc.expectErrSubstr, "incorrect error message substring for scenario: %s", tc.name)
				}
			}
		})
	}
}
