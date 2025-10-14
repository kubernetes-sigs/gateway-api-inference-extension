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
	fctypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/handlers"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// --- Mocks ---

type mockSaturationDetector struct {
	isSaturated bool
}

func (m *mockSaturationDetector) IsSaturated(_ context.Context, _ []backendmetrics.PodMetrics) bool {
	return m.isSaturated
}

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

func TestLegacyAdmissionController_Admit(t *testing.T) {
	t.Parallel()
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	candidatePods := []backendmetrics.PodMetrics{}
	reqCtx := &handlers.RequestContext{
		SchedulingRequest: &schedulingtypes.LLMRequest{RequestId: "test-req"},
	}

	testCases := []struct {
		name            string
		priority        int
		isSaturated     bool
		expectErr       bool
		expectErrCode   string
		expectErrSubstr string
	}{
		{
			name:        "non_sheddable_saturated_admit",
			priority:    0,
			isSaturated: true,
			expectErr:   false,
		},
		{
			name:        "sheddable_not_saturated_admit",
			priority:    -1,
			isSaturated: false,
			expectErr:   false,
		},
		{
			name:            "sheddable_saturated_reject",
			priority:        -1,
			isSaturated:     true,
			expectErr:       true,
			expectErrCode:   errutil.InferencePoolResourceExhausted,
			expectErrSubstr: "system saturated, sheddable request dropped",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			saturationDetector := &mockSaturationDetector{isSaturated: tc.isSaturated}
			ac := NewLegacyAdmissionController(saturationDetector)

			err := ac.Admit(ctx, reqCtx, candidatePods, tc.priority)

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

func TestFlowControlRequestAdapter(t *testing.T) {
	t.Parallel()
	candidatePods := []backendmetrics.PodMetrics{&backendmetrics.FakePodMetrics{}}

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
				candidatePods:   candidatePods,
			}

			assert.Equal(t, tc.requestID, fcReq.ID(), "ID() mismatch")
			assert.Equal(t, tc.requestByteSize, fcReq.ByteSize(), "ByteSize() mismatch")
			assert.Equal(t, candidatePods, fcReq.CandidatePodsForScheduling(), "CandidatePodsForScheduling() mismatch")
			assert.Equal(t, tc.expectFlowKey, fcReq.FlowKey(), "FlowKey() mismatch")
			assert.Zero(t, fcReq.InitialEffectiveTTL(), "InitialEffectiveTTL() should be zero")
		})
	}
}
func TestFlowControlAdmissionController_Admit(t *testing.T) {
	t.Parallel()
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	candidatePods := []backendmetrics.PodMetrics{}

	reqCtx := &handlers.RequestContext{
		SchedulingRequest: &schedulingtypes.LLMRequest{RequestId: "test-req"},
	}

	testCases := []struct {
		name            string
		priority        int
		isSaturated     bool
		fcOutcome       fctypes.QueueOutcome
		fcErr           error
		expectErr       bool
		expectErrCode   string
		expectErrSubstr string
		expectFCSkipped bool
	}{
		{
			name:            "sheddable_saturated_reject",
			priority:        -1,
			isSaturated:     true,
			expectErr:       true,
			expectErrCode:   errutil.InferencePoolResourceExhausted,
			expectErrSubstr: "system saturated, sheddable request dropped",
			expectFCSkipped: true,
		},
		{
			name:        "sheddable_not_saturated_dispatch",
			priority:    -1,
			isSaturated: false,
			fcOutcome:   fctypes.QueueOutcomeDispatched,
			expectErr:   false,
		},
		{
			name:        "non_sheddable_saturated_dispatch",
			priority:    0,
			isSaturated: true,
			fcOutcome:   fctypes.QueueOutcomeDispatched,
			expectErr:   false,
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
			sd := &mockSaturationDetector{isSaturated: tc.isSaturated}
			fc := &mockFlowController{outcome: tc.fcOutcome, err: tc.fcErr}
			ac := NewFlowControlAdmissionController(sd, fc)

			err := ac.Admit(ctx, reqCtx, candidatePods, tc.priority)

			if tc.expectFCSkipped {
				assert.False(t, fc.called, "FlowController should not have been called for scenario: %s", tc.name)
			} else {
				assert.True(t, fc.called, "FlowController should have been called for scenario: %s", tc.name)
			}

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
