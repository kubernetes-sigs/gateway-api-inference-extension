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
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/handlers"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
)

// AdmissionController defines the interface for making admission control decisions.
// Implementations of this interface determine whether an incoming inference request should be accepted or rejected
// based on various criteria such as system load, fairness, priority, and available capacity.
type AdmissionController interface {
	// Admit determines if a request should be admitted.
	// It is called by the Director for each incoming request.
	//
	// Args:
	//   ctx: The request context, carrying deadlines, cancellation signals, and logger.
	//   reqCtx: The handlers.RequestContext containing details about the incoming request.
	//   candidatePods: A list of potential backend pods that can serve the request.
	//   priority: The priority level of the request, as determined by the InferenceObjective.
	//
	// Returns:
	//   - nil: If the request is admitted and should proceed to scheduling.
	//   - errutil.Error: If the request is rejected.
	Admit(
		ctx context.Context,
		reqCtx *handlers.RequestContext,
		candidatePods []backendmetrics.PodMetrics,
		priority int,
	) error
}

// saturationDetector defines the minimal interface required for checking if the backend pool is saturated.
type saturationDetector interface {
	IsSaturated(ctx context.Context, candidatePods []backendmetrics.PodMetrics) bool
}

// flowController defines the minimal interface required by FlowControlAdmissionController for enqueuing requests and
// waiting for an admission outcome.
type flowController interface {
	EnqueueAndWait(ctx context.Context, req types.FlowControlRequest) (types.QueueOutcome, error)
}

// rejectIfSheddableAndSaturated checks if a request should be immediately rejected because it's sheddable
// (priority < 0) and the system is saturated.
func rejectIfSheddableAndSaturated(
	ctx context.Context,
	sd saturationDetector,
	reqCtx *handlers.RequestContext,
	candidatePods []backendmetrics.PodMetrics,
	priority int,
) error {
	if requtil.IsSheddable(priority) {
		logger := log.FromContext(ctx)
		if sd.IsSaturated(ctx, candidatePods) {
			logger.V(logutil.TRACE).Info("Request rejected: system saturated and request is sheddable",
				"requestID", reqCtx.SchedulingRequest.RequestId)
			return errutil.Error{
				Code: errutil.InferencePoolResourceExhausted,
				Msg:  "system saturated, sheddable request dropped",
			}
		}
	}
	return nil
}

// --- LegacyAdmissionController ---

// LegacyAdmissionController implements saturation-based admission control.
// It rejects sheddable requests (priority < 0) if the saturationDetector indicates that the system is currently
// saturated. Non-sheddable requests always bypass the saturation check.
type LegacyAdmissionController struct {
	saturationDetector saturationDetector
}

// NewLegacyAdmissionController creates a new LegacyAdmissionController.
func NewLegacyAdmissionController(sd saturationDetector) *LegacyAdmissionController {
	return &LegacyAdmissionController{saturationDetector: sd}
}

// Admit implements the AdmissionController interface for the legacy strategy.
// It checks for saturation only for requests with priority < 0.
func (lac *LegacyAdmissionController) Admit(
	ctx context.Context,
	reqCtx *handlers.RequestContext,
	candidatePods []backendmetrics.PodMetrics,
	priority int,
) error {
	logger := log.FromContext(ctx)
	logger.V(logutil.TRACE).Info("Executing LegacyAdmissionController",
		"priority", priority, "fairnessID", reqCtx.FairnessID)
	if err := rejectIfSheddableAndSaturated(ctx, lac.saturationDetector, reqCtx, candidatePods, priority); err != nil {
		return err
	}
	logger.V(logutil.TRACE).Info("Request admitted", "requestID", reqCtx.SchedulingRequest.RequestId)
	return nil
}

// --- FlowControlAdmissionController ---

// FlowControlAdmissionController delegates admission decisions to the Flow Control layer.
// It first checks if the request is sheddable and the system is saturated, rejecting immediately if both conditions are
// true. Otherwise, it uses the provided flowController to enqueue the request and await an outcome.
type FlowControlAdmissionController struct {
	saturationDetector saturationDetector
	flowController     flowController
}

// NewFlowControlAdmissionController creates a new FlowControlAdmissionController.
// It requires a SaturationDetector and a flowController instance.
func NewFlowControlAdmissionController(sd saturationDetector, fc flowController) *FlowControlAdmissionController {
	return &FlowControlAdmissionController{
		saturationDetector: sd,
		flowController:     fc,
	}
}

// Admit implements the AdmissionController interface by checking for saturation on sheddable requests first, then
// deferring to the Flow Control system.
func (fcac *FlowControlAdmissionController) Admit(
	ctx context.Context,
	reqCtx *handlers.RequestContext,
	candidatePods []backendmetrics.PodMetrics,
	priority int,
) error {
	logger := log.FromContext(ctx)
	logger.V(logutil.TRACE).Info("Executing FlowControlAdmissionController",
		"requestID", reqCtx.SchedulingRequest.RequestId, "priority", priority, "fairnessID", reqCtx.FairnessID)
	if err := rejectIfSheddableAndSaturated(ctx, fcac.saturationDetector, reqCtx, candidatePods, priority); err != nil {
		return err
	}

	logger.V(logutil.TRACE).Info("Request proceeding to flow control", "requestID", reqCtx.SchedulingRequest.RequestId)

	fcReq := &flowControlRequest{
		requestID:       reqCtx.SchedulingRequest.RequestId,
		fairnessID:      reqCtx.FairnessID,
		priority:        priority,
		requestByteSize: uint64(reqCtx.RequestSize),
		candidatePods:   candidatePods,
	}

	outcome, err := fcac.flowController.EnqueueAndWait(ctx, fcReq)
	logger.V(logutil.DEBUG).Info("Flow control outcome",
		"requestID", reqCtx.SchedulingRequest.RequestId, "outcome", outcome, "error", err)
	return translateFlowControlOutcome(outcome, err)
}

// flowControlRequest is an adapter that implements the types.FlowControlRequest interface.
type flowControlRequest struct {
	requestID       string
	fairnessID      string
	priority        int
	requestByteSize uint64
	candidatePods   []backendmetrics.PodMetrics
}

var _ types.FlowControlRequest = &flowControlRequest{}

func (r *flowControlRequest) ID() string                         { return r.requestID }
func (r *flowControlRequest) InitialEffectiveTTL() time.Duration { return 0 } // Use controller default.
func (r *flowControlRequest) ByteSize() uint64                   { return r.requestByteSize }
func (r *flowControlRequest) CandidatePodsForScheduling() []backendmetrics.PodMetrics {
	return r.candidatePods
}
func (r *flowControlRequest) FlowKey() types.FlowKey {
	return types.FlowKey{ID: r.fairnessID, Priority: r.priority}
}

// translateFlowControlOutcome maps the context-rich outcome of the Flow Control layer to the public errutil.Error
// contract used by the Director.
func translateFlowControlOutcome(outcome types.QueueOutcome, err error) error {
	msg := "request rejected by flow control"
	if err != nil {
		msg = err.Error()
	}

	switch outcome {
	case types.QueueOutcomeDispatched:
		return nil
	case types.QueueOutcomeRejectedCapacity:
		return errutil.Error{Code: errutil.InferencePoolResourceExhausted, Msg: msg}
	case types.QueueOutcomeEvictedTTL:
		return errutil.Error{Code: errutil.ServiceUnavailable, Msg: "request timed out in queue: " + msg}
	case types.QueueOutcomeEvictedContextCancelled:
		return errutil.Error{Code: errutil.ServiceUnavailable, Msg: "client disconnected: " + msg}
	case types.QueueOutcomeRejectedOther, types.QueueOutcomeEvictedOther:
		return errutil.Error{Code: errutil.Internal, Msg: "internal flow control error: " + msg}
	default:
		return errutil.Error{Code: errutil.Internal, Msg: "unhandled flow control outcome: " + msg}
	}
}
