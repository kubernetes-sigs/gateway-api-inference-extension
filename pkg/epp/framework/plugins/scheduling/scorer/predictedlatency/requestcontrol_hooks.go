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

package predictedlatency

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/go-logr/logr"
	"github.com/jellydator/ttlcache/v3"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/sidecars/latencypredictorasync"
)

var _ requestcontrol.PreRequest = &PredictedLatency{}
var _ requestcontrol.ResponseReceived = &PredictedLatency{}
var _ requestcontrol.ResponseStreaming = &PredictedLatency{}
var _ requestcontrol.ResponseComplete = &PredictedLatency{}

type predictedLatencyCtx struct {
	schedulingRequest         schedulingtypes.LLMRequest
	targetMetadata            *fwkdl.EndpointMetadata
	schedulingResult          *schedulingtypes.SchedulingResult
	lastSeenMetrics           map[string]*fwkdl.Metrics
	lastTokenTimestamp        time.Time
	requestReceivedTimestamp  time.Time
	generatedTokenCount       int
	incomingModelName         string
	ttft                      float64
	predictedTTFT             float64
	avgTPOT                   float64
	avgPredictedTPOT          float64
	tokenSampler              *tokenSampler
	tpotObservations          []float64
	predictedTPOTObservations []float64

	prefixCacheScoresForEndpoints map[string]float64

	// ttftSLO is the target time to first token SLO for the request.
	ttftSLO float64
	// TPOTSLO is the target time per output token SLO for the request.
	avgTPOTSLO float64

	// predictedTTFTForScheduling is the map of pod names to predicted TTFT values for scheduling.
	predictionsForScheduling []endpointPredictionResult

	// boolean set if request has valid endpoint based on predictions
	hasValidEndpoint bool
}

func newPredictedLatencyContext(request *schedulingtypes.LLMRequest) *predictedLatencyCtx {
	return &predictedLatencyCtx{
		schedulingRequest:             *request,
		lastSeenMetrics:               make(map[string]*fwkdl.Metrics),
		prefixCacheScoresForEndpoints: make(map[string]float64),
		predictionsForScheduling:      make([]endpointPredictionResult, 0),
		hasValidEndpoint:              true,
	}
}

func (s *PredictedLatency) getPredictedLatencyContextForRequest(request *schedulingtypes.LLMRequest) (*predictedLatencyCtx, error) {
	id := request.Headers[requtil.RequestIdHeaderKey]
	if item := s.sloContextStore.Get(id); item != nil {
		return item.Value(), nil
	}
	return nil, fmt.Errorf("SLO context not found for request ID: %s", id)
}

// GetAvgTPOTSLO returns the average TPOT SLO for a request.
// Used by wrappers (e.g., P/D scorer) to get priority values for tracking.
func (s *PredictedLatency) GetAvgTPOTSLO(request *schedulingtypes.LLMRequest) (float64, error) {
	ctx, err := s.getPredictedLatencyContextForRequest(request)
	if err != nil {
		return 0, err
	}
	return ctx.avgTPOTSLO, nil
}

// GetSchedulingResult returns the scheduling result for a request.
// Used by wrappers (e.g., P/D scorer) to access profile results and scheduling information
// for custom hook logic and cleanup.
func (s *PredictedLatency) GetSchedulingResult(request *schedulingtypes.LLMRequest) (*schedulingtypes.SchedulingResult, error) {
	ctx, err := s.getPredictedLatencyContextForRequest(request)
	if err != nil {
		return nil, err
	}
	return ctx.schedulingResult, nil
}

func (s *PredictedLatency) setPredictedLatencyContextForRequest(request *schedulingtypes.LLMRequest, ctx *predictedLatencyCtx) {
	id := request.Headers[requtil.RequestIdHeaderKey]
	s.sloContextStore.Set(id, ctx, ttlcache.DefaultTTL)
}

func (s *PredictedLatency) deletePredictedLatencyContextForRequest(request *schedulingtypes.LLMRequest) {
	id := request.Headers[requtil.RequestIdHeaderKey]
	s.sloContextStore.Delete(id)
}

// RecordTrainingForProfile records training data for a specific scheduling profile.
// This high-level method encapsulates all complexity of assembling and sending training data,
// allowing custom scorers (e.g., P/D-aware) to record training without manual assembly.
//
// Parameters:
//   - profileName: The scheduling profile to record training for (e.g., "prefill", "decode")
//   - actualTTFT: Measured time-to-first-token in milliseconds
//   - actualTPOT: Measured time-per-output-token in milliseconds
//   - generatedTokens: Number of tokens generated
//
// The method:
//  1. Retrieves request context and profile-specific data
//  2. Uses the configured requestBuilder to construct the training entry (respecting customizations like pod type labels)
//  3. Sends the training data to the predictor
//
// This maintains proper abstraction: GAIE handles HOW to record (mechanism),
// while custom builders define WHAT labels to add (policy).
func (s *PredictedLatency) RecordTrainingForProfile(
	ctx context.Context,
	request *schedulingtypes.LLMRequest,
	profileName string,
	actualTTFT float64,
	actualTPOT float64,
	generatedTokens int,
) error {
	logger := log.FromContext(ctx)

	// Get request context
	predictedLatencyCtx, err := s.getPredictedLatencyContextForRequest(request)
	if err != nil {
		return fmt.Errorf("failed to get request context: %w", err)
	}

	// Get scheduling result
	schedulingResult := predictedLatencyCtx.schedulingResult
	if schedulingResult == nil {
		return errors.New("no scheduling result available for request")
	}

	// Extract profile-specific data
	profileResult, exists := schedulingResult.ProfileResults[profileName]
	if !exists || profileResult == nil {
		return fmt.Errorf("profile %q not found in scheduling result", profileName)
	}

	if len(profileResult.TargetEndpoints) == 0 {
		return fmt.Errorf("no target endpoints for profile %q", profileName)
	}

	endpoint := profileResult.TargetEndpoints[0]
	endpointMetadata := endpoint.GetMetadata()

	// Get metrics for this profile
	metrics, exists := predictedLatencyCtx.lastSeenMetrics[profileName]
	if !exists || metrics == nil {
		return fmt.Errorf("no metrics available for profile %q", profileName)
	}

	// Get prefix cache score
	prefixCacheScore := predictedLatencyCtx.prefixCacheScoresForEndpoints[endpointMetadata.String()]

	// Get prompt
	prompt := predictedLatencyCtx.schedulingRequest.Body.Completions.Prompt

	// Build training entry using configured builder
	// The builder may add custom labels (e.g., PDPredictionRequestBuilder adds pod type)
	entry := s.requestBuilder.BuildTrainingEntry(
		ctx,
		endpointMetadata,
		metrics,
		prompt,
		actualTTFT,
		actualTPOT,
		time.Now(),
		generatedTokens,
		prefixCacheScore,
	)

	// Send to predictor
	if err := s.latencypredictor.AddTrainingDataBulk([]latencypredictor.TrainingEntry{entry}); err != nil {
		return fmt.Errorf("failed to record training data: %w", err)
	}

	logger.V(logutil.DEBUG).Info("Recorded training data for profile",
		"profile", profileName,
		"pod", endpointMetadata.NamespacedName.Name,
		"actualTTFT", actualTTFT,
		"actualTPOT", actualTPOT,
		"generatedTokens", generatedTokens,
	)

	return nil
}

// --- RequestControl Hooks ---

func (t *PredictedLatency) PreRequest(ctx context.Context, request *schedulingtypes.LLMRequest, schedulingResult *schedulingtypes.SchedulingResult) {
	logger := log.FromContext(ctx)
	if request == nil {
		logger.V(logutil.DEBUG).Info("PredictedLatency.PreRequest: request is nil, skipping")
		return
	}

	if schedulingResult == nil || len(schedulingResult.ProfileResults) == 0 {
		logger.V(logutil.TRACE).Info("PredictedLatency: Skipping PreRequest because no scheduling result was provided.")
		return
	}

	targetMetadata := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName].TargetEndpoints[0].GetMetadata()
	if !t.checkPredictor(logger, targetMetadata) {
		return
	}

	endpointName := types.NamespacedName{
		Name:      targetMetadata.NamespacedName.Name,
		Namespace: targetMetadata.NamespacedName.Namespace,
	}

	logger.V(logutil.TRACE).Info("request ID for SLO tracking", "requestID", request.Headers[requtil.RequestIdHeaderKey], "endpointName", endpointName)
	if request.Headers[requtil.RequestIdHeaderKey] == "" {
		logger.V(logutil.DEBUG).Error(errors.New("missing request ID"), "PredictedLatency.PreRequest: Request is missing request ID header")
		return
	}

	id := request.Headers[requtil.RequestIdHeaderKey]

	// Get or create queue for this endpoint using sync.Map
	actual, _ := t.runningRequestLists.LoadOrStore(endpointName, newRequestPriorityQueue())
	endpointRequestList := actual.(*requestPriorityQueue)

	predictedLatencyCtx, err := t.getPredictedLatencyContextForRequest(request)
	if err != nil {
		id := request.Headers[requtil.RequestIdHeaderKey]
		logger.V(logutil.DEBUG).Error(err, "PredictedLatency.PreRequest: Failed to get SLO context for request", "requestID", id)
		return
	}

	added := endpointRequestList.Add(id, predictedLatencyCtx.avgTPOTSLO)
	if !added {
		logger.V(logutil.TRACE).Info("PredictedLatency: Item already exists in queue", "endpointName", endpointName, "requestID", id)
	}

	// Set up SLO request context
	predictedLatencyCtx.targetMetadata = targetMetadata
	predictedLatencyCtx.schedulingResult = schedulingResult
	predictedLatencyCtx.requestReceivedTimestamp = time.Now()
	refreshLastSeenMetrics(ctx, predictedLatencyCtx)
	t.setPredictedLatencyContextForRequest(request, predictedLatencyCtx)

	if err := processPreRequestForLatencyPrediction(ctx, t.latencypredictor, t.requestBuilder, predictedLatencyCtx); err != nil {
		logger.V(logutil.DEBUG).Error(err, "Process PreRequest in latencypredictor failed")
	}
}

func (t *PredictedLatency) ResponseReceived(ctx context.Context, request *schedulingtypes.LLMRequest, response *requestcontrol.Response, targetMetadata *fwkdl.EndpointMetadata) {
	logger := log.FromContext(ctx)
	if request == nil {
		logger.V(logutil.DEBUG).Info("PredictedLatency.ResponseReceived: request is nil, skipping")
		return
	}
}

// --- Response Hooks when body chunks received---
func (t *PredictedLatency) ResponseStreaming(ctx context.Context, request *schedulingtypes.LLMRequest, response *requestcontrol.Response, targetMetadata *fwkdl.EndpointMetadata) {
	logger := log.FromContext(ctx)
	if request == nil {
		logger.V(logutil.DEBUG).Info("PredictedLatency.ResponseStreaming: request is nil, skipping")
		return
	}
	if !t.checkPredictor(logger, targetMetadata) || response.EndOfStream || !t.config.StreamingMode {
		return
	}

	now := time.Now()
	predictedLatencyCtx, err := t.getPredictedLatencyContextForRequest(request)
	if err != nil {
		id := request.Headers[requtil.RequestIdHeaderKey]
		logger.V(logutil.TRACE).Error(err, "PredictedLatency.ResponseStreaming: Failed to get SLO context for request", "requestID", id)
		return
	}

	if predictedLatencyCtx.ttft == 0 {
		processFirstTokenForLatencyPrediction(ctx, t.latencypredictor, t.config.StreamingMode, t.requestBuilder, predictedLatencyCtx, now, t.config.SamplingMean, t.config.MaxSampledTokens)
	} else {
		processTokenForLatencyPrediction(ctx, t.latencypredictor, t.requestBuilder, predictedLatencyCtx, targetMetadata, now, t.config.SamplingMean, t.config.MaxSampledTokens)
	}

}

func (t *PredictedLatency) ResponseComplete(ctx context.Context, request *schedulingtypes.LLMRequest, response *requestcontrol.Response, metadata *fwkdl.EndpointMetadata) {
	logger := log.FromContext(ctx)
	if request == nil {
		logger.V(logutil.DEBUG).Info("PredictedLatency.ResponseComplete: request is nil, skipping")
		return
	}
	targetMetadata := metadata
	if !t.checkPredictor(logger, targetMetadata) {
		return
	}

	predictedLatencyCtx, err := t.getPredictedLatencyContextForRequest(request)
	if err != nil {
		id := request.Headers[requtil.RequestIdHeaderKey]
		logger.V(logutil.DEBUG).Error(err, "PredictedLatency.ResponseComplete: Failed to get SLO context for request", "requestID", id)
		return
	}
	now := time.Now()
	if !t.config.StreamingMode {
		processFirstTokenForLatencyPrediction(ctx, t.latencypredictor, t.config.StreamingMode, t.requestBuilder, predictedLatencyCtx, now, t.config.SamplingMean, t.config.MaxSampledTokens)
	}

	if predictedLatencyCtx.ttft > 0 {
		logger.V(logutil.TRACE).Info("Averages calculated", "avgActualTTFT", predictedLatencyCtx.ttft, "avgPredictedTTFT", predictedLatencyCtx.predictedTTFT)
		metrics.RecordRequestTTFT(ctx, predictedLatencyCtx.incomingModelName, request.TargetModel, predictedLatencyCtx.ttft/1000)
		metrics.RecordRequestPredictedTTFT(ctx, predictedLatencyCtx.incomingModelName, request.TargetModel, predictedLatencyCtx.predictedTTFT/1000)
		if predictedLatencyCtx.ttftSLO > 0 {
			metrics.RecordRequestTTFTWithSLO(ctx, predictedLatencyCtx.incomingModelName, request.TargetModel, predictedLatencyCtx.ttft, predictedLatencyCtx.ttftSLO)
		}
	}

	if predictedLatencyCtx.avgTPOT > 0 {
		logger.V(logutil.TRACE).Info("Averages calculated", "avgActualTPOT", predictedLatencyCtx.avgTPOT, "avgPredictedTPOT", predictedLatencyCtx.avgPredictedTPOT)
		metrics.RecordRequestTPOT(ctx, predictedLatencyCtx.incomingModelName, request.TargetModel, predictedLatencyCtx.avgTPOT/1000)
		metrics.RecordRequestPredictedTPOT(ctx, predictedLatencyCtx.incomingModelName, request.TargetModel, predictedLatencyCtx.avgPredictedTPOT/1000)
		if predictedLatencyCtx.avgTPOTSLO > 0 {
			metrics.RecordRequestTPOTWithSLO(ctx, predictedLatencyCtx.incomingModelName, request.TargetModel, predictedLatencyCtx.avgTPOT, predictedLatencyCtx.avgTPOTSLO)
		}
	}

	id := request.Headers[requtil.RequestIdHeaderKey]
	t.removeRequestFromQueue(id, predictedLatencyCtx)
	t.deletePredictedLatencyContextForRequest(request)
}

func (t *PredictedLatency) checkPredictor(logger logr.Logger, metadata *fwkdl.EndpointMetadata) bool {
	if metadata == nil {
		logger.V(logutil.TRACE).Info("PredictedLatency: Skipping hook because no target metadata was provided.")
		return false
	}
	if t.latencypredictor == nil {
		logger.V(logutil.TRACE).Info("PredictedLatency: Skipping hook because predictor missing")
		return false
	}
	return true
}
