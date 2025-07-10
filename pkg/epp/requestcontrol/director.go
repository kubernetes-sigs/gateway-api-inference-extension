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

// Package requestcontrol defines the Director component responsible for orchestrating request processing after initial
// parsing.
package requestcontrol

import (
	"context"
	"fmt"
	"math/rand"
	"net"
	"strconv"
	"strings"
	"time"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/handlers"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"

	// Assuming the predictor is located here. Adjust the import path if necessary.
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/latencypredictorasync"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
)

// Datastore defines the interface required by the Director.
type Datastore interface {
	PoolGet() (*v1.InferencePool, error)
	ObjectiveGet(modelName string) *v1alpha2.InferenceObjective
	PodList(predicate func(backendmetrics.PodMetrics) bool) []backendmetrics.PodMetrics
}

/*
NOTE: To support this refined logic, the `handlers.RequestContext` struct
(defined in a different package) would need to be updated as follows:

type RequestContext struct {
    // ... existing fields ...
	RequestReceivedTimestamp time.Time
	FirstTokenTimestamp      time.Time
	ResponseCompleteTimestamp time.Time
	IsModelServerStreaming   func() bool
	ResponseComplete         bool
	Prompt                   string
	LastSeenMetrics           *backend.Metrics
    // ... etc ...

    // -- New fields for latency predictor --
    PredictedTTFT                float64   // The predicted TTFT in milliseconds
    PredictedTPOT                float64   // The predicted TPOT in milliseconds
    TTFT                         float64   // Actual Time To First Token in milliseconds
    LastTokenTimestamp           time.Time // Timestamp of the last token received
    TPOTObservations            []float64  // All actual inter-token latencies (for which we have predictions)
    PredictedTPOTObservations   []float64  // Predicted inter-token latencies (only for sampled tokens)
    GeneratedTokenCount          int       // Current number of tokens generated
}

*/

const (
	subsetHintNamespace = "envoy.lb.subset_hint"
	subsetHintKey       = "x-gateway-destination-endpoint-subset"
	// Poisson sampling parameters for predictions
	defaultSamplingMean = 50 // Mean interval between prediction samples (tokens)
	maxSampledTokens    = 50 // Maximum number of prediction samples per request
)

// splitWords splits a string into words based on whitespace and returns the resulting slice.
func splitWords(input string) []string {
	return strings.Fields(input)
}

// calculateRunningAverage calculates the running average efficiently
func calculateRunningAverage(currentAvg float64, newValue float64, count int) float64 {
	if count == 0 {
		return 0
	}
	if count == 1 {
		return newValue
	}
	return currentAvg + (newValue-currentAvg)/float64(count)
}

// Scheduler defines the interface required by the Director for scheduling.
type Scheduler interface {
	Schedule(ctx context.Context, request *schedulingtypes.LLMRequest, candidatePods []schedulingtypes.Pod) (result *schedulingtypes.SchedulingResult, err error)
}

// SaturationDetector provides a signal indicating whether the backends are considered saturated.
type SaturationDetector interface {
	IsSaturated(ctx context.Context, candidatePods []backendmetrics.PodMetrics) bool
}

// Predictor defines the interface required by the Director for latency prediction and training.
// The real *latencypredictor.Predictor satisfies this interface.
type Predictor interface {
	Predict(ctx context.Context, req latencypredictor.PredictionRequest) (*latencypredictor.PredictionResponse, error)
	AddTrainingDataBulk(entry []latencypredictor.TrainingEntry) error
}

// NewDirectorWithConfig creates a new Director instance with all dependencies.
// It accepts a pre-initialized latency predictor. The caller is responsible for creating
// and managing the lifecycle (Start/Stop) of the predictor.
func NewDirectorWithConfig(datastore Datastore, scheduler Scheduler, saturationDetector SaturationDetector, config *Config, predictor Predictor) *Director {
	return &Director{
		datastore:           datastore,
		scheduler:           scheduler,
		saturationDetector:  saturationDetector,
		latencyPredictor:    predictor,
		preRequestPlugins:   config.preRequestPlugins,
		postResponsePlugins: config.postResponsePlugins,
		defaultPriority:     0, // define default priority explicitly
	}
}

// Director orchestrates the request handling flow, including scheduling.
type Director struct {
	datastore           Datastore
	scheduler           Scheduler
	saturationDetector  SaturationDetector
	latencyPredictor    latencypredictor.PredictorInterface
	preRequestPlugins   []PreRequest
	postResponsePlugins []PostResponse
	// we just need a pointer to an int variable since priority is a pointer in InferenceObjective
	// no need to set this in the constructor, since the value we want is the default int val
	// and value types cannot be nil
	defaultPriority int
}

const (
	// Maximum number of TPOT observations to retain per request
	maxTPOTObservations = 4096
)

// HandleRequest orchestrates the request lifecycle:
//  1. Parses request details.
//  2. Calls admitRequest for admission control.
//  3. Calls Scheduler.Schedule if request is approved.
//  4. Calls prepareRequest to populate RequestContext with result and call PreRequest plugins.
//
// It always returns the requestContext even in the error case, as the request context is used in error handling.
func (d *Director) HandleRequest(ctx context.Context, reqCtx *handlers.RequestContext) (*handlers.RequestContext, error) {
	logger := log.FromContext(ctx)

	// Parse Request, Resolve Target Models, and Determine Parameters
	requestBodyMap := reqCtx.Request.Body
	var ok bool
	reqCtx.IncomingModelName, ok = requestBodyMap["model"].(string)

	if !ok {
		return reqCtx, errutil.Error{Code: errutil.BadRequest, Msg: "model not found in request body"}
	}
	if reqCtx.TargetModelName == "" {
		// Default to incoming model name
		reqCtx.TargetModelName = reqCtx.IncomingModelName
	}
	reqCtx.Request.Body["model"] = reqCtx.TargetModelName

	prompt, err := requtil.ExtractPromptFromRequestBody(requestBodyMap)
	if err != nil {
		return reqCtx, err
	}

	infObjective := d.datastore.ObjectiveGet(reqCtx.ObjectiveKey)
	if infObjective == nil {
		logger.V(logutil.VERBOSE).Info("No associated InferenceObjective found, using default", "objectiveKey", reqCtx.ObjectiveKey)
		infObjective = &v1alpha2.InferenceObjective{
			Spec: v1alpha2.InferenceObjectiveSpec{
				Priority: &d.defaultPriority,
			},
		}
	} else if infObjective.Spec.Priority == nil {
		// Default to 0 if not specified.
		infObjective.Spec.Priority = &d.defaultPriority
	}

	// Prepare LLMRequest (needed for both saturation detection and Scheduler)
	reqCtx.SchedulingRequest = &schedulingtypes.LLMRequest{
		RequestId:   reqCtx.Request.Headers[requtil.RequestIdHeaderKey],
		TargetModel: reqCtx.TargetModelName,
		Prompt:      prompt,
		Headers:     reqCtx.Request.Headers,
	}

	logger = logger.WithValues("objectiveKey", reqCtx.ObjectiveKey, "incomingModelName", reqCtx.IncomingModelName, "targetModelName", reqCtx.TargetModelName, "priority", infObjective.Spec.Priority)

	ctx = log.IntoContext(ctx, logger)
	logger.V(logutil.DEBUG).Info("LLM request assembled")

	// Get candidate pods for scheduling
	candidatePods := d.getCandidatePodsForScheduling(ctx, reqCtx.Request.Metadata)
	if len(candidatePods) == 0 {
		return reqCtx, errutil.Error{Code: errutil.ServiceUnavailable, Msg: "failed to find candidate pods for serving the request"}
	}

	// Admission Control check
	if err := d.admitRequest(ctx, candidatePods, *infObjective.Spec.Priority, reqCtx.FairnessID); err != nil {
		return reqCtx, err
	}

	result, err := d.scheduler.Schedule(ctx, reqCtx.SchedulingRequest, d.toSchedulerPodMetrics(candidatePods))
	if err != nil {
		return reqCtx, errutil.Error{Code: errutil.InferencePoolResourceExhausted, Msg: fmt.Errorf("failed to find target pod: %w", err).Error()}
	}

	// Prepare Request (Populates RequestContext and call PreRequest plugins)
	// Insert target endpoint to instruct Envoy to route requests to the specified target pod and attach the port number.
	// Invoke PreRequest registered plugins.
	reqCtx, err = d.prepareRequest(ctx, reqCtx, result)
	if err != nil {
		return reqCtx, err
	}

	return reqCtx, nil
}

// getCandidatePodsForScheduling gets the list of relevant endpoints for the scheduling cycle from the datastore.
// according to EPP protocol, if "x-gateway-destination-endpoint-subset" is set on the request metadata and specifies
// a subset of endpoints, only these endpoints will be considered as candidates for the scheduler.
// Snapshot pod metrics from the datastore to:
// 1. Reduce concurrent access to the datastore.
// 2. Ensure consistent data during the scheduling operation of a request between all scheduling cycles.
func (d *Director) getCandidatePodsForScheduling(ctx context.Context, requestMetadata map[string]any) []backendmetrics.PodMetrics {
	loggerTrace := log.FromContext(ctx).V(logutil.TRACE)

	subsetMap, found := requestMetadata[metadata.SubsetFilterNamespace].(map[string]any)
	if !found {
		return d.datastore.PodList(backendmetrics.AllPodsPredicate)
	}

	// Check if endpoint key is present in the subset map and ensure there is at least one value
	endpointSubsetList, found := subsetMap[metadata.SubsetFilterKey].([]any)
	if !found {
		return d.datastore.PodList(backendmetrics.AllPodsPredicate)
	} else if len(endpointSubsetList) == 0 {
		loggerTrace.Info("found empty subset filter in request metadata, filtering all pods")
		return []backendmetrics.PodMetrics{}
	}

	// Create a map of endpoint addresses for easy lookup
	endpoints := make(map[string]bool)
	for _, endpoint := range endpointSubsetList {
		// Extract address from endpoint
		// The endpoint is formatted as "<address>:<port>" (ex. "10.0.1.0:8080")
		epStr := strings.Split(endpoint.(string), ":")[0]
		endpoints[epStr] = true
	}

	podTotalCount := 0
	podFilteredList := d.datastore.PodList(func(pm backendmetrics.PodMetrics) bool {
		podTotalCount++
		if _, found := endpoints[pm.GetPod().Address]; found {
			return true
		}
		return false
	})

	loggerTrace.Info("filtered candidate pods by subset filtering", "podTotalCount", podTotalCount, "filteredCount", len(podFilteredList))

	return podFilteredList
}

// admitRequest handles admission control to decide whether or not to accept the request
// based on the request priority and saturation state.
func (d *Director) admitRequest(ctx context.Context, candidatePods []backendmetrics.PodMetrics, requestPriority int, fairnessID string) error {
	loggerTrace := log.FromContext(ctx).V(logutil.TRACE)

	loggerTrace.Info("Entering Flow Control", "priority", requestPriority, "fairnessID", fairnessID)

	// This will be removed in favor of a more robust implementation (Flow Control) in the very near future.
	// TODO: Make this a configurable value.
	// Tracking issue https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/1347
	if requestPriority >= 0 {
		loggerTrace.Info("Non-sheddable request bypassing saturation check.")
		return nil
	}

	if d.saturationDetector.IsSaturated(ctx, candidatePods) {
		return errutil.Error{
			Code: errutil.InferencePoolResourceExhausted,
			Msg:  "system saturated, sheddable request dropped",
		}
	}

	return nil
}

// prepareRequest populates the RequestContext and calls the registered PreRequest plugins
// for allowing plugging customized logic based on the scheduling result.
func (d *Director) prepareRequest(ctx context.Context, reqCtx *handlers.RequestContext, result *schedulingtypes.SchedulingResult) (*handlers.RequestContext, error) {
	if result == nil || len(result.ProfileResults) == 0 {
		return reqCtx, errutil.Error{Code: errutil.Internal, Msg: "empty scheduling results"}
	}

	pr, ok := result.ProfileResults[result.PrimaryProfileName]
	if ok && pr.TargetPods != nil {
		reqCtx.LastSeenMetrics = pr.TargetPods[0].GetMetrics().Clone()
	}

	// Always set endpoint even if metrics missing
	pod := pr.TargetPods[0].GetPod()
	pool, err := d.datastore.PoolGet()
	if err != nil {
		return reqCtx, err
	}
	targetPods := []*backend.Pod{}
	if len(pool.Spec.TargetPorts) != 1 {
		return reqCtx, errutil.Error{Code: errutil.BadRequest, Msg: "targetPorts should have length 1"}
	}
	targetPort := int(pool.Spec.TargetPorts[0].Number)
	targetEndpoints := []string{}

	for _, pod := range result.ProfileResults[result.PrimaryProfileName].TargetPods {
		curPod := pod.GetPod()
		curEndpoint := net.JoinHostPort(curPod.Address, strconv.Itoa(targetPort))
		targetPods = append(targetPods, curPod)
		targetEndpoints = append(targetEndpoints, curEndpoint)
	}

	multiEndpointString := strings.Join(targetEndpoints, ",")
	logger.V(logutil.VERBOSE).Info("Request handled", "objectiveKey", reqCtx.ObjectiveKey, "incomingModelName", reqCtx.IncomingModelName, "targetModel", reqCtx.TargetModelName, "endpoint", multiEndpointString)

	reqCtx.TargetPod = targetPods[0]
	reqCtx.TargetEndpoint = multiEndpointString

	reqCtx.LastSeenMetrics = result.ProfileResults[result.PrimaryProfileName].TargetPod.GetMetrics()
	reqCtx.SchedulingResult = result
	d.runPreRequestPlugins(ctx, reqCtx.SchedulingRequest, result, int(pool.Spec.TargetPortNumber))
	return reqCtx, nil
}

func (d *Director) toSchedulerPodMetrics(pods []backendmetrics.PodMetrics) []schedulingtypes.Pod {
	pm := make([]schedulingtypes.Pod, len(pods))
	for i, pod := range pods {
		pm[i] = &schedulingtypes.PodMetrics{Pod: pod.GetPod().Clone(), MetricsState: pod.GetMetrics().Clone()}
	}

	return pm
}

// HandleResponseHeaders is called when the first chunk of the response arrives.
func (d *Director) HandleResponseHeaders(ctx context.Context, reqCtx *handlers.RequestContext) (*handlers.RequestContext, error) {
	logger := log.FromContext(ctx).WithValues("stage", "headers")
	logger.V(logutil.DEBUG).Info("Entering HandleResponseHeaders")

	response := &Response{
		RequestId: reqCtx.Request.Headers[requtil.RequestIdHeaderKey],
		Headers:   reqCtx.Response.Headers,
	}
	d.runPostResponsePlugins(ctx, reqCtx.SchedulingRequest, response, reqCtx.TargetPod)

	if d.latencyPredictor == nil {
		logger.V(logutil.DEBUG).Info("No latency predictor configured; skipping header prediction")
		return reqCtx, nil
	}
	if reqCtx.SchedulingResult == nil {
		logger.V(logutil.DEBUG).Info("No scheduling result; skipping header prediction")
		return reqCtx, nil
	}

	pr, ok := reqCtx.SchedulingResult.ProfileResults[reqCtx.SchedulingResult.PrimaryProfileName]
	if !ok || pr.TargetPods[0] == nil {
		logger.V(logutil.DEBUG).Info("No target pod metrics; skipping header prediction", "primaryProfile", reqCtx.SchedulingResult.PrimaryProfileName)
		return reqCtx, nil
	}

	// Refresh metrics
	reqCtx.LastSeenMetrics = pr.TargetPods[0].GetMetrics().Clone()
	logger.V(logutil.DEBUG).Info("Refreshed LastSeenMetrics at header",
		"KVCache%", reqCtx.LastSeenMetrics.KVCacheUsagePercent,
		"Waiting", reqCtx.LastSeenMetrics.WaitingQueueSize,
		"Running", reqCtx.LastSeenMetrics.RunningQueueSize,
	)

	// Build prediction request for TTFT
	predictionReq := latencypredictor.PredictionRequest{
		KVCachePercentage:  reqCtx.LastSeenMetrics.KVCacheUsagePercent,
		InputTokenLength:   len(splitWords(reqCtx.Prompt)),
		NumRequestWaiting:  reqCtx.LastSeenMetrics.WaitingQueueSize,
		NumRequestRunning:  reqCtx.LastSeenMetrics.RunningQueueSize,
		NumTokensGenerated: 0, // TTFT is for the first token
	}
	logger.V(logutil.DEBUG).Info("Header prediction request built", "req", predictionReq)

	// Always predict TTFT (not sampled since it's critical for scheduling decisions)
	if prediction, err := d.makePredictionSafely(ctx, predictionReq, "TTFT"); err != nil {
		logger.V(logutil.DEBUG).Error(err, "TTFT prediction failed")
		reqCtx.PredictedTTFT = 0 // Default to 0 on error
	} else {
		reqCtx.PredictedTTFT = prediction
		logger.V(logutil.DEBUG).Info("Predicted TTFT at header stage",
			"predicted_ttft_ms", prediction)
	}

	logger.V(logutil.DEBUG).Info("Exiting HandleResponseHeaders")
	return reqCtx, nil
}

func (d *Director) HandleResponseBodyChunk(ctx context.Context, reqCtx *handlers.RequestContext) error {
	logger := log.FromContext(ctx).WithValues("stage", "bodyChunk")
	logger.V(logutil.DEBUG).Info("Entering HandleResponseBodyChunk")

	if d.latencyPredictor == nil || reqCtx.SchedulingResult == nil {
		logger.V(logutil.DEBUG).Info("Skipping body-chunk logic; predictor or scheduling missing")
		return nil
	}

	pr, ok := reqCtx.SchedulingResult.ProfileResults[reqCtx.SchedulingResult.PrimaryProfileName]
	if !ok || pr.TargetPods[0] == nil {
		logger.V(logutil.DEBUG).Info("Skipping body-chunk logic; no valid target pod")
		return nil
	}

	now := time.Now()

	// Initialize per-request sampler on first call
	if reqCtx.TokenSampler == nil {
		requestID := reqCtx.Request.Headers[requtil.RequestIdHeaderKey]
		reqCtx.TokenSampler = requtil.NewTokenSampler(requestID, defaultSamplingMean, maxSampledTokens)
		logger.V(logutil.DEBUG).Info("Initialized per-request token sampler for predictions",
			"first_prediction_token", reqCtx.TokenSampler.GetNextSampleToken(),
			"request_id", requestID)
	}

	// Determine if this is the first token
	isFirstToken := reqCtx.TTFT == 0

	if isFirstToken {
		// Calculate and record TTFT
		reqCtx.TTFT = float64(now.Sub(reqCtx.RequestReceivedTimestamp).Milliseconds())
		reqCtx.GeneratedTokenCount = 1

		logger.V(logutil.DEBUG).Info("First token received", "ttft_ms", reqCtx.TTFT)

		// ALWAYS add TTFT training data (no sampling for training)
		entry := latencypredictor.TrainingEntry{
			KVCachePercentage:  reqCtx.LastSeenMetrics.KVCacheUsagePercent,
			InputTokenLength:   len(splitWords(reqCtx.Prompt)),
			ActualTTFT:         reqCtx.TTFT,
			ActualTPOT:         0, // Not applicable for TTFT
			Timestamp:          now,
			NumRequestWaiting:  reqCtx.LastSeenMetrics.WaitingQueueSize,
			NumRequestRunning:  reqCtx.LastSeenMetrics.RunningQueueSize,
			NumTokensGenerated: 0, // TTFT is for the first token
		}

		if err := d.latencyPredictor.AddTrainingDataBulk([]latencypredictor.TrainingEntry{entry}); err != nil {
			logger.V(logutil.DEBUG).Error(err, "Failed to add TTFT training sample")
		} else {
			logger.V(logutil.DEBUG).Info("Successfully added TTFT training sample")
		}

		// ALWAYS predict the first TPOT using current metrics state
		// This predicts what the latency will be for the NEXT token (token 2)
		firstTPOTPredictionReq := latencypredictor.PredictionRequest{
			KVCachePercentage:  reqCtx.LastSeenMetrics.KVCacheUsagePercent,
			InputTokenLength:   len(splitWords(reqCtx.Prompt)),
			NumRequestWaiting:  reqCtx.LastSeenMetrics.WaitingQueueSize,
			NumRequestRunning:  reqCtx.LastSeenMetrics.RunningQueueSize,
			NumTokensGenerated: reqCtx.GeneratedTokenCount, // Currently 1, predicting for token 2
		}

		if prediction, err := d.makePredictionSafely(ctx, firstTPOTPredictionReq, "TPOT"); err != nil {
			logger.V(logutil.DEBUG).Error(err, "First TPOT prediction failed")
			reqCtx.PredictedTPOTObservations = append(reqCtx.PredictedTPOTObservations, 0)
			// Update average with 0 prediction
			reqCtx.AvgPredictedTPOT = calculateRunningAverage(reqCtx.AvgPredictedTPOT, 0, len(reqCtx.PredictedTPOTObservations))
		} else {
			reqCtx.PredictedTPOTObservations = append(reqCtx.PredictedTPOTObservations, prediction)
			reqCtx.AvgPredictedTPOT = calculateRunningAverage(reqCtx.AvgPredictedTPOT, prediction, len(reqCtx.PredictedTPOTObservations))
			logger.V(logutil.DEBUG).Info("Predicted first TPOT based on current metrics",
				"predicted_first_tpot_ms", prediction,
				"kv_cache_percent", reqCtx.LastSeenMetrics.KVCacheUsagePercent,
				"waiting_queue", reqCtx.LastSeenMetrics.WaitingQueueSize,
				"running_queue", reqCtx.LastSeenMetrics.RunningQueueSize,
			)
		}

	} else {
		// Calculate inter-token latency (TPOT)
		interTokenLatency := float64(now.Sub(reqCtx.LastTokenTimestamp).Milliseconds())
		reqCtx.GeneratedTokenCount++

		//log the inter-token latency for predicted samples
		if reqCtx.GeneratedTokenCount == 2 || reqCtx.TokenSampler.ShouldPredict(reqCtx.GeneratedTokenCount) { //tricky logic, since next sample token is always +1 from current token
			reqCtx.TPOTObservations = append(reqCtx.TPOTObservations, interTokenLatency)
			reqCtx.AvgTPOT = calculateRunningAverage(reqCtx.AvgTPOT, interTokenLatency, len(reqCtx.TPOTObservations))
		}

		// ALWAYS record actual TPOT for training (store ALL observations)

		logger.V(logutil.DEBUG).Info("Inter-token latency measured",
			"latency_ms", interTokenLatency,
			"token_count", reqCtx.GeneratedTokenCount,
			"total_sampled_observations", len(reqCtx.TPOTObservations),
			"next_prediction_token", reqCtx.TokenSampler.GetNextSampleToken(),
		)

		// ALWAYS add training data (every token contributes to learning)
		trainingEntry := latencypredictor.TrainingEntry{
			KVCachePercentage:  reqCtx.LastSeenMetrics.KVCacheUsagePercent,
			InputTokenLength:   len(splitWords(reqCtx.Prompt)),
			ActualTTFT:         0, // Not applicable for TPOT
			ActualTPOT:         interTokenLatency,
			Timestamp:          now,
			NumRequestWaiting:  reqCtx.LastSeenMetrics.WaitingQueueSize,
			NumRequestRunning:  reqCtx.LastSeenMetrics.RunningQueueSize,
			NumTokensGenerated: reqCtx.GeneratedTokenCount - 1, // Current token count
		}

		if err := d.latencyPredictor.AddTrainingDataBulk([]latencypredictor.TrainingEntry{trainingEntry}); err != nil {
			logger.V(logutil.DEBUG).Error(err, "Failed to add TPOT training sample")
		} else {
			logger.V(logutil.DEBUG).Info("Successfully added TPOT training sample",
				"token_count", reqCtx.GeneratedTokenCount,
				"total_predicting_samples", len(reqCtx.TPOTObservations))
		}

		// Only make predictions for SAMPLED tokens (to reduce overhead)
		if reqCtx.TokenSampler.ShouldPredict(reqCtx.GeneratedTokenCount) {
			logger.V(logutil.DEBUG).Info("Making TPOT prediction for sampled token",
				"token_count", reqCtx.GeneratedTokenCount,
				"prediction_number", reqCtx.TokenSampler.GetSampleCount()+1,
			)

			// Make TPOT prediction for next sampled token
			predictionReq := latencypredictor.PredictionRequest{
				KVCachePercentage:  reqCtx.LastSeenMetrics.KVCacheUsagePercent,
				InputTokenLength:   len(splitWords(reqCtx.Prompt)),
				NumRequestWaiting:  reqCtx.LastSeenMetrics.WaitingQueueSize,
				NumRequestRunning:  reqCtx.LastSeenMetrics.RunningQueueSize,
				NumTokensGenerated: reqCtx.GeneratedTokenCount, // Current token count
			}

			if prediction, err := d.makePredictionSafely(ctx, predictionReq, "TPOT"); err != nil {
				logger.V(logutil.DEBUG).Error(err, "TPOT prediction failed", "token", reqCtx.GeneratedTokenCount)
				reqCtx.PredictedTPOTObservations = append(reqCtx.PredictedTPOTObservations, 0)
				// Update average with 0 prediction
				reqCtx.AvgPredictedTPOT = calculateRunningAverage(reqCtx.AvgPredictedTPOT, 0, len(reqCtx.PredictedTPOTObservations))
			} else {
				reqCtx.PredictedTPOTObservations = append(reqCtx.PredictedTPOTObservations, prediction)
				reqCtx.AvgPredictedTPOT = calculateRunningAverage(reqCtx.AvgPredictedTPOT, prediction, len(reqCtx.PredictedTPOTObservations))
				logger.V(logutil.DEBUG).Info("Predicted TPOT for sampled token",
					"predicted_tpot_ms", prediction,
					"token", reqCtx.GeneratedTokenCount,
					"avg_tpot_ms", reqCtx.AvgTPOT,
					"sampled_tokens", len(reqCtx.PredictedTPOTObservations),
				)
			}

			// Record the prediction and calculate next sample token
			reqCtx.TokenSampler.RecordPrediction(reqCtx.GeneratedTokenCount)

			if reqCtx.TokenSampler.GetSampleCount() < maxSampledTokens {
				logger.V(logutil.DEBUG).Info("Scheduled next prediction",
					"current_token", reqCtx.GeneratedTokenCount,
					"next_prediction_token", reqCtx.TokenSampler.GetNextSampleToken(),
				)
			} else {
				logger.V(logutil.DEBUG).Info("Reached maximum predictions, no more predictions",
					"max_predictions", maxSampledTokens)
			}
		} else {
			logger.V(logutil.DEBUG).Info("Skipping prediction for this token (training still performed)",
				"token_count", reqCtx.GeneratedTokenCount,
				"next_prediction_token", reqCtx.TokenSampler.GetNextSampleToken(),
				"predictions_made", reqCtx.TokenSampler.GetSampleCount(),
			)
		}

	}
	// Always update timestamp for next calculation
	reqCtx.LastTokenTimestamp = now
	// Refresh metrics
	reqCtx.LastSeenMetrics = pr.TargetPods[0].GetMetrics().Clone()
	logger.V(logutil.DEBUG).Info("Refreshed LastSeenMetrics at body chunk",
		"KVCache%", reqCtx.LastSeenMetrics.KVCacheUsagePercent,
		"Waiting", reqCtx.LastSeenMetrics.WaitingQueueSize,
		"Running", reqCtx.LastSeenMetrics.RunningQueueSize,
	)

	logger.V(logutil.DEBUG).Info("Exiting HandleResponseBodyChunk")
	return nil
}

func (d *Director) makePredictionSafely(ctx context.Context, req latencypredictor.PredictionRequest, predictionType string) (float64, error) {
	// Validate input
	if req.InputTokenLength < 0 {
		return 0, fmt.Errorf("invalid prediction request: negative token counts")
	}

	start := time.Now()
	prediction, err := d.latencyPredictor.Predict(ctx, req)
	duration := time.Since(start)

	if err != nil {
		log.FromContext(ctx).V(logutil.DEBUG).Error(err,
			"Prediction failed",
			"type", predictionType,
			"duration", duration,
		)
		return 0, err
	}

	if prediction == nil {
		return 0, fmt.Errorf("predictor returned nil prediction")
	}

	var result float64
	switch predictionType {
	case "TTFT":
		result = prediction.TTFT
	case "TPOT":
		result = prediction.TPOT
	default:
		return 0, fmt.Errorf("unknown prediction type: %s", predictionType)
	}

	// Validate result
	if result < 0 {
		log.FromContext(ctx).V(logutil.DEBUG).Info("Negative prediction received",
			"type", predictionType,
			"value", result,
		)
		return 0, nil // Return 0 for negative predictions
	}

	log.FromContext(ctx).V(logutil.DEBUG).Info("Prediction successful",
		"type", predictionType,
		"value", result,
		"duration", duration,
	)

	return result, nil
}

// HandleResponseTrailers calculates final aggregate metrics and adds them to response trailers.
func (d *Director) HandleResponseTrailers(ctx context.Context, reqCtx *handlers.RequestContext) (*handlers.RequestContext, error) {
	logger := log.FromContext(ctx).WithValues("stage", "trailers")
	logger.V(logutil.DEBUG).Info("Entering HandleResponseTrailers")
	return reqCtx, nil
}

func (d *Director) GetRandomPod() *backend.Pod {
	pods := d.datastore.PodList(backendmetrics.AllPodsPredicate)
	if len(pods) == 0 {
		return nil
	}
	number := rand.Intn(len(pods))
	pod := pods[number]
	return pod.GetPod()
}

func RandomWeightedDraw(logger logr.Logger, model *v1alpha2.InferenceModel, seed int64) string {
	// TODO: after we are down to 1 server implementation, make these methods a part of the struct
	// and handle random seeding on the struct.
	source := rand.NewSource(rand.Int63())
	if seed > 0 {
		source = rand.NewSource(seed)
	}
	r := rand.New(source)

	// all the weight values are nil, then we should return random model name
	if model.Spec.TargetModels[0].Weight == nil {
		index := r.Int31n(int32(len(model.Spec.TargetModels)))
		return model.Spec.TargetModels[index].Name
	}

	var weights int32
	for _, model := range model.Spec.TargetModels {
		weights += *model.Weight
	}
	logger.V(logutil.DEBUG).Info("Weights for model computed", "model", model.Name, "weights", weights)
	randomVal := r.Int31n(weights)
	// TODO: optimize this without using loop
	for _, model := range model.Spec.TargetModels {
		if randomVal < *model.Weight {
			return model.Name
		}
		randomVal -= *model.Weight
	}
	return ""
}

func (d *Director) runPreRequestPlugins(ctx context.Context, request *schedulingtypes.LLMRequest, schedulingResult *schedulingtypes.SchedulingResult,
	targetPort int,
) {
	for _, plugin := range d.preRequestPlugins {
		loggerDebug.Info("Running pre-request plugin", "plugin", plugin.TypedName())
		before := time.Now()
		plugin.PreRequest(ctx, request, schedulingResult, targetPort)
		metrics.RecordPluginProcessingLatency(PreRequestExtensionPoint, plugin.TypedName().Type, plugin.TypedName().Name, time.Since(before))
		loggerDebug.Info("Completed running pre-request plugin successfully", "plugin", plugin.TypedName())
	}
}

func (d *Director) runPostResponsePlugins(ctx context.Context, request *schedulingtypes.LLMRequest, response *Response, targetPod *backend.Pod) {
	loggerDebug := log.FromContext(ctx).V(logutil.DEBUG)
	for _, plugin := range d.postResponsePlugins {
		loggerDebug.Info("Running post-response plugin", "plugin", plugin.TypedName())
		before := time.Now()
		plugin.PostResponse(ctx, request, response, targetPod)
		metrics.RecordPluginProcessingLatency(PostResponseExtensionPoint, plugin.TypedName().Type, plugin.TypedName().Name, time.Since(before))
		loggerDebug.Info("Completed running post-response plugin successfully", "plugin", plugin.TypedName())
	}
}

func (d *Director) IsPredictorAvailable() bool {
	return d.latencyPredictor != nil
}
