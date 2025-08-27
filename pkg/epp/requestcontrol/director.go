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

	"sigs.k8s.io/controller-runtime/pkg/log"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/handlers"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"
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
)

const (
	// Poisson sampling parameters for predictions
	defaultSamplingMean = 100 // Mean interval between prediction samples (tokens)
	maxSampledTokens    = 20  // Maximum number of prediction samples per request
)

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

// parseFloatHeader retrieves a header by name, parses it as a float64,
// and returns the value or an error if the header is missing or invalid.
func parseFloatHeader(reqCtx *handlers.RequestContext, headerName string) (float64, bool, error) {
	// 1. Get header value from the map
	headerValue, ok := reqCtx.Request.Headers[headerName]
	if !ok {
		return 0, false, nil // Header not found, return 0 and false
	}

	// 2. Parse the header value to a float64
	parsedFloat, err := strconv.ParseFloat(headerValue, 64)
	if err != nil {
		return 0, false, errutil.Error{
			Code: errutil.BadRequest,
			Msg:  fmt.Sprintf("%s must be a float", headerName),
		}
	}

	// 3. Return the successfully parsed value
	return parsedFloat, true, nil
}

// parseFloatHeader retrieves a header by name, parses it as a bool,
// and returns the value or an error if the header is missing or invalid.
func parseBoolHeader(reqCtx *handlers.RequestContext, headerName string) (bool, error) {
	// 1. Get header value from the map
	headerValue, ok := reqCtx.Request.Headers[headerName]
	if !ok {
		return false, nil // Header not found, return 0 and false
	}

	// 2. Parse the header value to a bool
	parsedBool, err := strconv.ParseBool(headerValue)
	if err != nil {
		return false, errutil.Error{
			Code: errutil.BadRequest,
			Msg:  fmt.Sprintf("%s must be a bool", headerName),
		}
	}

	// 3. Return the successfully parsed value
	return parsedBool, nil
}

// Scheduler defines the interface required by the Director for scheduling.
type Scheduler interface {
	Schedule(ctx context.Context, request *schedulingtypes.LLMRequest, candidatePods []schedulingtypes.Pod) (result *schedulingtypes.SchedulingResult, err error)
}

// SaturationDetector provides a signal indicating whether the backends are considered saturated.
type SaturationDetector interface {
	IsSaturated(ctx context.Context, candidatePods []backendmetrics.PodMetrics) bool
}

// NewDirectorWithConfig creates a new Director instance with all dependencies.
func NewDirectorWithConfig(datastore datastore.Datastore, scheduler Scheduler, saturationDetector SaturationDetector, config *Config) *Director {
	return &Director{
		datastore:                   datastore,
		scheduler:                   scheduler,
		saturationDetector:          saturationDetector,
		preRequestPlugins:           config.preRequestPlugins,
		postResponsePlugins:         config.postResponsePlugins,
		postResponseChunkPlugins:    config.postResponseChunkPlugins,
		postResponseCompletePlugins: config.postResponseCompletePlugins,
	}
}

// Director orchestrates the request handling flow, including scheduling.
type Director struct {
	datastore                   datastore.Datastore
	scheduler                   Scheduler
	saturationDetector          SaturationDetector
	preRequestPlugins           []PreRequest
	postResponsePlugins         []PostResponse
	postResponseChunkPlugins    []PostResponseChunk
	postResponseCompletePlugins []PostResponseComplete
	// we just need a pointer to an int variable since priority is a pointer in InferenceObjective
	// no need to set this in the constructor, since the value we want is the default int val
	// and value types cannot be nil
	defaultPriority int
}

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

	// get request slos
	// Get Request SLOs from request header
	ttftSLO, _, err := parseFloatHeader(reqCtx, "ttft_slo")
	if err != nil {
		return reqCtx, errutil.Error{Code: errutil.BadRequest, Msg: fmt.Sprintf("ttft_slo must be a float: %v", err)}
	}
	avgTPOTSLO, _, err := parseFloatHeader(reqCtx, "avg_tpot_slo")
	if err != nil {
		return reqCtx, errutil.Error{Code: errutil.BadRequest, Msg: fmt.Sprintf("avg_tpot_slo must be a float: %v", err)}
	}
	predictionBasedScheduling, err := parseBoolHeader(reqCtx, "prediction_based_scheduling")
	if err != nil {
		return reqCtx, errutil.Error{Code: errutil.BadRequest, Msg: fmt.Sprintf("prediction_based_scheduling must be a bool: %v", err)}
	}

	// Prepare LLMRequest (needed for both saturation detection and Scheduler)
	reqCtx.SchedulingRequest = &schedulingtypes.LLMRequest{
		RequestId:                reqCtx.Request.Headers[requtil.RequestIdHeaderKey],
		TargetModel:              reqCtx.TargetModelName,
		Prompt:                   prompt,
		Headers:                  reqCtx.Request.Headers,
		TTFTSLO:                  ttftSLO,
		AvgTPOTSLO:               avgTPOTSLO,
		PredictorBasedScheduling: predictionBasedScheduling,
	}

	logger = logger.WithValues("objectiveKey", reqCtx.ObjectiveKey, "incomingModelName", reqCtx.IncomingModelName, "targetModelName", reqCtx.TargetModelName, "priority", infObjective.Spec.Priority)

	ctx = log.IntoContext(ctx, logger)
	logger.V(logutil.DEBUG).Info("LLM request assembled")

	// Get candidate pods for scheduling
	candidatePods := d.getCandidatePodsForScheduling(ctx, reqCtx.Request.Metadata)
	if len(candidatePods) == 0 {
		return reqCtx, errutil.Error{Code: errutil.ServiceUnavailable, Msg: "failed to find candidate pods for serving the request"}
	}
	result, err := d.scheduler.Schedule(ctx, reqCtx.SchedulingRequest, candidatePods)
	if err != nil {
		return reqCtx, errutil.Error{Code: errutil.InferencePoolResourceExhausted, Msg: fmt.Errorf("failed to find target pod: %w", err).Error()}
	}

	// --- 4. Prepare Request (Populates RequestContext and call PreRequest plugins) ---
	// Insert target endpoint to instruct Envoy to route requests to the specified target pod and attach the port number.
	// Invoke PreRequest registered plugins.
	reqCtx, err = d.prepareRequest(ctx, reqCtx, result)
	if err != nil {
		return reqCtx, err
	}

	return reqCtx, nil
}

// admitRequest handles admission control to decide whether or not to accept the request
// based on the request priority and system saturation state.
func (d *Director) admitRequest(ctx context.Context, requestPriority int, fairnessID string) error {
	logger := log.FromContext(ctx)

	logger.V(logutil.TRACE).Info("Entering Flow Control", "priority", requestPriority, "fairnessID", fairnessID)

	// This will be removed in favor of a more robust implementation (Flow Control) in the very near future.
	// TODO: Make this a configurable value.
	// Tracking issue https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/1347
	if requestPriority >= 0 {
		logger.V(logutil.TRACE).Info("Non-sheddable request bypassing saturation check.")
		return nil
	}

	if d.saturationDetector.IsSaturated(ctx) { // Assuming non-nil Saturation Detector
		return errutil.Error{
			Code: errutil.InferencePoolResourceExhausted,
			Msg:  "system saturated, sheddable request dropped",
		}
	}

	return nil
}

// getCandidatePodsForScheduling gets the list of relevant endpoints for the scheduling cycle from the datastore.
// according to EPP protocol, if "x-gateway-destination-endpoint-subset" is set on the request metadata and specifies
// a subset of endpoints, only these endpoints will be considered as candidates for the scheduler.
// Snapshot pod metrics from the datastore to:
// 1. Reduce concurrent access to the datastore.
// 2. Ensure consistent data during the scheduling operation of a request between all scheduling cycles.
func (d *Director) getCandidatePodsForScheduling(ctx context.Context, requestMetadata map[string]any) []schedulingtypes.Pod {
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

	return d.toSchedulerPodMetrics(podFitleredList)
}

// prepareRequest populates the RequestContext and calls the registered PreRequest plugins
// for allowing plugging customized logic based on the scheduling result.
func (d *Director) prepareRequest(ctx context.Context, reqCtx *handlers.RequestContext, result *schedulingtypes.SchedulingResult) (*handlers.RequestContext, error) {
	logger := log.FromContext(ctx)
	if result == nil || len(result.ProfileResults) == 0 {
		return reqCtx, errutil.Error{Code: errutil.Internal, Msg: "results must be greater than zero"}
	}
	// primary profile is used to set destination
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

	d.runPreRequestPlugins(ctx, reqCtx.SchedulingRequest, result, targetPort)
	reqCtx.SchedulingResult = result
	reqCtx.LastSeenMetrics = make(map[string]*backendmetrics.MetricsState)
	RefreshLastSeenMetrics(ctx, reqCtx)

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
func (d *Director) HandleResponse(ctx context.Context, reqCtx *handlers.RequestContext) (*handlers.RequestContext, error) {
	logger := log.FromContext(ctx).WithValues("stage", "headers")
	logger.V(logutil.DEBUG).Info("Entering HandleResponseHeaders")

	d.runPostResponsePlugins(ctx, reqCtx)

	logger.V(logutil.DEBUG).Info("Exiting HandleResponseHeaders")
	return reqCtx, nil
}

func (d *Director) HandleResponseBodyChunk(ctx context.Context, reqCtx *handlers.RequestContext) error {
	logger := log.FromContext(ctx).WithValues("stage", "bodyChunk")
	logger.V(logutil.TRACE).Info("Entering HandleResponseBodyChunk")

	d.runPostResponseChunkPlugins(ctx, reqCtx)

	logger.V(logutil.TRACE).Info("Exiting HandleResponseBodyChunk")
	return nil
}

// HandleResponseBodyComplete is called when the response body is fully received.
// It runs the PostResponseComplete plugins.
func (d *Director) HandleResponseBodyComplete(ctx context.Context, reqCtx *handlers.RequestContext) error {
	logger := log.FromContext(ctx).WithValues("stage", "bodyChunk")
	logger.V(logutil.DEBUG).Info("Entering HandleResponseBodyComplete")

	d.runPostResponseCompletePlugins(ctx, reqCtx)

	logger.V(logutil.DEBUG).Info("Exiting HandleResponseBodyComplete")
	return nil
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

func (d *Director) runPreRequestPlugins(ctx context.Context, request *schedulingtypes.LLMRequest,
	schedulingResult *schedulingtypes.SchedulingResult, targetPort int) {
	loggerDebug := log.FromContext(ctx).V(logutil.DEBUG)
	for _, plugin := range d.preRequestPlugins {
		loggerDebug.Info("Running pre-request plugin", "plugin", plugin.TypedName())
		before := time.Now()
		plugin.PreRequest(ctx, request, schedulingResult, targetPort)
		metrics.RecordPluginProcessingLatency(PreRequestExtensionPoint, plugin.TypedName().Type, plugin.TypedName().Name, time.Since(before))
		loggerDebug.Info("Completed running pre-request plugin successfully", "plugin", plugin.TypedName())
	}
}

func (d *Director) runPostResponsePlugins(ctx context.Context, reqCtx *handlers.RequestContext) {
	loggerDebug := log.FromContext(ctx).V(logutil.DEBUG)
	for _, plugin := range d.postResponsePlugins {
		loggerDebug.Info("Running post-response plugin", "plugin", plugin.TypedName())
		before := time.Now()
		plugin.PostResponse(ctx, reqCtx)
		metrics.RecordPluginProcessingLatency(PostResponseExtensionPoint, plugin.TypedName().Type, plugin.TypedName().Name, time.Since(before))
		loggerDebug.Info("Completed running post-response plugin successfully", "plugin", plugin.TypedName())
	}
}

func (d *Director) runPostResponseChunkPlugins(ctx context.Context, reqCtx *handlers.RequestContext) {
	loggerTrace := log.FromContext(ctx).V(logutil.TRACE)
	for _, plugin := range d.postResponseChunkPlugins {
		loggerTrace.Info("Running post-response chunk plugin", "plugin", plugin.TypedName().Type)
		before := time.Now()
		plugin.PostResponseChunk(ctx, reqCtx)
		metrics.RecordPluginProcessingLatency(PostResponseChunkExtensionPoint, plugin.TypedName().Type, plugin.TypedName().Name, time.Since(before))
	}
}

func (d *Director) runPostResponseCompletePlugins(ctx context.Context, reqCtx *handlers.RequestContext) {
	loggerDebug := log.FromContext(ctx).V(logutil.DEBUG)
	for _, plugin := range d.postResponseCompletePlugins {
		loggerDebug.Info("Running post-response complete plugin", "plugin", plugin.TypedName().Type)
		before := time.Now()
		plugin.PostResponseComplete(ctx, reqCtx)
		metrics.RecordPluginProcessingLatency(PostResponseCompleteExtensionPoint, plugin.TypedName().Type, plugin.TypedName().Name, time.Since(before))
		loggerDebug.Info("Completed running post-response complete plugin successfully", "plugin", plugin.TypedName())
	}
}
