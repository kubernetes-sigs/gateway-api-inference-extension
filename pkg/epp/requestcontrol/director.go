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

	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/handlers"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
)

const (
	subsetHintNamespace = "envoy.lb.subset_hint"
	subsetHintKey       = "x-gateway-destination-endpoint-subset"
)

// Scheduler defines the interface required by the Director for scheduling.
type Scheduler interface {
	Schedule(ctx context.Context, request *schedulingtypes.LLMRequest, candidatePods []schedulingtypes.Pod) (result *schedulingtypes.SchedulingResult, err error)
}

// SaturationDetector provides a signal indicating whether the backends are considered saturated.
type SaturationDetector interface {
	IsSaturated(ctx context.Context) bool
}

// NewDirectorWithConfig creates a new Director instance with all dependencies.
func NewDirectorWithConfig(datastore datastore.Datastore, scheduler Scheduler, saturationDetector SaturationDetector, config *Config) *Director {
	return &Director{
		datastore:           datastore,
		scheduler:           scheduler,
		saturationDetector:  saturationDetector,
		preRequestPlugins:   config.preRequestPlugins,
		postResponsePlugins: config.postResponsePlugins,
	}
}

// Director orchestrates the request handling flow, including scheduling.
type Director struct {
	datastore           datastore.Datastore
	scheduler           Scheduler
	saturationDetector  SaturationDetector
	preRequestPlugins   []PreRequest
	postResponsePlugins []PostResponse
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

	// --- 1. Parse Request, Resolve Target Models, and Determine Parameters ---
	var ok bool
	requestBodyMap := reqCtx.Request.Body
	reqCtx.Model, ok = requestBodyMap["model"].(string)
	if !ok {
		return reqCtx, errutil.Error{Code: errutil.BadRequest, Msg: "model not found in request body"}
	}
	prompt, err := requtil.ExtractPromptFromRequestBody(requestBodyMap)
	if err != nil {
		return reqCtx, err
	}

	modelObj := d.datastore.ObjectiveGet(reqCtx.Model)
	if modelObj == nil {
		logger.Info("No associated InferenceObjective found, using default", "model", reqCtx.Model)
		sheddable := v1alpha2.Sheddable
		modelObj = &v1alpha2.InferenceObjective{
			Spec: v1alpha2.InferenceObjectiveSpec{
				ModelName:   reqCtx.Model,
				Criticality: &sheddable,
			},
		}
	}

	reqCtx.ResolvedTargetModel = reqCtx.Model
	if len(modelObj.Spec.TargetModels) > 0 {
		reqCtx.ResolvedTargetModel = RandomWeightedDraw(logger, modelObj, 0)
		if reqCtx.ResolvedTargetModel == "" {
			return reqCtx, errutil.Error{Code: errutil.BadConfiguration, Msg: fmt.Sprintf("error getting target model name for model %v", modelObj.Name)}
		}
		reqCtx.Request.Body["model"] = reqCtx.ResolvedTargetModel // Update target model in the body.
	}

	requestCriticality := v1alpha2.Standard
	if modelObj.Spec.Criticality != nil {
		requestCriticality = *modelObj.Spec.Criticality
	}

	// Prepare LLMRequest (needed for both saturation detection and Scheduler)
	reqCtx.SchedulingRequest = &schedulingtypes.LLMRequest{
		RequestId:   reqCtx.Request.Headers[requtil.RequestIdHeaderKey],
		TargetModel: reqCtx.ResolvedTargetModel,
		Prompt:      prompt,
		Headers:     reqCtx.Request.Headers,
	}

	logger = logger.WithValues("model", reqCtx.Model, "resolvedTargetModel", reqCtx.ResolvedTargetModel, "criticality", requestCriticality)

	ctx = log.IntoContext(ctx, logger)
	logger.V(logutil.DEBUG).Info("LLM request assembled")

	// --- 2. Admission Control check --
	if err := d.admitRequest(ctx, requestCriticality); err != nil {
		return reqCtx, err
	}

	// --- 3. Call Scheduler (with the relevant candidate pods) ---
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
// based on the request criticality and system saturation state.
func (d *Director) admitRequest(ctx context.Context, requestCriticality v1alpha2.Criticality) error {
	logger := log.FromContext(ctx)

	if requestCriticality == v1alpha2.Critical {
		logger.V(logutil.DEBUG).Info("Critical request bypassing saturation check.")
		return nil
	}

	logger.V(logutil.DEBUG).Info("Performing saturation check for non-critical request.")
	if d.saturationDetector.IsSaturated(ctx) { // Assuming non-nil Saturation Detector
		return errutil.Error{
			Code: errutil.InferencePoolResourceExhausted,
			Msg:  "system saturated, non-critical request dropped",
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

	subsetMap, found := requestMetadata[subsetHintNamespace].(map[string]any)
	if !found {
		return d.toSchedulerPodMetrics(d.datastore.PodList(backendmetrics.AllPodPredicate))
	}

	// Check if endpoint key is present in the subset map and ensure there is at least one value
	endpointSubsetList, found := subsetMap[subsetHintKey].([]any)
	if !found {
		return d.toSchedulerPodMetrics(d.datastore.PodList(backendmetrics.AllPodPredicate))
	} else if len(endpointSubsetList) == 0 {
		loggerTrace.Info("found empty subset filter in request metadata, filtering all pods")
		return []schedulingtypes.Pod{}
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
	podFitleredList := d.datastore.PodList(func(pm backendmetrics.PodMetrics) bool {
		podTotalCount++
		if _, found := endpoints[pm.GetPod().Address]; found {
			return true
		}
		return false
	})

	loggerTrace.Info("filtered candidate pods by subset filtering", "podTotalCount", podTotalCount, "filteredCount", len(podFitleredList))

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
	targetPort := int(pool.Spec.TargetPortNumber)
	targetEndpoints := []string{}

	for _, pod := range result.ProfileResults[result.PrimaryProfileName].TargetPods {
		curPod := pod.GetPod()
		curEndpoint := net.JoinHostPort(curPod.Address, strconv.Itoa(targetPort))
		targetPods = append(targetPods, curPod)
		targetEndpoints = append(targetEndpoints, curEndpoint)
	}

	multiEndpointString := strings.Join(targetEndpoints, ",")
	logger.V(logutil.DEFAULT).Info("Request handled", "model", reqCtx.Model, "targetModel", reqCtx.ResolvedTargetModel, "endpoint", multiEndpointString)

	reqCtx.TargetPod = targetPods[0]
	reqCtx.TargetEndpoint = multiEndpointString

	d.runPreRequestPlugins(ctx, reqCtx.SchedulingRequest, result, targetPort)

	return reqCtx, nil
}

func (d *Director) toSchedulerPodMetrics(pods []backendmetrics.PodMetrics) []schedulingtypes.Pod {
	pm := make([]schedulingtypes.Pod, len(pods))
	for i, pod := range pods {
		pm[i] = &schedulingtypes.PodMetrics{Pod: pod.GetPod().Clone(), MetricsState: pod.GetMetrics().Clone()}
	}

	return pm
}

func (d *Director) HandleResponse(ctx context.Context, reqCtx *handlers.RequestContext) (*handlers.RequestContext, error) {
	response := &Response{
		RequestId: reqCtx.Request.Headers[requtil.RequestIdHeaderKey],
		Headers:   reqCtx.Response.Headers,
	}

	// TODO: to extend fallback functionality, handle cases where target pod is unavailable
	// https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/1224
	d.runPostResponsePlugins(ctx, reqCtx.SchedulingRequest, response, reqCtx.TargetPod)

	return reqCtx, nil
}

func (d *Director) GetRandomPod() *backend.Pod {
	pods := d.datastore.PodList(backendmetrics.AllPodPredicate)
	if len(pods) == 0 {
		return nil
	}
	number := rand.Intn(len(pods))
	pod := pods[number]
	return pod.GetPod()
}

func RandomWeightedDraw(logger logr.Logger, model *v1alpha2.InferenceObjective, seed int64) string {
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
	logger.V(logutil.TRACE).Info("Weights for model computed", "model", model.Name, "weights", weights)
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
