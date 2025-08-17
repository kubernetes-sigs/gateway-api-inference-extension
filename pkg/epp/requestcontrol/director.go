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

// Scheduler defines the interface required by the Director for scheduling.
type Scheduler interface {
	Schedule(ctx context.Context, request *schedulingtypes.LLMRequest, candidatePods []schedulingtypes.Pod) (result *schedulingtypes.SchedulingResult, err error)
}

// SaturationDetector provides a signal indicating whether the backends are considered saturated.
type SaturationDetector interface {
	IsSaturated(ctx context.Context, candidatePods []backendmetrics.PodMetrics) bool
}

// NewDirectorWithConfig creates a new Director instance with all dependencies.
func NewDirectorWithConfig(datastore Datastore, scheduler Scheduler, saturationDetector SaturationDetector, config *Config) *Director {
	return &Director{
		datastore:           datastore,
		scheduler:           scheduler,
		saturationDetector:  saturationDetector,
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
	preRequestPlugins   []PreRequest
	postResponsePlugins []PostResponse
	// we just need a pointer to an int variable since priority is a pointer in InferenceObjective
	// no need to set this in the constructor, since the value we want is the default int val
	// and value types cannot be nil
	defaultPriority int
}

// HandleRequest orchestrates the request lifecycle.
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

// admitRequest handles admission control to decide whether or not to accept the request
// based on the request priority and system saturation state.
func (d *Director) admitRequest(ctx context.Context, candidatePods []backendmetrics.PodMetrics,
	requestPriority int, fairnessID string) error {
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
	podFitleredList := d.datastore.PodList(func(pm backendmetrics.PodMetrics) bool {
		podTotalCount++
		if _, found := endpoints[pm.GetPod().Address]; found {
			return true
		}
		return false
	})

	loggerTrace.Info("filtered candidate pods by subset filtering", "podTotalCount", podTotalCount, "filteredCount", len(podFitleredList))

	return podFitleredList
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
