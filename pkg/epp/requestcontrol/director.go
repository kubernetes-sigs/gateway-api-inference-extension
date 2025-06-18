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
	"strconv"
	"time"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/handlers"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
)

// Scheduler defines the interface required by the Director for scheduling.
type Scheduler interface {
	Schedule(ctx context.Context, b *schedulingtypes.LLMRequest) (result *schedulingtypes.SchedulingResult, err error)
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
		postResponsePlugins: config.postResponsePlugins,
	}
}

// Director orchestrates the request handling flow, including scheduling.
type Director struct {
	datastore           datastore.Datastore
	scheduler           Scheduler
	saturationDetector  SaturationDetector
	postResponsePlugins []PostResponse
}

// HandleRequest orchestrates the request lifecycle:
//  1. Parses request details.
//  2. Calls PreDispatch for admission control.
//  3. Calls Dispatch (which calls Scheduler) if request is approved.
//  4. Calls PostDispatch to populate RequestContext with results.
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

	// NOTE: The nil checking for the modelObject means that we DO allow passthrough currently.
	// This might be a security risk in the future where adapters not registered in the InferenceModel
	// are able to be requested by using their distinct name.
	modelObj := d.datastore.ModelGet(reqCtx.Model)
	if modelObj == nil {
		return reqCtx, errutil.Error{Code: errutil.BadConfiguration, Msg: fmt.Sprintf("error finding a model object in InferenceModel for input %v", reqCtx.Model)}
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
	logger = logger.WithValues(
		"model", reqCtx.Model,
		"resolvedTargetModel", reqCtx.ResolvedTargetModel,
		"criticality", requestCriticality,
	)
	ctx = log.IntoContext(ctx, logger)
	logger.V(logutil.DEBUG).Info("LLM request assembled")

	// --- 2. Saturation Check ---
	preDispatchErr := d.PreDispatch(ctx, reqCtx, requestCriticality)
	if preDispatchErr != nil {
		return reqCtx, preDispatchErr
	}

	// --- 3. Dispatch (Calls Scheduler) ---
	results, dispatchErr := d.Dispatch(ctx, reqCtx.SchedulingRequest)
	if dispatchErr != nil {
		return reqCtx, dispatchErr
	}

	// --- 4. PostDispatch (Populates RequestContext) ---
	// Insert target endpoint to instruct Envoy to route requests to the specified target pod.
	// Attach the port number.
	reqCtx, postDispatchErr := d.PostDispatch(ctx, reqCtx, results)
	if postDispatchErr != nil {
		return reqCtx, postDispatchErr
	}

	return reqCtx, nil
}

// PreDispatch handles admission control before dispatch.
func (d *Director) PreDispatch(ctx context.Context, reqCtx *handlers.RequestContext, reqCriticality v1alpha2.Criticality) error {
	logger := log.FromContext(ctx)

	if reqCriticality == v1alpha2.Critical {
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

// Dispatch runs one or many scheduling cycles.
func (d *Director) Dispatch(ctx context.Context, llmReq *schedulingtypes.LLMRequest) (*schedulingtypes.SchedulingResult, error) {
	var err error
	res, err := d.scheduler.Schedule(ctx, llmReq)
	if err != nil {
		return nil, errutil.Error{Code: errutil.InferencePoolResourceExhausted, Msg: fmt.Errorf("failed to find target pod: %w", err).Error()}
	}

	return res, nil // TODO handle multi cycle result after defining the PostDispatch extension point
}

// PostDispatch populates the RequestContext based on scheduling results.
func (d *Director) PostDispatch(ctx context.Context, reqCtx *handlers.RequestContext, result *schedulingtypes.SchedulingResult) (*handlers.RequestContext, error) {
	logger := log.FromContext(ctx)
	// currently only get a single result. Will refactor to pluggably implement the PostSchedule
	if result == nil || len(result.ProfileResults) == 0 {
		return reqCtx, errutil.Error{Code: errutil.Internal, Msg: "results must be greater than zero"}
	}
	// primary profile is used to set destination
	targetPod := result.ProfileResults[result.PrimaryProfileName].TargetPod.GetPod()

	pool, err := d.datastore.PoolGet()
	if err != nil {
		return reqCtx, err
	}

	endpoint := targetPod.Address + ":" + strconv.Itoa(int(pool.Spec.TargetPortNumber))
	logger.V(logutil.DEFAULT).Info("Request handled", "model", reqCtx.Model, "targetModel", reqCtx.ResolvedTargetModel, "endpoint", targetPod)

	reqCtx.TargetPod = targetPod
	reqCtx.TargetEndpoint = endpoint

	return reqCtx, nil
}

func (d *Director) HandleResponse(ctx context.Context, reqCtx *handlers.RequestContext) (*handlers.RequestContext, error) {
	response := &Response{
		RequestId: reqCtx.Request.Headers[requtil.RequestIdHeaderKey],
		Headers:   reqCtx.Response.Headers,
	}

	d.runPostResponsePlugins(ctx, reqCtx.SchedulingRequest, response, reqCtx.TargetPod)

	return reqCtx, nil
}

func (d *Director) GetRandomPod() *backend.Pod {
	pods := d.datastore.PodGetAll()
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

func (d *Director) runPostResponsePlugins(ctx context.Context, request *schedulingtypes.LLMRequest, response *Response, targetPod *backend.Pod) {
	for _, plugin := range d.postResponsePlugins {
		log.FromContext(ctx).V(logutil.DEBUG).Info("Running post-response plugin", "plugin", plugin.Name())
		before := time.Now()
		plugin.PostResponse(ctx, request, response, targetPod)
		metrics.RecordRequestControlPluginProcessingLatency(PostResponsePluginType, plugin.Name(), time.Since(before))
	}
}
