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

// Package requestcontrol defines the Director component responsible for
// orchestrating request processing after initial parsing.
package requestcontrol

import (
	"context"
	"fmt"
	"math/rand"
	"strconv"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/handlers"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// Scheduler defines the interface required by the Director for scheduling.
type Scheduler interface {
	Schedule(ctx context.Context, b *schedulingtypes.LLMRequest) (result *schedulingtypes.Result, err error)
}

// SaturationDetector provides a signal indicating whether the backends are
// considered saturated.
type SaturationDetector interface {
	IsSaturated() bool
}

// Director orchestrates the request handling flow, including scheduling.
type Director struct {
	datastore          datastore.Datastore
	scheduler          Scheduler
	saturationDetector SaturationDetector
}

// NewDirector creates a new Director instance with all dependencies.
func NewDirector(ds datastore.Datastore, sched Scheduler, sd SaturationDetector) *Director {
	return &Director{
		datastore:          ds,
		scheduler:          sched,
		saturationDetector: sd,
	}
}

// HandleRequest orchestrates the request lifecycle:
//  1. Parses request details.
//  2. Calls PreDispatch for admission control.
//  3. Calls Dispatch (which calls Scheduler) if request is approved.
//  4. Calls PostDispatch to populate RequestContext with results.
//
// It always returns the requestContext even in the error case, as the request
// context is used in error handling.
func (d *Director) HandleRequest(ctx context.Context, reqCtx *handlers.RequestContext) (*handlers.RequestContext, error) {
	logger := log.FromContext(ctx)

	// --- 1. Parse Request, Resolve Target Models, and Determine Parameters ---
	var ok bool
	requestBodyMap := reqCtx.Request.Body
	reqCtx.Model, ok = requestBodyMap["model"].(string)
	if !ok {
		return reqCtx, errutil.Error{Code: errutil.BadRequest, Msg: "model not found in request body"}
	}
	prompt, ok := requestBodyMap["prompt"].(string)
	if !ok {
		return reqCtx, errutil.Error{Code: errutil.BadRequest, Msg: "prompt not found in request body"}
	}

	// NOTE: The nil checking for the modelObject means that we DO allow
	// passthrough currently.
	// This might be a security risk in the future where adapters not registered
	// in the InferenceModel are able to be requested by using their distinct
	// name.
	modelObj := d.datastore.ModelGet(reqCtx.Model)
	if modelObj == nil {
		logger.V(logutil.DEFAULT).Info("InferenceModel not found in datastore",
			"model", reqCtx.Model)
		return reqCtx, errutil.Error{
			Code: errutil.BadConfiguration,
			Msg:  fmt.Sprintf("InferenceModel %s not found", reqCtx.Model),
		}
	}

	reqCtx.ResolvedTargetModel = reqCtx.Model
	if len(modelObj.Spec.TargetModels) > 0 {
		reqCtx.ResolvedTargetModel = RandomWeightedDraw(logger, modelObj, 0)
		if reqCtx.ResolvedTargetModel == "" {
			logger.Error(nil, "Failed to get a resolved target model from TargetModels spec",
				"model", reqCtx.Model)
			return reqCtx, errutil.Error{
				Code: errutil.BadConfiguration,
				Msg:  "error resolving target model for " + reqCtx.Model,
			}
		}
	}

	requestCriticality := v1alpha2.Standard
	if modelObj.Spec.Criticality != nil {
		requestCriticality = *modelObj.Spec.Criticality
	}

	// Prepare LLMRequest (needed for both saturation detection and Scheduler)
	llmReq := &schedulingtypes.LLMRequest{
		Model:               reqCtx.Model,
		ResolvedTargetModel: reqCtx.ResolvedTargetModel,
		Critical:            requestCriticality == v1alpha2.Critical,
		Prompt:              prompt,
		Headers:             reqCtx.Request.Headers,
	}
	logger = logger.WithValues("model", llmReq.Model,
		"resolvedTargetModel", llmReq.ResolvedTargetModel,
		"criticality", requestCriticality,
		"isCriticalFlag", llmReq.Critical)
	ctx = log.IntoContext(ctx, logger)
	logger.V(logutil.DEBUG).Info("LLM request assembled")

	// --- 2. Saturation Check ---
	logger.V(logutil.DEBUG).Info("Calling PreDispatch")
	preDispatchErr := d.PreDispatch(ctx, reqCtx, requestCriticality)
	if preDispatchErr != nil {
		logger.Error(preDispatchErr, "PreDispatch failed")
		return reqCtx, preDispatchErr
	}

	// --- 3. Dispatch (Calls Scheduler) ---
	logger.V(logutil.DEBUG).Info("Calling Dispatch")
	results, dispatchErr := d.Dispatch(ctx, llmReq)
	if dispatchErr != nil {
		logger.Error(dispatchErr, "Dispatch failed")
		return reqCtx, dispatchErr
	}

	// --- 4. PostDispatch (Populates RequestContext) ---
	// Insert target endpoint to instruct Envoy to route requests to the
	// specified target pod.
	// Attach the port number.
	logger.V(logutil.DEBUG).Info("Calling PostDispatch")
	reqCtx, postDispatchErr := d.PostDispatch(ctx, reqCtx, results)
	if postDispatchErr != nil {
		logger.Error(postDispatchErr, "PostDispatch failed")
		return reqCtx, postDispatchErr
	}

	return reqCtx, nil
}

// PreDispatch handles admission control before dispatch.
func (d *Director) PreDispatch(ctx context.Context, reqCtx *handlers.RequestContext, reqCriticality v1alpha2.Criticality) error {
	logger := log.FromContext(ctx)
	logger.V(logutil.DEBUG).Info("Performing saturation check if request is non-critical.")
	if d.saturationDetector == nil {
		// Should we fail close here?
		logger.Error(nil, "SaturationDetector is nil; cannot perform direct saturation check. Proceeding.")
		return nil
	}

	// Check saturation directly ONLY for non-critical requests.
	if reqCriticality != v1alpha2.Critical && d.saturationDetector.IsSaturated() {
		logger.Info("System saturated, dropping non-critical request")
		return errutil.Error{
			Code: errutil.InferencePoolResourceExhausted,
			Msg:  "system saturated, non-critical request dropped",
		}
	}
	logger.V(logutil.DEBUG).Info("Proceeding to Dispatch (request is critical or system not saturated).")
	return nil
}

// Dispatch runs one or many scheduling cycles.
func (d *Director) Dispatch(ctx context.Context, llmReq *schedulingtypes.LLMRequest) ([]*schedulingtypes.Result, error) {
	res, err := d.scheduler.Schedule(ctx, llmReq)
	if err != nil {
		return nil, errutil.Error{
			Code: errutil.InferencePoolResourceExhausted,
			Msg:  fmt.Errorf("scheduler failed: %w", err).Error(),
		}
	}
	if res == nil { // Defensive check
		return nil, errutil.Error{
			Code: errutil.Internal,
			Msg:  "scheduler returned nil result without error",
		}
	}

	return []*schedulingtypes.Result{res}, nil
}

// PostDispatch populates the RequestContext based on scheduling results.
func (d *Director) PostDispatch(ctx context.Context, reqCtx *handlers.RequestContext, results []*schedulingtypes.Result) (*handlers.RequestContext, error) {
	logger := log.FromContext(ctx)
	// Currently only get a single result. Will refactor to pluggably implement
	// the PostSchedule.
	if len(results) == 0 || results[0] == nil || results[0].TargetPod == nil || results[0].TargetPod.GetPod() == nil {
		logger.Error(nil, "PostDispatch called with invalid scheduling results")
		return reqCtx, errutil.Error{
			Code: errutil.Internal,
			Msg:  "invalid scheduling result in PostDispatch",
		}
	}
	targetPod := results[0].TargetPod.GetPod()

	pool, err := d.datastore.PoolGet()
	if err != nil {
		logger.Error(err, "Failed to get InferencePool info for port")
		return reqCtx, errutil.Error{
			Code: errutil.Internal,
			Msg:  "failed to get pool configuration",
		}
	}

	endpoint := targetPod.Address + ":" + strconv.Itoa(int(pool.Spec.TargetPortNumber))
	logger.V(logutil.DEFAULT).Info("Request scheduled",
		"targetPod", targetPod.NamespacedName.String(), "endpoint", endpoint)

	// Update target models in the body if needed (traffic splitting)
	// This is where the body mutation happens based on the resolved model.
	if reqCtx.Model != reqCtx.ResolvedTargetModel {
		reqCtx.Request.Body["model"] = reqCtx.ResolvedTargetModel
	}

	// Populate context for response generation
	reqCtx.TargetPod = targetPod.NamespacedName.String()
	reqCtx.TargetEndpoint = endpoint

	return reqCtx, nil
}

// GetRandomPod selects a random pod.
func (d *Director) GetRandomPod() *backend.Pod {
	pods := d.datastore.PodGetAll()
	if len(pods) == 0 {
		return nil
	}
	number := rand.Intn(len(pods))
	pod := pods[number]
	return pod.GetPod()
}

// RandomWeightedDraw selects a model name based on weighted random
// distribution.
func RandomWeightedDraw(logger logr.Logger, model *v1alpha2.InferenceModel, seed int64) string {
	// TODO: after we are down to 1 server implementation, make these methods a
	// part of the struct and handle random seeding on the struct.
	source := rand.NewSource(rand.Int63())
	if seed > 0 {
		source = rand.NewSource(seed)
	}
	r := rand.New(source)

	// All the weight values are nil, then we should return random model name.
	if model.Spec.TargetModels[0].Weight == nil {
		index := r.Int31n(int32(len(model.Spec.TargetModels)))
		return model.Spec.TargetModels[index].Name
	}

	var weights int32
	for _, model := range model.Spec.TargetModels {
		weights += *model.Weight
	}
	logger.V(logutil.TRACE).Info("Weights for model computed",
		"model", model.Name, "weights", weights)
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
