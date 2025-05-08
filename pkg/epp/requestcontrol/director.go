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
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
)

type Scheduler interface {
	Schedule(ctx context.Context, b *schedulingtypes.LLMRequest) (result *schedulingtypes.Result, err error)
}

type Director struct {
	datastore datastore.Datastore
	scheduler Scheduler
}

func NewDirector(datastore datastore.Datastore, scheduler Scheduler) *Director {
	return &Director{
		datastore: datastore,
		scheduler: scheduler,
	}
}

// HandleRequest always returns the requestContext even in the error case, as the request context is used in error handling.
func (d *Director) HandleRequest(ctx context.Context, reqCtx *handlers.RequestContext) (*handlers.RequestContext, error) {
	logger := log.FromContext(ctx)

	// Resolve target models.
	var ok bool
	requestBodyMap := reqCtx.Request.Body
	reqCtx.Model, ok = requestBodyMap["model"].(string)
	if !ok {
		return reqCtx, errutil.Error{Code: errutil.BadRequest, Msg: "model not found in request"}
	}
	prompt, ok := requestBodyMap["prompt"].(string)
	if !ok {
		return reqCtx, errutil.Error{Code: errutil.BadRequest, Msg: "prompt not found in request"}
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
	}

	llmReq := &schedulingtypes.LLMRequest{
		RequestId:           reqCtx.Request.Headers[requtil.RequestIdHeaderKey],
		Model:               reqCtx.Model,
		ResolvedTargetModel: reqCtx.ResolvedTargetModel,
		Critical:            modelObj.Spec.Criticality != nil && *modelObj.Spec.Criticality == v1alpha2.Critical,
		Prompt:              prompt,
		Headers:             reqCtx.Request.Headers,
	}
	logger.V(logutil.DEBUG).Info("LLM request assembled", "request", llmReq)
	results, err := d.Dispatch(ctx, llmReq)
	if err != nil {
		return reqCtx, err
	}

	// Insert target endpoint to instruct Envoy to route requests to the specified target pod.
	// Attach the port number
	reqCtx, err = d.PostDispatch(ctx, reqCtx, results)
	if err != nil {
		return reqCtx, err
	}

	return reqCtx, nil
}

// Dispatch runs one or many scheduling cycles.
func (d *Director) Dispatch(ctx context.Context, llmReq *schedulingtypes.LLMRequest) ([]*schedulingtypes.Result, error) {
	var err error
	res, err := d.scheduler.Schedule(ctx, llmReq)
	if err != nil {
		return nil, errutil.Error{Code: errutil.InferencePoolResourceExhausted, Msg: fmt.Errorf("failed to find target pod: %w", err).Error()}
	}

	return []*schedulingtypes.Result{res}, nil
}

func (d *Director) PostDispatch(ctx context.Context, reqCtx *handlers.RequestContext, results []*schedulingtypes.Result) (*handlers.RequestContext, error) {
	logger := log.FromContext(ctx)
	// currently only get a single result. Will refactor to pluggably implement the PostSchedule
	if len(results) == 0 {
		return reqCtx, errutil.Error{Code: errutil.Internal, Msg: "results must be greater than zero"}
	}
	targetPod := results[0].TargetPod.GetPod()

	pool, err := d.datastore.PoolGet()
	if err != nil {
		return reqCtx, err
	}

	endpoint := targetPod.Address + ":" + strconv.Itoa(int(pool.Spec.TargetPortNumber))
	logger.V(logutil.DEFAULT).Info("Request handled",
		"model", reqCtx.Model, "targetModel", reqCtx.ResolvedTargetModel, "endpoint", targetPod)

	// Update target models in the body.
	if reqCtx.Model != reqCtx.ResolvedTargetModel {
		reqCtx.Request.Body["model"] = reqCtx.ResolvedTargetModel
	}
	reqCtx.TargetPod = targetPod.NamespacedName.String()
	reqCtx.TargetEndpoint = endpoint

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
