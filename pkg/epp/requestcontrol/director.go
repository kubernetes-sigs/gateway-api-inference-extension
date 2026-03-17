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
	"encoding/json"
	"fmt"
	"math/rand"
	"net"
	"strings"
	"time"

	"google.golang.org/protobuf/proto"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	errcommon "sigs.k8s.io/gateway-api-inference-extension/pkg/common/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	reqcommon "sigs.k8s.io/gateway-api-inference-extension/pkg/common/request"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwk "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	fwkrh "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
	fwksched "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/handlers"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
)

const (
	// TODO(https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/2081):
	// Make this timeout configurable per-plugin or globally via the Director configuration to support plugins with
	// varying latency profiles.
	prepareDataTimeout = 400 * time.Millisecond
)

// Datastore defines the interface required by the Director.
type Datastore interface {
	PoolGet() (*datalayer.EndpointPool, error)
	ObjectiveGet(objectiveName string) *v1alpha2.InferenceObjective
	PodList(predicate func(fwkdl.Endpoint) bool) []fwkdl.Endpoint
	// ModelRewriteGet returns the rewrite rule for a given model name and the name of the InferenceModelRewrite object.
	ModelRewriteGet(modelName string) (*v1alpha2.InferenceModelRewriteRule, string)
}

// Scheduler defines the interface required by the Director for scheduling.
type Scheduler interface {
	Schedule(ctx context.Context, request *fwksched.LLMRequest, candidateEndpoints []fwksched.Endpoint) (result *fwksched.SchedulingResult, err error)
}

// NewDirectorWithConfig creates a new Director instance with all dependencies.
func NewDirectorWithConfig(
	datastore Datastore,
	scheduler Scheduler,
	admissionController AdmissionController,
	parser fwkrh.Parser,
	podLocator contracts.PodLocator,
	config *Config,
) *Director {
	return &Director{
		datastore:             datastore,
		scheduler:             scheduler,
		admissionController:   admissionController,
		podLocator:            podLocator,
		requestControlPlugins: *config,
		parser:                parser,
		defaultPriority:       0, // define default priority explicitly
	}
}

// Director orchestrates the request handling flow after initial parsing by the handler.
// Its responsibilities include:
// - Retrieving request metadata and relevant objectives.
// - Determining candidate pods.
// - Performing admission control via the AdmissionController.
// - Scheduling the request to target pod(s) via the Scheduler.
// - Running PreRequest plugins.
// - Preparing the request context for the Envoy ext_proc filter to route the request.
// - Running PostResponse plugins.
type Director struct {
	datastore             Datastore
	scheduler             Scheduler
	admissionController   AdmissionController
	podLocator            contracts.PodLocator
	requestControlPlugins Config
	// we just need a pointer to an int variable since priority is a pointer in InferenceObjective
	// no need to set this in the constructor, since the value we want is the default int val
	// and value types cannot be nil
	defaultPriority int
	parser          fwkrh.Parser
}

// getInferenceObjective fetches the inferenceObjective from the datastore otherwise creates a new one based on reqCtx.
func (d *Director) getInferenceObjective(ctx context.Context, reqCtx *handlers.RequestContext) *v1alpha2.InferenceObjective {
	infObjective := d.datastore.ObjectiveGet(reqCtx.ObjectiveKey)
	if infObjective == nil {
		log.FromContext(ctx).V(logutil.VERBOSE).Info("No associated InferenceObjective found, using default", "objectiveKey", reqCtx.ObjectiveKey)
		infObjective = &v1alpha2.InferenceObjective{
			Spec: v1alpha2.InferenceObjectiveSpec{
				Priority: &d.defaultPriority,
			},
		}
	} else if infObjective.Spec.Priority == nil {
		// Default to 0 if not specified.
		infObjective.Spec.Priority = &d.defaultPriority
	}
	return infObjective
}

// HandleRequest orchestrates the request lifecycle.
// It always returns the requestContext even in the error case, as the request context is used in error handling.
func (d *Director) HandleRequest(ctx context.Context, reqCtx *handlers.RequestContext) (*handlers.RequestContext, error) {
	logger := log.FromContext(ctx)

	// Parse, mutate, and extract the request body
	llmRequestBody, err := d.processRequestBody(ctx, reqCtx, d.parser)
	if err != nil {
		return reqCtx, err
	}

	infObjective := d.getInferenceObjective(ctx, reqCtx)
	requestObjectives := fwksched.RequestObjectives{Priority: *infObjective.Spec.Priority}

	reqCtx.SchedulingRequest = &fwksched.LLMRequest{
		RequestId:        reqCtx.Request.Headers[reqcommon.RequestIdHeaderKey],
		TargetModel:      reqCtx.TargetModelName,
		Body:             llmRequestBody,
		Headers:          reqCtx.Request.Headers,
		Objectives:       requestObjectives,
		RequestSizeBytes: reqCtx.RequestSize,
	}

	logger = logger.WithValues("objectiveKey", reqCtx.ObjectiveKey, "incomingModelName", reqCtx.IncomingModelName, "targetModelName", reqCtx.TargetModelName, "priority", infObjective.Spec.Priority)
	ctx = log.IntoContext(ctx, logger)
	logger.V(logutil.DEBUG).Info("LLM request assembled")

	if err := d.admissionController.Admit(ctx, reqCtx, *infObjective.Spec.Priority); err != nil {
		logger.V(logutil.DEFAULT).Info("Request rejected by admission control", "error", err)
		return reqCtx, err
	}
	candidateEndpoints := d.podLocator.Locate(ctx, reqCtx.Request.Metadata)
	if len(candidateEndpoints) == 0 {
		return reqCtx, errcommon.Error{
			Code: errcommon.ServiceUnavailable,
			Msg:  "failed to find candidate endpoints for serving the request",
		}
	}
	candidateSnapshot := d.toSchedulerEndpoints(candidateEndpoints)

	// Prepare per request data by running PrepareData plugins.
	err = d.runPrepareDataPlugins(ctx, reqCtx.SchedulingRequest, candidateSnapshot)
	if err != nil {
		// Don't fail the request if PrepareData plugins fail.
		logger.V(logutil.DEFAULT).Error(err, "failed to prepare per request data")
	}

	// Run admit request plugins
	if !d.runAdmissionPlugins(ctx, reqCtx.SchedulingRequest, candidateSnapshot) {
		logger.V(logutil.DEFAULT).Info("Request cannot be admitted")
		return reqCtx, errcommon.Error{Code: errcommon.Internal, Msg: "request cannot be admitted"}
	}

	result, err := d.scheduler.Schedule(ctx, reqCtx.SchedulingRequest, candidateSnapshot)
	if err != nil {
		return reqCtx, errcommon.Error{Code: errcommon.ResourceExhausted, Msg: fmt.Errorf("failed to find target endpoint: %w", err).Error()}
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

func (d *Director) processRequestBody(ctx context.Context, reqCtx *handlers.RequestContext, parser fwkrh.Parser) (*fwksched.LLMRequestBody, error) {
	llmRequestBody, err := parser.ParseRequest(ctx, reqCtx.Request.RawBody, reqCtx.Request.Headers)
	if err != nil {
		return nil, errcommon.Error{Code: errcommon.BadRequest, Msg: err.Error()}
	}

	switch v := llmRequestBody.ParsedBody.(type) {
	case proto.Message:
		// Protos are not currently mutated, return as-is.
		reqCtx.RequestSize = len(reqCtx.Request.RawBody)
	case map[string]any:
		if err := d.mutateAndRepackage(ctx, reqCtx, v); err != nil {
			return nil, err
		}
	default:
		return nil, errcommon.Error{Code: errcommon.BadRequest, Msg: "Unsupported llmRequest parsedBody"}
	}
	return llmRequestBody, nil
}

func (d *Director) mutateAndRepackage(ctx context.Context, reqCtx *handlers.RequestContext, bodyMap map[string]any) error {
	logger := log.FromContext(ctx)

	// Mutate the model name inside the map
	_, err := d.mutateModel(reqCtx, bodyMap)
	if err != nil {
		return err
	}

	// Marshal back to bytes so downstream ExtProc filters see the updated model
	requestBodyBytes, err := json.Marshal(bodyMap)
	if err != nil {
		logger.V(logutil.DEFAULT).Error(err, "Error marshalling request body")
		return errcommon.Error{Code: errcommon.Internal, Msg: "Error marshalling request body"}
	}

	reqCtx.Request.RawBody = requestBodyBytes
	reqCtx.RequestSize = len(requestBodyBytes)

	return nil
}

func (d *Director) mutateModel(reqCtx *handlers.RequestContext, bodyMap map[string]any) (*handlers.RequestContext, error) {
	var ok bool
	reqCtx.IncomingModelName, ok = bodyMap["model"].(string)
	if !ok {
		return reqCtx, errcommon.Error{Code: errcommon.BadRequest, Msg: "model not found in request body"}
	}
	if reqCtx.TargetModelName == "" {
		// Default to incoming model name
		reqCtx.TargetModelName = reqCtx.IncomingModelName
	}
	d.applyWeightedModelRewrite(reqCtx)
	bodyMap["model"] = reqCtx.TargetModelName
	return reqCtx, nil
}

func (d *Director) applyWeightedModelRewrite(reqCtx *handlers.RequestContext) {
	rewriteRule, modelRewriteName := d.datastore.ModelRewriteGet(reqCtx.IncomingModelName)
	if rewriteRule == nil {
		return
	}
	reqCtx.TargetModelName = d.selectWeightedModel(rewriteRule.Targets)
	metrics.RecordInferenceModelRewriteDecision(modelRewriteName, reqCtx.IncomingModelName, reqCtx.TargetModelName)
}

func (d *Director) selectWeightedModel(models []v1alpha2.TargetModel) string {
	if len(models) == 0 {
		return ""
	}

	var totalWeight int32
	for _, model := range models {
		totalWeight += model.Weight
	}

	if totalWeight == 0 {
		// If total weight is 0, distribute evenly
		return models[rand.Intn(len(models))].ModelRewrite
	}

	randomNum := rand.Intn(int(totalWeight))
	var currentWeight int32
	for _, model := range models {
		currentWeight += model.Weight
		if randomNum < int(currentWeight) {
			return model.ModelRewrite
		}
	}

	// Should not happen
	return models[len(models)-1].ModelRewrite
}

// prepareRequest populates the RequestContext and calls the registered PreRequest plugins
// for allowing plugging customized logic based on the scheduling result.
func (d *Director) prepareRequest(ctx context.Context, reqCtx *handlers.RequestContext, result *fwksched.SchedulingResult) (*handlers.RequestContext, error) {
	logger := log.FromContext(ctx)
	if result == nil || len(result.ProfileResults) == 0 {
		return reqCtx, errcommon.Error{Code: errcommon.Internal, Msg: "results must be greater than zero"}
	}
	// primary profile is used to set destination
	targetMetadatas := []*fwkdl.EndpointMetadata{}
	targetEndpoints := []string{}

	for _, pod := range result.ProfileResults[result.PrimaryProfileName].TargetEndpoints {
		curMetadata := pod.GetMetadata()
		curEndpoint := net.JoinHostPort(curMetadata.GetIPAddress(), curMetadata.GetPort())
		targetMetadatas = append(targetMetadatas, curMetadata)
		targetEndpoints = append(targetEndpoints, curEndpoint)
	}

	multiEndpointString := strings.Join(targetEndpoints, ",")
	logger.V(logutil.VERBOSE).Info("Request handled", "objectiveKey", reqCtx.ObjectiveKey, "incomingModelName", reqCtx.IncomingModelName, "targetModel", reqCtx.TargetModelName, "endpoint", multiEndpointString)

	reqCtx.TargetPod = targetMetadatas[0]
	reqCtx.TargetEndpoint = multiEndpointString

	d.runPreRequestPlugins(ctx, reqCtx.SchedulingRequest, result)

	return reqCtx, nil
}

func (d *Director) toSchedulerEndpoints(endpoints []fwkdl.Endpoint) []fwksched.Endpoint {
	result := make([]fwksched.Endpoint, len(endpoints))
	for i, endpoint := range endpoints {
		result[i] = fwksched.NewEndpoint(endpoint.GetMetadata(), endpoint.GetMetrics(), endpoint.GetAttributes())
	}

	return result
}

// HandleResponseHeader is called when the response headers are received.
func (d *Director) HandleResponseHeader(ctx context.Context, reqCtx *handlers.RequestContext) *handlers.RequestContext {
	response := &fwk.Response{
		RequestId:   reqCtx.Request.Headers[reqcommon.RequestIdHeaderKey],
		Headers:     reqCtx.Response.Headers,
		ReqMetadata: reqCtx.Request.Metadata,
	}
	// TODO: to extend fallback functionality, handle cases where target pod is unavailable
	// https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/1224
	d.runResponseHeaderPlugins(ctx, reqCtx.SchedulingRequest, response, reqCtx.TargetPod)
	return reqCtx
}

// HandleResponseBody is invoked by the director for every chunk received in a streaming
// response, or exactly once for a non-streaming response.
func (d *Director) HandleResponseBody(ctx context.Context, reqCtx *handlers.RequestContext, endOfStream bool) *handlers.RequestContext {
	logger := log.FromContext(ctx).WithValues("stage", "bodyChunk")
	logger.V(logutil.TRACE).Info("Entering HandleResponseBodyChunk")
	response := &fwk.Response{
		RequestId:   reqCtx.Request.Headers[reqcommon.RequestIdHeaderKey],
		Headers:     reqCtx.Response.Headers,
		EndOfStream: endOfStream,
	}
	d.runResponseBodyPlugins(ctx, reqCtx.SchedulingRequest, response, reqCtx.TargetPod)
	logger.V(logutil.TRACE).Info("Exiting HandleResponseBodyChunk")
	return reqCtx
}

func (d *Director) GetRandomEndpoint() *fwkdl.EndpointMetadata {
	pods := d.datastore.PodList(datastore.AllPodsPredicate)
	if len(pods) == 0 {
		return nil
	}
	number := rand.Intn(len(pods))
	pod := pods[number]
	return pod.GetMetadata()
}

func (d *Director) runPreRequestPlugins(ctx context.Context, request *fwksched.LLMRequest,
	schedulingResult *fwksched.SchedulingResult) {
	loggerDebug := log.FromContext(ctx).V(logutil.DEBUG)
	for _, plugin := range d.requestControlPlugins.preRequestPlugins {
		loggerDebug.Info("Running PreRequest plugin", "plugin", plugin.TypedName())
		before := time.Now()
		plugin.PreRequest(ctx, request, schedulingResult)
		metrics.RecordPluginProcessingLatency(fwk.PreRequestExtensionPoint, plugin.TypedName().Type, plugin.TypedName().Name, time.Since(before))
		loggerDebug.Info("Completed running PreRequest plugin successfully", "plugin", plugin.TypedName())
	}
}

func (d *Director) runPrepareDataPlugins(ctx context.Context,
	request *fwksched.LLMRequest, endpoints []fwksched.Endpoint) error {
	if len(d.requestControlPlugins.prepareDataPlugins) == 0 {
		return nil
	}
	return prepareDataPluginsWithTimeout(prepareDataTimeout, d.requestControlPlugins.prepareDataPlugins, ctx, request, endpoints)
}

func (d *Director) runAdmissionPlugins(ctx context.Context,
	request *fwksched.LLMRequest, endpoints []fwksched.Endpoint) bool {
	loggerDebug := log.FromContext(ctx).V(logutil.DEBUG)
	for _, plugin := range d.requestControlPlugins.admissionPlugins {
		loggerDebug.Info("Running AdmitRequest plugin", "plugin", plugin.TypedName())
		if denyReason := plugin.AdmitRequest(ctx, request, endpoints); denyReason != nil {
			loggerDebug.Info("AdmitRequest plugin denied the request", "plugin", plugin.TypedName(), "reason", denyReason.Error())
			return false
		}
		loggerDebug.Info("Completed running AdmitRequest plugin successfully", "plugin", plugin.TypedName())
	}
	return true
}

func (d *Director) runResponseHeaderPlugins(ctx context.Context, request *fwksched.LLMRequest, response *fwk.Response, targetEndpoint *fwkdl.EndpointMetadata) {
	loggerDebug := log.FromContext(ctx).V(logutil.DEBUG)
	for _, plugin := range d.requestControlPlugins.responseReceivedPlugins {
		loggerDebug.Info("Running ResponseReceived plugin", "plugin", plugin.TypedName())
		before := time.Now()
		plugin.ResponseHeader(ctx, request, response, targetEndpoint)
		metrics.RecordPluginProcessingLatency(fwk.ResponseReceivedExtensionPoint, plugin.TypedName().Type, plugin.TypedName().Name, time.Since(before))
		loggerDebug.Info("Completed running ResponseReceived plugin successfully", "plugin", plugin.TypedName())
	}
}

func (d *Director) runResponseBodyPlugins(ctx context.Context, request *fwksched.LLMRequest, response *fwk.Response, targetEndpoint *fwkdl.EndpointMetadata) {
	loggerTrace := log.FromContext(ctx).V(logutil.TRACE)
	for _, plugin := range d.requestControlPlugins.responseStreamingPlugins {
		loggerTrace.Info("Running ResponseStreaming plugin", "plugin", plugin.TypedName())
		before := time.Now()
		plugin.ResponseBody(ctx, request, response, targetEndpoint)
		metrics.RecordPluginProcessingLatency(fwk.ResponseStreamingExtensionPoint, plugin.TypedName().Type, plugin.TypedName().Name, time.Since(before))
		loggerTrace.Info("Completed running ResponseStreaming plugin successfully", "plugin", plugin.TypedName())
	}
}
