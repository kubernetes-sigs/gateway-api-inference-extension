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

package types

import (
	"fmt"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
)

// LLMRequest is a structured representation of the fields we parse out of the LLMRequest body.
type LLMRequest struct {
	// RequestId is the Envoy generated Id for the request being processed
	RequestId string
	// Model is the name of the model that the user specified in the request body.
	Model string
	// ResolvedTargetModel is the final target model after traffic split.
	ResolvedTargetModel string
	// Critical is a boolean that specifies if a request is critical or not.
	Critical bool
	// Prompt is the prompt that was sent in the request body.
	Prompt string
	// Headers is a map of the request headers.
	Headers map[string]string
}

func (r *LLMRequest) String() string {
	return fmt.Sprintf("Model: %s, ResolvedTargetModel: %s, Critical: %t, PromptLength: %d, Headers: %v",
		r.Model, r.ResolvedTargetModel, r.Critical, len(r.Prompt), r.Headers)
}

// LLMResponse contains information from the response received to be passed to plugins
type LLMResponse struct {
	// RequestId is the Envoy generated Id for the request being processed
	RequestId string
	// Headers is a map of the response headers. Nil during body processing
	Headers map[string]string
	// Body Is the body of the response or nil during header processing
	Body string
	// IsStreaming indicates whether or not the response is being streamed by the model
	IsStreaming bool
	// EndOfStream when true indicates that this invocation contains the last chunk of the response
	EndOfStream bool
}

type Pod interface {
	GetPod() *backend.Pod
	GetMetrics() *backendmetrics.Metrics
	String() string
}

type ScoredPod struct {
	Pod
	Score float64
}

func (pm *PodMetrics) String() string {
	if pm == nil {
		return ""
	}
	return fmt.Sprintf("%+v", *pm)
}

func (pm *PodMetrics) GetPod() *backend.Pod {
	return pm.Pod
}

func (pm *PodMetrics) GetMetrics() *backendmetrics.Metrics {
	return pm.Metrics
}

type PodMetrics struct {
	*backend.Pod
	*backendmetrics.Metrics
}

<<<<<<< HEAD
func NewSchedulingContext(ctx context.Context, req *LLMRequest, resp *LLMResponse, pods []Pod) *SchedulingContext {
	var logger logr.Logger
	if req != nil {
		logger = log.FromContext(ctx).WithValues("request", req)
	} else {
		logger = log.FromContext(ctx).WithValues("response", resp)
	}
	return &SchedulingContext{
		Context:       ctx,
		Logger:        logger,
		Req:           req,
		Resp:          resp,
		PodsSnapshot:  pods,
		PluginState:   make(map[PluginName]any),
		pluginStateMu: &sync.RWMutex{},
	}
}

=======
>>>>>>> bc29bd0 (generalize scheduling cycle state concept (#818))
func ToSchedulerPodMetrics(pods []backendmetrics.PodMetrics) []Pod {
	pm := make([]Pod, 0, len(pods))
	for _, pod := range pods {
		pm = append(pm, &PodMetrics{Pod: pod.GetPod().Clone(), Metrics: pod.GetMetrics().Clone()})
	}
	return pm
}

// Result captures the scheduler result.
type Result struct {
	TargetPod Pod
}
