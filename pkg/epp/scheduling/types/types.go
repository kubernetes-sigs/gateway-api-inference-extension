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
	"context"
	"fmt"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
)

// LLMRequest is a structured representation of the fields we parse out of the LLMRequest body.
type LLMRequest struct {
	Model string
	// Target models is a map of target model name to weight.
	TargetModels map[string]int
	Prompt       string
	// Resolved target model is the final target model after traffic split.
	ResolvedTargetModel string
	Critical            bool
}

func (r *LLMRequest) String() string {
	return fmt.Sprintf("Model: %s, TargetModels: %v, ResolvedTargetModel: %s, Critical: %t, PromptLength: %v", r.Model, r.TargetModels, r.ResolvedTargetModel, r.Critical, len(r.Prompt))
}

// Context holds contextual information during a scheduling operation.
type Context struct {
	context.Context
	Logger       logr.Logger
	Req          *LLMRequest
	PodsSnapshot []Pod
}

func (pm *PodMetrics) String() string {
	if pm == nil {
		return ""
	}
	return fmt.Sprintf("%+v", *pm)
}

func (pm *PodMetrics) GetPod() *backendmetrics.Pod {
	return pm.Pod
}

func (pm *PodMetrics) GetMetrics() *backendmetrics.Metrics {
	return pm.Metrics
}

func (pm *PodMetrics) SetScore(score float64) {
	pm.score = score
}

func (pm *PodMetrics) Score() float64 {
	return pm.score
}

type PodMetrics struct {
	score float64
	*backendmetrics.Pod
	*backendmetrics.Metrics
}

func NewContext(ctx context.Context, req *LLMRequest, pods []Pod) *Context {
	logger := log.FromContext(ctx).WithValues("request", req)
	return &Context{
		Context:      ctx,
		Logger:       logger,
		Req:          req,
		PodsSnapshot: pods,
	}
}

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
