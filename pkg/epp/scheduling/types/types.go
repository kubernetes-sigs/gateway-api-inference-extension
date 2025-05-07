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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	podinfo "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/pod-info"
)

// LLMRequest is a structured representation of the fields we parse out of the LLMRequest body.
type LLMRequest struct {
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

type Pod interface {
	GetPod() *backend.Pod
	GetData() map[string]podinfo.ScrapedData
	String() string
}

type ScoredPod struct {
	Pod
	Score float64
}

func NewSchedulingContext(ctx context.Context, req *LLMRequest, pods []Pod) *SchedulingContext {
	return &SchedulingContext{
		Context:      ctx,
		Logger:       log.FromContext(ctx).WithValues("request", req),
		Req:          req,
		PodsSnapshot: pods,
	}
}

// SchedulingContext holds contextual information during a scheduling operation.
type SchedulingContext struct {
	context.Context
	Logger       logr.Logger
	Req          *LLMRequest
	PodsSnapshot []Pod
}

type PodData struct {
	*backend.Pod
	Data map[string]podinfo.ScrapedData
}

func (pd *PodData) String() string {
	if pd == nil {
		return ""
	}
	return fmt.Sprintf("%+v", *pd)
}

func (pd *PodData) GetPod() *backend.Pod {
	return pd.Pod
}

func (pd *PodData) GetData() map[string]podinfo.ScrapedData {
	return pd.Data
}

func ToSchedulerPodData(podsInfo []podinfo.PodInfo) []Pod {
	pods := make([]Pod, 0, len(podsInfo))
	for _, podInfo := range podsInfo {
		pods = append(pods, &PodData{Pod: podInfo.GetPod().Clone(), Data: clonePodData(podInfo)})
	}
	return pods
}

func clonePodData(podInfo podinfo.PodInfo) map[string]podinfo.ScrapedData {
	dataKeys := podInfo.GetDataKeys()
	copy := make(map[string]podinfo.ScrapedData, len(dataKeys))
	for _, key := range dataKeys {
		if data, ok := podInfo.GetData(key); ok {
			copy[key] = podinfo.CloneScrapedData(data)
		}
	}

	return copy
}

// Result captures the scheduler result.
type Result struct {
	TargetPod Pod
}
