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
	// TargetModel is the final target model after traffic split.
	TargetModel string
	// Prompt is the prompt that was sent in the request body.
	Prompt string
	// Headers is a map of the request headers.
	Headers map[string]string
}

func (r *LLMRequest) String() string {
	return fmt.Sprintf("RequestID: %s, TargetModel: %s, PromptLength: %d, Headers: %v", r.RequestId, r.TargetModel, len(r.Prompt), r.Headers)
}

type Pod interface {
	GetPod() *backend.Pod
	GetMetrics() *backendmetrics.MetricsState
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

func (pm *PodMetrics) GetMetrics() *backendmetrics.MetricsState {
	return pm.MetricsState
}

type PodMetrics struct {
	*backend.Pod
	*backendmetrics.MetricsState
}

// ProfileRunResult captures the profile run result.
type ProfileRunResult struct {
	TargetPods []Pod
	// RawScores is a map of raw scores for each pod, keyed by scorer type.
	RawScores map[string]map[Pod]float64
}

// SchedulingResult captures the result of the scheduling cycle.
type SchedulingResult struct {
	ProfileResults     map[string]*ProfileRunResult
	AllProfileRunResults map[string]*ProfileRunResult
	PrimaryProfileName string
}
