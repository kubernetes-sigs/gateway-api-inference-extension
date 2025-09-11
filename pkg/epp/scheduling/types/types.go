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

const nilString = "<nil>"

// LLMRequest is a structured representation of the fields we parse out of the LLMRequest body.
type LLMRequest struct {
	// RequestId is the Envoy generated Id for the request being processed
	RequestId string
	// TargetModel is the final target model after traffic split.
	TargetModel string
	// Data contains the request-body fields that we parse out as user input.
	Body *LLMRequestBody
	// Headers is a map of the request headers.
	Headers map[string]string

	// TTFTSLO is the target time to first token SLO for the request.
	TTFTSLO float64
	// TPOTSLO is the target time per output token SLO for the request.
	AvgTPOTSLO float64
	// PredictorBasedScheduling indicates whether to use predictor based scheduling.

	// ### TODO Move below fields to the datalayer request object

	PredictorBasedScheduling bool
	//PredictedTTFTForScheduling is the map of pod names to predicted TTFT values for scheduling.
	PredictedTTFTForScheduling map[string]float64
	// PredictedTPOTForScheduling is the map of pod names to predicted TPOT values for scheduling.
	PredictedTPOTForScheduling map[string]float64

	// boolean set if request has valid pod based on predictions
	HasValidPod bool
}

func (r *LLMRequest) String() string {
	if r == nil {
		return nilString
	}

	return fmt.Sprintf("RequestID: %s, TargetModel: %s, Body: %s, Headers: %v",
		r.RequestId, r.TargetModel, r.Body, r.Headers)
}

// LLMRequestBody contains the request-body fields that we parse out as user input,
// to be used in forming scheduling decisions.
// An LLMRequestBody must contain exactly one of CompletionsRequest or ChatCompletionsRequest.
type LLMRequestBody struct {
	// CompletionsRequest is the representation of the OpenAI /v1/completions request body.
	Completions *CompletionsRequest `json:"completions,omitempty"`
	// ChatCompletionsRequest is the representation of the OpenAI /v1/chat_completions request body.
	ChatCompletions *ChatCompletionsRequest `json:"chat_completions,omitempty"`
}

// CompletionsRequest is a structured representation of the fields we parse out of the
// /v1/completions request body.
// This struct includes fields usable for plugins and scheduling decisions - and not the entire
// API spec.
type CompletionsRequest struct {
	// Prompt is the prompt that was sent in the request body.
	Prompt string `json:"prompt,omitempty"`
}

func (r *CompletionsRequest) String() string {
	if r == nil {
		return nilString
	}

	return fmt.Sprintf("{PromptLength: %d}", len(r.Prompt))
}

// ChatCompletionsRequest is a structured representation of the fields we parse out of the
// /v1/chat/completions request body.
// This struct includes fields usable for plugins and scheduling decisions - and not the entire
// API spec.
type ChatCompletionsRequest struct {
	/* parameters from the official OpenAI chat-completions API */
	Messages []Message     `json:"messages,omitempty"`
	Tools    []interface{} `json:"tools,omitempty"`
	/* parameters from the HuggingFace transformers chat-templates API */
	Documents                 []interface{}          `json:"documents,omitempty"`
	ChatTemplate              string                 `json:"chat_template,omitempty"`
	ReturnAssistantTokensMask bool                   `json:"return_assistant_tokens_mask,omitempty"`
	ContinueFinalMessage      bool                   `json:"continue_final_message,omitempty"`
	AddGenerationPrompt       bool                   `json:"add_generation_prompt,omitempty"`
	ChatTemplateKWArgs        map[string]interface{} `json:"chat_template_kwargs,omitempty"`
}

func (r *ChatCompletionsRequest) String() string {
	if r == nil {
		return nilString
	}

	messagesLen := 0
	for _, msg := range r.Messages {
		messagesLen += len(msg.Content)
	}

	return fmt.Sprintf("{MessagesLength: %d}", messagesLen)
}

// Message represents a single message in a chat-completions request.
type Message struct {
	Role    string
	Content string // TODO: support multi-modal content
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
		return nilString
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
	ProfileResults       map[string]*ProfileRunResult
	AllProfileRunResults map[string]*ProfileRunResult
	PrimaryProfileName   string
}
