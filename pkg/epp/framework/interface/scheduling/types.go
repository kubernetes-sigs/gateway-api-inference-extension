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

package scheduling

import (
	"context"
	"fmt"
	"reflect"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
)

const nilString = "<nil>"

// RequestObjectives represents the scheduling objectives parsed from the InferenceObjectiveSpec, to be used in scheduling decisions.
type RequestObjectives struct {
	Priority int
}

// LLMRequest is a structured representation of the fields we parse out of the LLMRequest body.
type LLMRequest struct {
	// RequestId is the Envoy generated Id for the request being processed
	RequestId string
	// TargetModel is the final target model after traffic split.
	TargetModel string
	// Data contains the request-body fields that we parse out as user input.
	Body *requesthandling.RequestBody
	// Headers is a map of the request headers.
	Headers map[string]string
	// Request Objective
	Objectives RequestObjectives
	// RequestSizeBytes is the size of the raw request body in bytes when available.
	// Used for token estimation (e.g. inputTokens ≈ RequestSizeBytes/4) without parsing body or calling PlainText().
	RequestSizeBytes int
	// TokenizedPrompt contains the tokenization results if external tokenization is enabled.
	// This is nil if tokenization was not performed or if the tokenizer is not configured.
	TokenizedPrompt *TokenizedPrompt
}

// TokenizedPrompt contains the result of tokenizing the request prompt.
// It is populated by external tokenization plugins (e.g., via a PrepareData plugin)
// and consumed by scheduling plugins that benefit from actual token data
// (e.g., prefix cache scoring, latency prediction).
type TokenizedPrompt struct {
	// TokenIDs are the token IDs for the prompt.
	TokenIDs []uint32
}

type InferenceRequest struct {
	LLM        *LLMRequest
	ParsedBody any `json:"-"`

	// Stream indicates whether the request specifies a streaming response (e.g., via a stream field).
	// This typically implies the model server's response will be streamed.
	Stream bool `json:"-"`
}

func (r *InferenceRequest) String() string {
	if r == nil || r.LLM == nil {
		return nilString
	}

	return fmt.Sprintf("RequestID: %s, TargetModel: %s, Body: %s, Headers: %v",
		r.LLM.RequestId, r.LLM.TargetModel, r.LLM.Body, r.LLM.Headers)
}

type Endpoint interface {
	GetMetadata() *fwkdl.EndpointMetadata
	GetMetrics() *fwkdl.Metrics
	String() string
	Get(string) (fwkdl.Cloneable, bool)
	Put(string, fwkdl.Cloneable)
	Keys() []string
}

func (ep *endpoint) String() string {
	if ep == nil {
		return nilString
	}

	return fmt.Sprintf("%+v", *ep)
}

func (ep *endpoint) GetMetadata() *fwkdl.EndpointMetadata {
	return ep.EndpointMetadata
}

func (ep *endpoint) GetMetrics() *fwkdl.Metrics {
	return ep.Metrics
}

type endpoint struct {
	*fwkdl.EndpointMetadata
	*fwkdl.Metrics
	fwkdl.AttributeMap
}

func NewEndpoint(meta *fwkdl.EndpointMetadata, metrics *fwkdl.Metrics, attr fwkdl.AttributeMap) Endpoint {
	if attr == nil {
		attr = fwkdl.NewAttributes()
	}

	return &endpoint{
		EndpointMetadata: meta.Clone(),
		Metrics:          metrics.Clone(),
		AttributeMap:     attr.Clone(),
	}
}

func EndpointComparer(a, b Endpoint) bool {
	a_ep := a.(*endpoint)
	b_ep := b.(*endpoint)
	return reflect.DeepEqual(a_ep, b_ep)
}

func ScoredEndpointComparer(a, b ScoredEndpoint) bool {
	return a.Score == b.Score && EndpointComparer(a.Endpoint, b.Endpoint)
}

type ScoredEndpoint struct {
	Endpoint
	Score float64
}

// ProfileRunResult captures the profile run result.
type ProfileRunResult struct {
	TargetEndpoints []Endpoint
}

// SchedulingResult captures the result of the scheduling cycle.
type SchedulingResult struct {
	ProfileResults     map[string]*ProfileRunResult
	PrimaryProfileName string
}

type SchedulerProfile interface {
	Run(ctx context.Context, request *InferenceRequest, cycleState *CycleState, candidateEndpoints []Endpoint) (*ProfileRunResult, error)
}
