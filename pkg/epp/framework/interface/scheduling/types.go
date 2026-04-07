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
	fwkrh "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
)

const nilString = "<nil>"

// Modality identifies the type of multimodal content in a prompt.
type Modality string

// ModalityImage is the only currently supported modality.
const ModalityImage Modality = "image"

// RequestObjectives represents the scheduling objectives parsed from the InferenceObjectiveSpec, to be used in scheduling decisions.
type RequestObjectives struct {
	Priority int
}

// InferenceRequest is a structured representation of the fields we parse out of the InferenceRequest body.
type InferenceRequest struct {
	// RequestId is the Envoy generated Id for the request being processed
	RequestId string
	// TargetModel is the final target model after traffic split.
	TargetModel string
	// Data contains the request-body fields that we parse out as user input.
	Body *fwkrh.InferenceRequestBody
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
	// TokenIDs are the token IDs for the prompt, including multimodal placeholder tokens.
	TokenIDs []uint32
	// MultiModalFeatures holds one entry per multimodal item in prompt order.
	// Nil if the prompt contains no multimodal content.
	MultiModalFeatures []MultiModalFeature
}

// MultiModalFeature holds all data needed for precise prefix-cache scoring of a single
// multimodal item. Items are ordered by token position within the prompt.
// Currently only ModalityImage is supported.
type MultiModalFeature struct {
	// Modality identifies the type of content.
	Modality Modality
	// Hash is the content hash of the item, used for KV-cache reuse across requests.
	Hash string
	// Offset is the index of the first placeholder token for this item in TokenIDs.
	Offset int
	// Length is the number of placeholder tokens this item occupies in TokenIDs.
	Length int
}

func (r *InferenceRequest) String() string {
	if r == nil {
		return nilString
	}

	return fmt.Sprintf("RequestID: %s, TargetModel: %s, Body: %v, Headers: %v",
		r.RequestId, r.TargetModel, r.Body, r.Headers)
}

type Endpoint interface {
	GetMetadata() *fwkdl.EndpointMetadata
	GetMetrics() *fwkdl.Metrics
	String() string
	Get(string) (fwkdl.Cloneable, bool)
	Put(string, fwkdl.Cloneable)
	Keys() []string
	Clone() fwkdl.AttributeMap
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

func (ep *endpoint) Clone() fwkdl.AttributeMap {
	return ep.AttributeMap.Clone()
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

	if !reflect.DeepEqual(a_ep.EndpointMetadata, b_ep.EndpointMetadata) {
		return false
	}
	if !reflect.DeepEqual(a_ep.Metrics, b_ep.Metrics) {
		return false
	}

	// Compare keys and values in AttributeMap for both endpoints. DeepEqual is not used here because the order of keys may differ.
	a_keys := a_ep.Keys()
	b_keys := b_ep.Keys()
	if len(a_keys) != len(b_keys) {
		return false
	}

	for _, k := range a_keys {
		v1, ok1 := a_ep.Get(k)
		v2, ok2 := b_ep.Get(k)
		if !ok1 || !ok2 || !reflect.DeepEqual(v1, v2) {
			return false
		}
	}

	return true
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
