/*
Copyright 2026 The Kubernetes Authors.

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

package inflightload

import (
	"math"

	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

// TokenEstimator estimates the number of tokens for an LLM request.
type TokenEstimator interface {
	Estimate(request *framework.LLMRequest) int64
}

// SimpleTokenEstimator estimates tokens from character count. tokens = characters / CharactersPerToken.
type SimpleTokenEstimator struct {
	CharactersPerToken float64
	OutputRatio        float64
}

// NewSimpleTokenEstimator returns a SimpleTokenEstimator with default 4.0 chars per token.
func NewSimpleTokenEstimator() TokenEstimator {
	return &SimpleTokenEstimator{
		CharactersPerToken: 4.0,
		OutputRatio:        1.5,
	}
}

// Estimate returns the total estimated token count (input + output) for the request.
// When RequestSizeBytes is set, input tokens are derived from request size (~4 bytes per token)
// to avoid allocations. Otherwise, input tokens are estimated from prompt/message character count
// using CharactersPerToken; output tokens are estimated as inputTokens * OutputRatio.
func (e *SimpleTokenEstimator) Estimate(request *framework.LLMRequest) int64 {
	if request == nil {
		return 0
	}
	// Prefer request body size when available: avoids PlainText() and reduces GC pressure.
	var inputTokens int64
	switch {
	case request.RequestSizeBytes > 0:
		inputTokens = max(int64(request.RequestSizeBytes)/4, 1)
	case request.Body != nil:
		// Fallback: character count from prompt text across all API types
		// (completions, chat/completions, responses, conversations).
		chars := len(request.Body.PromptText())
		inputTokens = int64(math.Max(1, math.Round(float64(chars)/e.CharactersPerToken)))
	default:
		return 0
	}
	outputTokens := int64(math.Round(float64(inputTokens) * e.OutputRatio))
	return inputTokens + outputTokens
}
