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

package concurrencydetector

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

func (e *SimpleTokenEstimator) Estimate(request *framework.LLMRequest) int64 {
	if request == nil || request.Body == nil {
		return 0
	} // Todo: Should we return 0 or a small number?
	var chars int
	if request.Body.Completions != nil {
		chars = len(request.Body.Completions.Prompt)
	} else if request.Body.ChatCompletions != nil {
		for _, m := range request.Body.ChatCompletions.Messages {
			chars += len(m.Content.PlainText())
		}
	} else {
		chars = 0 // TODO: Is this this right way to handle this?
	}
	inputTokens := int64(math.Max(1, math.Round(float64(chars)/e.CharactersPerToken)))
	outputTokens := int64(math.Round(float64(inputTokens) * e.OutputRatio))
	return inputTokens + outputTokens
}
