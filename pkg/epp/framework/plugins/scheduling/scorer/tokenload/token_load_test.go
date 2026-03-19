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

package tokenload

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwksched "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/saturationdetector/framework/plugins/concurrencydetector"
)

type mockHandle struct {
	fwkplugin.Handle
	plugins map[string]fwkplugin.Plugin
}

func (m *mockHandle) Plugin(name string) fwkplugin.Plugin {
	return m.plugins[name]
}

func TestTokenLoadScorer(t *testing.T) {
	threshold := 1000.0
	mode := concurrencydetector.Tokens
	detector := concurrencydetector.NewDetector(concurrencydetector.Config{
		ConcurrencyMode:     &mode,
		MaxTokenConcurrency: 1000,
	})

	handle := &mockHandle{
		plugins: map[string]fwkplugin.Plugin{
			concurrencydetector.ConcurrencyDetectorType: detector,
		},
	}

	scorer := &TokenLoadScorer{
		handle:         handle,
		queueThreshold: threshold,
	}

	pod1NN := types.NamespacedName{Namespace: "default", Name: "pod1"}
	pod2NN := types.NamespacedName{Namespace: "default", Name: "pod2"}
	pod3NN := types.NamespacedName{Namespace: "default", Name: "pod3"}

	endpoints := []fwksched.Endpoint{
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: pod1NN}, &fwkdl.Metrics{}, nil),
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: pod2NN}, &fwkdl.Metrics{}, nil),
		fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: pod3NN}, &fwkdl.Metrics{}, nil),
	}

	// SimpleTokenEstimator uses 4 characters per token and OutputRatio 1.5.
	// Total tokens = inputTokens + inputTokens * 1.5 = 2.5 * inputTokens.
	// To get 500 tokens, inputTokens = 200. inputTokens = chars / 4 => chars = 800.

	// req for pod2: 500 tokens
	prompt2 := strings.Repeat("a", 800)
	req2 := &fwksched.LLMRequest{RequestId: "req2", Body: &fwksched.LLMRequestBody{Completions: &fwksched.CompletionsRequest{Prompt: prompt2}}}
	result2 := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{endpoints[1]}},
		},
	}
	detector.PreRequest(context.Background(), req2, result2)

	// req for pod3: 1000 tokens
	// To get 1000 tokens, inputTokens = 400. chars = 400 * 4 = 1600.
	prompt3 := strings.Repeat("a", 1600)
	req3 := &fwksched.LLMRequest{RequestId: "req3", Body: &fwksched.LLMRequestBody{Completions: &fwksched.CompletionsRequest{Prompt: prompt3}}}
	result3 := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{endpoints[2]}},
		},
	}
	detector.PreRequest(context.Background(), req3, result3)

	scores := scorer.Score(context.Background(), fwksched.NewCycleState(), &fwksched.LLMRequest{}, endpoints)

	assert.InDelta(t, 1.0, scores[endpoints[0]], 0.0001, "Pod1 (0 tokens) should have score 1.0")
	assert.InDelta(t, 0.5, scores[endpoints[1]], 0.0001, "Pod2 (500 tokens) should have score 0.5")
	assert.InDelta(t, 0.0, scores[endpoints[2]], 0.0001, "Pod3 (1000 tokens) should have score 0.0")
}
