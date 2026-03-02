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

package predictedlatency

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/sidecars/latencypredictorasync"
)

func TestBulkPredictWithMetrics(t *testing.T) {
	mockPredictor := &mockPredictor{
		predictions: map[string]*latencypredictor.PredictionResponse{
			"0.5": {TTFT: 0.5, TPOT: 0.03},
			"0.6": {TTFT: 0.6, TPOT: 0.04},
		},
	}

	metricsStates := []*fwkdl.Metrics{
		{KVCacheUsagePercent: 0.5},
		{KVCacheUsagePercent: 0.6},
	}
	pods := []*fwkdl.EndpointMetadata{
		{
			NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
		},
		{
			NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod2"},
		},
	}
	prompts := []string{"prompt1", "prompt2"}
	generatedTokenCounts := []int{1, 1}
	prefixCacheScores := []float64{0.0, 0.0}

	results, err := bulkPredictWithMetrics(context.Background(), nil, mockPredictor, metricsStates, "", pods, prompts, generatedTokenCounts, prefixCacheScores)

	assert.NoError(t, err)
	assert.Len(t, results, 2)
	assert.Equal(t, 0.5, results[0].TTFT)
	assert.Equal(t, 0.03, results[0].TPOT)
	assert.Equal(t, 0.6, results[1].TTFT)
	assert.Equal(t, 0.04, results[1].TPOT)
}

func TestBulkPredictWithMetrics_Error(t *testing.T) {
	mockPredictor := &mockPredictor{
		err: errors.New("prediction failed"),
	}

	metricsStates := []*fwkdl.Metrics{
		{KVCacheUsagePercent: 0.5},
	}
	pods := []*fwkdl.EndpointMetadata{
		{
			NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
		},
	}
	prompts := []string{"prompt1"}
	generatedTokenCounts := []int{1}
	prefixCacheScores := []float64{0.0}

	results, err := bulkPredictWithMetrics(context.Background(), nil, mockPredictor, metricsStates, "", pods, prompts, generatedTokenCounts, prefixCacheScores)

	assert.Error(t, err)
	assert.Nil(t, results)
}

func TestBulkPredictWithMetrics_InputMismatch(t *testing.T) {
	mockPredictor := &mockPredictor{}
	metricsStates := []*fwkdl.Metrics{{}}
	pods := []*fwkdl.EndpointMetadata{
		{
			NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
		},
	}
	prompts := []string{"prompt1", "prompt2"} // Mismatch length
	generatedTokenCounts := []int{1}
	prefixCacheScores := []float64{0.0}

	results, err := bulkPredictWithMetrics(context.Background(), nil, mockPredictor, metricsStates, "", pods, prompts, generatedTokenCounts, prefixCacheScores)

	assert.Error(t, err)
	assert.Nil(t, results)
	assert.True(t, strings.Contains(err.Error(), "input slice lengths must match"))
}

func TestBulkPredictWithMetrics_WithPredictedLatencyCtx(t *testing.T) {
	mockPredictor := &mockPredictor{
		predictions: map[string]*latencypredictor.PredictionResponse{
			"0.5": {TTFT: 0.5, TPOT: 0.03},
		},
	}

	metricsStates := []*fwkdl.Metrics{
		{KVCacheUsagePercent: 0.5},
	}
	pods := []*fwkdl.EndpointMetadata{
		{
			NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
		},
	}
	prompts := []string{"prompt1"}
	generatedTokenCounts := []int{1}
	prefixCacheScores := []float64{0.0}

	plCtx := &predictedLatencyCtx{
		schedulingRequest: schedulingtypes.LLMRequest{
			TargetModel: "test-model",
		},
		incomingModelName: "incoming-model",
	}

	results, err := bulkPredictWithMetrics(context.Background(), plCtx, mockPredictor, metricsStates, "", pods, prompts, generatedTokenCounts, prefixCacheScores)

	assert.NoError(t, err)
	assert.Len(t, results, 1)
	assert.Equal(t, 0.5, results[0].TTFT)
	assert.Equal(t, 0.03, results[0].TPOT)
}

func TestBulkPredictWithMetrics_ChatCompletionsPrompt(t *testing.T) {
	mp := &mockPredictor{
		predictions: map[string]*latencypredictor.PredictionResponse{
			"0.5": {TTFT: 0.5, TPOT: 0.03},
		},
	}

	metricsStates := []*fwkdl.Metrics{{KVCacheUsagePercent: 0.5}}
	pods := []*fwkdl.EndpointMetadata{
		{NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"}},
	}

	chatBody := &schedulingtypes.LLMRequestBody{
		ChatCompletions: &schedulingtypes.ChatCompletionsRequest{
			Messages: []schedulingtypes.Message{
				{Role: "user", Content: schedulingtypes.Content{Raw: "Hello world"}},
			},
		},
	}
	prompts := []string{chatBody.PromptText()}
	generatedTokenCounts := []int{1}
	prefixCacheScores := []float64{0.0}

	results, err := bulkPredictWithMetrics(context.Background(), nil, mp, metricsStates, "", pods, prompts, generatedTokenCounts, prefixCacheScores)

	assert.NoError(t, err)
	assert.Len(t, results, 1)
	assert.Equal(t, 0.5, results[0].TTFT)
}

func TestBulkPredictWithMetrics_NilMetricsState(t *testing.T) {
	mockPredictor := &mockPredictor{}
	metricsStates := []*fwkdl.Metrics{nil} // Nil metrics state
	pods := []*fwkdl.EndpointMetadata{
		{
			NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
		},
	}
	prompts := []string{"prompt1"}
	generatedTokenCounts := []int{1}
	prefixCacheScores := []float64{0.0}

	results, err := bulkPredictWithMetrics(context.Background(), nil, mockPredictor, metricsStates, "", pods, prompts, generatedTokenCounts, prefixCacheScores)

	assert.Error(t, err)
	assert.Nil(t, results)
	assert.True(t, strings.Contains(err.Error(), "metrics state at index 0 cannot be nil"))
}
