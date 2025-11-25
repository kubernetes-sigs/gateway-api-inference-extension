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

package slo_aware_router

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/sidecars/latencypredictorasync"
)

func TestBulkPredictWithMetrics(t *testing.T) {
	mockPredictor := &mockPredictor{
		predictions: map[string]*latencypredictor.PredictionResponse{
			"0.5": {TTFT: 0.5, TPOT: 0.03},
			"0.6": {TTFT: 0.6, TPOT: 0.04},
		},
	}

	metricsStates := []*backendmetrics.MetricsState{
		{KVCacheUsagePercent: 0.5},
		{KVCacheUsagePercent: 0.6},
	}
	prompts := []string{"prompt1", "prompt2"}
	generatedTokenCounts := []int{1, 1}
	prefixCacheScores := []float64{0.0, 0.0}

	results, err := bulkPredictWithMetrics(context.Background(), mockPredictor, metricsStates, prompts, generatedTokenCounts, prefixCacheScores)

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

	metricsStates := []*backendmetrics.MetricsState{
		{KVCacheUsagePercent: 0.5},
	}
	prompts := []string{"prompt1"}
	generatedTokenCounts := []int{1}
	prefixCacheScores := []float64{0.0}

	results, err := bulkPredictWithMetrics(context.Background(), mockPredictor, metricsStates, prompts, generatedTokenCounts, prefixCacheScores)

	assert.Error(t, err)
	assert.Nil(t, results)
}

func TestBulkPredictWithMetrics_InputMismatch(t *testing.T) {
	mockPredictor := &mockPredictor{}
	metricsStates := []*backendmetrics.MetricsState{{}}
	prompts := []string{"prompt1", "prompt2"} // Mismatch length
	generatedTokenCounts := []int{1}
	prefixCacheScores := []float64{0.0}

	results, err := bulkPredictWithMetrics(context.Background(), mockPredictor, metricsStates, prompts, generatedTokenCounts, prefixCacheScores)

	assert.Error(t, err)
	assert.Nil(t, results)
	assert.True(t, strings.Contains(err.Error(), "input slice lengths must match"))
}

func TestBulkPredictWithMetrics_NilMetricsState(t *testing.T) {
	mockPredictor := &mockPredictor{}
	metricsStates := []*backendmetrics.MetricsState{nil} // Nil metrics state
	prompts := []string{"prompt1"}
	generatedTokenCounts := []int{1}
	prefixCacheScores := []float64{0.0}

	results, err := bulkPredictWithMetrics(context.Background(), mockPredictor, metricsStates, prompts, generatedTokenCounts, prefixCacheScores)

	assert.Error(t, err)
	assert.Nil(t, results)
	assert.True(t, strings.Contains(err.Error(), "metrics state at index 0 cannot be nil"))
}
