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
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/sidecars/latencypredictorasync"
	"sigs.k8s.io/gateway-api-inference-extension/test/utils"
)

// mockPredictor implements PredictorInterface for testing
type mockPredictor struct {
	predictions map[string]*latencypredictor.PredictionResponse
	err         error
}

func (m *mockPredictor) Predict(ctx context.Context, request latencypredictor.PredictionRequest) (*latencypredictor.PredictionResponse, error) {
	if m.err != nil {
		return nil, m.err
	}
	// Generate a key based on KV cache percentage to return different predictions for different pods
	key := fmt.Sprintf("%.1f", request.KVCachePercentage)
	if pred, ok := m.predictions[key]; ok {
		return pred, nil
	}
	// Default prediction
	return &latencypredictor.PredictionResponse{TTFT: 0.5, TPOT: 0.03}, nil
}

func (m *mockPredictor) PredictBulk(ctx context.Context, requests []latencypredictor.PredictionRequest) (*latencypredictor.BulkPredictionResponse, error) {
	if m.err != nil {
		return nil, m.err
	}
	// Generate a key based on KV cache percentage to return different predictions for different pods
	responses := make([]latencypredictor.PredictionResponse, 0, len(requests))
	for _, request := range requests {
		key := fmt.Sprintf("%.1f", request.KVCachePercentage)
		if pred, ok := m.predictions[key]; ok {
			responses = append(responses, *pred)
		} else {
			return nil, fmt.Errorf("no prediction for key %s", key)
		}
	}
	return &latencypredictor.BulkPredictionResponse{Predictions: responses}, nil
}

func (m *mockPredictor) PredictBulkStrict(ctx context.Context, requests []latencypredictor.PredictionRequest) (*latencypredictor.BulkPredictionResponse, error) {
	if m.err != nil {
		return nil, m.err
	}
	// Generate a key based on KV cache percentage to return different predictions for different pods
	responses := make([]latencypredictor.PredictionResponse, 0, len(requests))
	for _, request := range requests {
		key := fmt.Sprintf("%.1f", request.KVCachePercentage)
		if pred, ok := m.predictions[key]; ok {
			responses = append(responses, *pred)
		} else {
			return nil, fmt.Errorf("no prediction for key %s", key)
		}
	}
	return &latencypredictor.BulkPredictionResponse{Predictions: responses}, nil
}

func (m *mockPredictor) AddTrainingDataBulk(data []latencypredictor.TrainingEntry) error {
	return nil
}

func (m *mockPredictor) AddTrainingData(data latencypredictor.TrainingEntry) error {
	return nil
}

func (m *mockPredictor) HealthCheck() error {
	return nil
}

func (m *mockPredictor) GetServerStatus(ctx context.Context) (*latencypredictor.ServerStatusResponse, error) {
	return &latencypredictor.ServerStatusResponse{}, nil
}

func createTestPod(name string, kvCacheUsage float64, runningRequestsSize, waitingQueueSize int) schedulingtypes.Pod {
	return &schedulingtypes.PodMetrics{
		Pod: &backend.Pod{
			NamespacedName: types.NamespacedName{
				Name:      name,
				Namespace: "default",
			},
		},
		MetricsState: &backendmetrics.MetricsState{
			KVCacheUsagePercent: kvCacheUsage,
			RunningRequestsSize: runningRequestsSize,
			WaitingQueueSize:    waitingQueueSize,
		},
	}
}

func createTestLLMRequest(reqID string, ttftSLO, tpotSLO float64, predictionBased bool) *schedulingtypes.LLMRequest {
	headers := make(map[string]string)
	headers[requtil.RequestIdHeaderKey] = reqID
	if ttftSLO > 0 {
		headers["x-ttft-slo"] = fmt.Sprintf("%f", ttftSLO)
	}
	if tpotSLO > 0 {
		headers["x-avg-tpot-slo"] = fmt.Sprintf("%f", tpotSLO)
	}
	if !predictionBased {
		headers["x-prediction-based-scheduling-off"] = "true"
	}

	return &schedulingtypes.LLMRequest{
		Headers: headers,
		Body: &schedulingtypes.LLMRequestBody{
			Completions: &schedulingtypes.CompletionsRequest{
				Prompt: "test prompt",
			},
		},
	}
}

func TestSLOAwareRouter_Score(t *testing.T) {
	tests := []struct {
		name           string
		predictor      *mockPredictor
		strategy       headroomStrategy
		request        *schedulingtypes.LLMRequest
		pods           []schedulingtypes.Pod
		expectedScores map[string]float64 // Map of pod name to expected score
		expectNil      bool
	}{
		{
			name:      "Prediction-based scheduling disabled",
			predictor: &mockPredictor{},
			strategy:  headroomStrategyLeast,
			request:   createTestLLMRequest("test", 1.0, 0.05, false), // predictionBased = false
			pods: []schedulingtypes.Pod{
				createTestPod("pod1", 0.5, 2, 1), // 50% KV cache, 2 running, 1 waiting
				createTestPod("pod2", 0.7, 3, 2), // 70% KV cache, 3 running, 2 waiting
			},
			expectNil: true,
		},
		{
			name:      "No predictor configured",
			predictor: nil,
			strategy:  headroomStrategyLeast,
			request:   createTestLLMRequest("test", 1.0, 0.05, true),
			pods: []schedulingtypes.Pod{
				createTestPod("pod1", 0.5, 2, 1),
			},
			expectNil: true,
		},
		{
			name: "All pods have positive headroom",
			predictor: &mockPredictor{
				predictions: map[string]*latencypredictor.PredictionResponse{
					"0.5": {TTFT: 0.5, TPOT: 0.03}, // 50% KV cache
					"0.6": {TTFT: 0.6, TPOT: 0.04}, // 60% KV cache
					"0.3": {TTFT: 0.4, TPOT: 0.02}, // 30% KV cache
				},
			},
			strategy: headroomStrategyLeast,
			request:  createTestLLMRequest("test", 1.0, 0.05, true),
			pods: []schedulingtypes.Pod{
				createTestPod("pod1", 0.5, 2, 1), // 50% KV cache
				createTestPod("pod2", 0.6, 3, 2), // 60% KV cache
				createTestPod("pod3", 0.3, 1, 0), // 30% KV cache
			},
			// One pod should be selected with score 1, others 0
			expectedScores: map[string]float64{
				// We can't predict which one due to randomness, but exactly one should be 1
			},
		},
		{
			name: "All pods have negative headroom",
			predictor: &mockPredictor{
				predictions: map[string]*latencypredictor.PredictionResponse{
					"0.8": {TTFT: 1.5, TPOT: 0.08}, // 80% KV cache - high load
					"0.9": {TTFT: 1.8, TPOT: 0.09}, // 90% KV cache - very high load
				},
			},
			strategy: headroomStrategyLeast,
			request:  createTestLLMRequest("test", 1.0, 0.05, true),
			pods: []schedulingtypes.Pod{
				createTestPod("pod1", 0.8, 5, 3), // 80% KV cache, high load
				createTestPod("pod2", 0.9, 6, 4), // 90% KV cache, very high load
			},
			// One pod should still be selected even with negative headroom
			expectedScores: map[string]float64{},
		},
		{
			name: "Mixed positive and negative headroom",
			predictor: &mockPredictor{
				predictions: map[string]*latencypredictor.PredictionResponse{
					"0.3": {TTFT: 0.5, TPOT: 0.03}, // 30% KV cache - Positive headroom
					"0.9": {TTFT: 1.5, TPOT: 0.08}, // 90% KV cache - Negative headroom
				},
			},
			strategy: headroomStrategyLeast,
			request:  createTestLLMRequest("test", 1.0, 0.05, true),
			pods: []schedulingtypes.Pod{
				createTestPod("pod-positive", 0.3, 1, 0), // Low KV cache, positive headroom
				createTestPod("pod-negative", 0.9, 6, 4), // High KV cache, negative headroom
			},
			// With 99% probability, positive headroom pod should be selected
			expectedScores: map[string]float64{},
		},
		{
			name: "Prediction errors - fallback to composite scoring",
			predictor: &mockPredictor{
				err: errors.New("prediction failed"),
			},
			strategy: headroomStrategyLeast,
			request:  createTestLLMRequest("test", 1.0, 0.05, true),
			pods: []schedulingtypes.Pod{
				createTestPod("pod1", 0.5, 2, 1),
				createTestPod("pod2", 0.6, 3, 2),
			},
			// Should fall back to composite-only scoring and select one pod
			expectedScores: map[string]float64{
				// One pod should be selected with score 1, verified in general validation below
			},
		},
		{
			name:      "Empty pod list",
			predictor: &mockPredictor{},
			strategy:  headroomStrategyLeast,
			request:   createTestLLMRequest("test", 1.0, 0.05, true),
			pods:      []schedulingtypes.Pod{},
			// Should return empty scores map
			expectedScores: map[string]float64{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var router *SLOAwareRouter
			cfg := DefaultConfig
			cfg.HeadroomSelectionStrategy = string(tt.strategy)

			var predictor latencypredictor.PredictorInterface
			if tt.predictor != nil {
				predictor = tt.predictor
			} else if tt.name != "No predictor configured" {
				// Keep logic consistent with original test setup if needed,
				// but here we just use what's in tt.predictor which is nil or set.
				predictor = nil
			}

			router = NewSLOAwareRouter(cfg, predictor)

			scores := router.Score(context.Background(), schedulingtypes.NewCycleState(), tt.request, tt.pods)

			if tt.expectNil {
				assert.Nil(t, scores, "Expected nil scores")
				return
			}

			assert.NotNil(t, scores, "Expected non-nil scores")

			// If we have specific expected scores, verify them
			if len(tt.expectedScores) > 0 {
				for _, pod := range tt.pods {
					podName := pod.GetPod().NamespacedName.Name
					if expectedScore, ok := tt.expectedScores[podName]; ok {
						assert.InDelta(t, expectedScore, scores[pod], 0.0001, "Pod %s should have score %f", podName, expectedScore)
					}
				}
			}

			// General validation: exactly one pod should have score 1 (selected), others should have score 0
			// This applies even when predictions fail because we fall back to composite scoring
			if !tt.expectNil && len(tt.pods) > 0 && tt.predictor != nil {
				selectedCount := 0
				for _, score := range scores {
					if score == 1.0 {
						selectedCount++
					} else {
						assert.InDelta(t, 0.0, score, 0.0001, "Non-selected pods should have score 0")
					}
				}
				assert.Equal(t, 1, selectedCount, "Exactly one pod should be selected with score 1")
			}
		})
	}
}

func TestSLOAwareRouter_Strategies(t *testing.T) {
	tests := []struct {
		name     string
		strategy headroomStrategy
	}{
		{
			name:     "HeadroomStrategyLeast",
			strategy: headroomStrategyLeast,
		},
		{
			name:     "HeadroomStrategyMost",
			strategy: headroomStrategyMost,
		},
		{
			name:     "HeadroomStrategyCompositeMost",
			strategy: headroomStrategyCompositeMost,
		},
		{
			name:     "HeadroomStrategyCompositeLeast",
			strategy: headroomStrategyCompositeLeast,
		},
		{
			name:     "HeadroomStrategyCompositeOnly",
			strategy: headroomStrategyCompositeOnly,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			predictor := &mockPredictor{
				predictions: map[string]*latencypredictor.PredictionResponse{
					"0.5": {TTFT: 0.5, TPOT: 0.03},
					"0.6": {TTFT: 0.6, TPOT: 0.04},
					"0.3": {TTFT: 0.4, TPOT: 0.02},
				},
			}
			cfg := DefaultConfig
			cfg.HeadroomSelectionStrategy = string(tt.strategy)
			router := NewSLOAwareRouter(cfg, predictor)

			request := createTestLLMRequest("test", 1.0, 0.05, true)
			pods := []schedulingtypes.Pod{
				createTestPod("pod1", 0.5, 2, 1),
				createTestPod("pod2", 0.6, 3, 2),
				createTestPod("pod3", 0.3, 1, 0),
			}

			scores := router.Score(context.Background(), schedulingtypes.NewCycleState(), request, pods)

			assert.NotNil(t, scores, "Expected non-nil scores for strategy %s", tt.strategy)

			// Verify exactly one pod is selected
			selectedCount := 0
			for _, score := range scores {
				if score == 1.0 {
					selectedCount++
				}
			}
			assert.Equal(t, 1, selectedCount, "Strategy %s should select exactly one pod", tt.strategy)
		})
	}
}

func TestSLOAwareRouter_TypedName(t *testing.T) {
	predictor := &mockPredictor{}
	cfg := DefaultConfig
	cfg.HeadroomSelectionStrategy = string(headroomStrategyLeast)
	router := NewSLOAwareRouter(cfg, predictor)

	tn := router.TypedName()
	assert.Equal(t, "predicted-latency-scorer", tn.Type, "Type should be predicted-latency-scorer")
	assert.Equal(t, "predicted-latency-scorer", tn.Name, "Default name should be predicted-latency-scorer")
}

func TestSLOAwareRouter_WithName(t *testing.T) {
	predictor := &mockPredictor{}
	cfg := DefaultConfig
	cfg.HeadroomSelectionStrategy = string(headroomStrategyLeast)
	router := NewSLOAwareRouter(cfg, predictor)

	customName := "custom-router"
	router = router.WithName(customName)

	tn := router.TypedName()
	assert.Equal(t, "predicted-latency-scorer", tn.Type, "Type should remain predicted-latency-scorer")
	assert.Equal(t, customName, tn.Name, "Name should be updated to custom name")
}

func TestSLOAwareRouter_GetPodRunningRequestCount(t *testing.T) {
	tests := []struct {
		name          string
		setupRequests func(*SLOAwareRouter, schedulingtypes.Pod)
		expectedCount int
	}{
		{
			name:          "No running requests",
			setupRequests: func(r *SLOAwareRouter, p schedulingtypes.Pod) {},
			expectedCount: 0,
		},
		{
			name: "One running request",
			setupRequests: func(r *SLOAwareRouter, p schedulingtypes.Pod) {
				podName := types.NamespacedName{
					Name:      p.GetPod().NamespacedName.Name,
					Namespace: p.GetPod().NamespacedName.Namespace,
				}
				r.runningRequestLists[podName] = newRequestPriorityQueue()
				r.runningRequestLists[podName].Add("req1", 0.04)
			},
			expectedCount: 1,
		},
		{
			name: "Multiple running requests",
			setupRequests: func(r *SLOAwareRouter, p schedulingtypes.Pod) {
				podName := types.NamespacedName{
					Name:      p.GetPod().NamespacedName.Name,
					Namespace: p.GetPod().NamespacedName.Namespace,
				}
				r.runningRequestLists[podName] = newRequestPriorityQueue()
				r.runningRequestLists[podName].Add("req1", 0.04)
				r.runningRequestLists[podName].Add("req2", 0.03)
				r.runningRequestLists[podName].Add("req3", 0.05)
			},
			expectedCount: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			predictor := &mockPredictor{}
			cfg := DefaultConfig
			cfg.HeadroomSelectionStrategy = string(headroomStrategyLeast)
			router := NewSLOAwareRouter(cfg, predictor)
			pod := createTestPod("test-pod", 0.5, 2, 1)

			tt.setupRequests(router, pod)

			count := router.getPodRunningRequestCount(pod)
			assert.Equal(t, tt.expectedCount, count, "Running request count should match expected")
		})
	}
}

func TestSLOAwareRouter_GetPodMinTPOTSLO(t *testing.T) {
	tests := []struct {
		name          string
		setupRequests func(*SLOAwareRouter, schedulingtypes.Pod)
		expectedSLO   float64
	}{
		{
			name:          "No running requests",
			setupRequests: func(r *SLOAwareRouter, p schedulingtypes.Pod) {},
			expectedSLO:   0.0,
		},
		{
			name: "One running request",
			setupRequests: func(r *SLOAwareRouter, p schedulingtypes.Pod) {
				podName := types.NamespacedName{
					Name:      p.GetPod().NamespacedName.Name,
					Namespace: p.GetPod().NamespacedName.Namespace,
				}
				r.runningRequestLists[podName] = newRequestPriorityQueue()
				r.runningRequestLists[podName].Add("req1", 0.04)
			},
			expectedSLO: 0.04,
		},
		{
			name: "Multiple running requests - should return minimum",
			setupRequests: func(r *SLOAwareRouter, p schedulingtypes.Pod) {
				podName := types.NamespacedName{
					Name:      p.GetPod().NamespacedName.Name,
					Namespace: p.GetPod().NamespacedName.Namespace,
				}
				r.runningRequestLists[podName] = newRequestPriorityQueue()
				// Add in any order - heap will maintain minimum at top
				r.runningRequestLists[podName].Add("req1", 0.05)
				r.runningRequestLists[podName].Add("req2", 0.03) // This is the minimum
				r.runningRequestLists[podName].Add("req3", 0.04)
			},
			expectedSLO: 0.03, // Minimum TPOT (heap guarantees this is at items[0])
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			predictor := &mockPredictor{}
			cfg := DefaultConfig
			cfg.HeadroomSelectionStrategy = string(headroomStrategyLeast)
			router := NewSLOAwareRouter(cfg, predictor)
			pod := createTestPod("test-pod", 0.5, 2, 1)

			tt.setupRequests(router, pod)

			minSLO := router.getPodMinTPOTSLO(pod)
			assert.InDelta(t, tt.expectedSLO, minSLO, 0.0001, "Min TPOT SLO should match expected")
		})
	}
}

func TestSLOAwareRouter_GetPrefixCacheScoreForPod(t *testing.T) {
	tests := []struct {
		name          string
		setupState    func(*schedulingtypes.CycleState)
		expectedScore float64
	}{
		{
			name:          "No prefix cache state",
			setupState:    func(s *schedulingtypes.CycleState) {},
			expectedScore: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			predictor := &mockPredictor{}
			cfg := DefaultConfig
			cfg.HeadroomSelectionStrategy = string(headroomStrategyLeast)
			router := NewSLOAwareRouter(cfg, predictor)

			state := schedulingtypes.NewCycleState()
			tt.setupState(state)

			pod := createTestPod("test-pod", 0.5, 2, 1)

			score := router.getPrefixCacheScoreForPod(context.Background(), state, pod)
			assert.InDelta(t, tt.expectedScore, score, 0.0001, "Prefix cache score should match expected")
		})
	}
}

func TestSLOAwareRouterFactory(t *testing.T) {
	tests := []struct {
		name       string
		pluginName string
		jsonParams string
		expectErr  bool
	}{
		{
			name:       "valid config with all fields",
			pluginName: "full-config",
			jsonParams: `{
				"samplingMean": 150.0,
				"maxSampledTokens": 30,
				"sloBufferFactor": 1.2,
				"negHeadroomTTFTWeight": 0.7,
				"negHeadroomTPOTWeight": 0.3,
				"headroomTTFTWeight": 0.9,
				"headroomTPOTWeight": 0.1,
				"headroomSelectionStrategy": "least",
				"compositeKVWeight": 1.0,
				"compositeQueueWeight": 0.8,
				"compositePrefixWeight": 0.5,
				"epsilonExploreSticky": 0.02,
				"epsilonExploreNeg": 0.03,
				"affinityGateTau": 0.85,
				"affinityGateTauGlobal": 0.95,
				"selectionMode": "linear"
			}`,
			expectErr: false,
		},
		{
			name:       "valid config with minimal override (uses defaults)",
			pluginName: "minimal",
			jsonParams: `{}`,
			expectErr:  false,
		},
		{
			name:       "valid config with composite strategy",
			pluginName: "composite",
			jsonParams: `{
				"headroomSelectionStrategy": "composite-least",
				"selectionMode": "linear"
			}`,
			expectErr: false,
		},
		{
			name:       "invalid samplingMean <= 0",
			pluginName: "bad-sampling-mean",
			jsonParams: `{"samplingMean": -1.0}`,
			expectErr:  true,
		},
		{
			name:       "invalid maxSampledTokens <= 0",
			pluginName: "bad-max-tokens",
			jsonParams: `{"maxSampledTokens": 0}`,
			expectErr:  true,
		},
		{
			name:       "invalid sloBufferFactor <= 0",
			pluginName: "bad-buffer",
			jsonParams: `{"sloBufferFactor": 0}`,
			expectErr:  true,
		},
		{
			name:       "negative headroom weight",
			pluginName: "neg-weight",
			jsonParams: `{"negHeadroomTTFTWeight": -0.1}`,
			expectErr:  true,
		},
		{
			name:       "epsilonExploreSticky > 1",
			pluginName: "epsilon-too-high",
			jsonParams: `{"epsilonExploreSticky": 1.1}`,
			expectErr:  true,
		},
		{
			name:       "epsilonExploreNeg < 0",
			pluginName: "epsilon-negative",
			jsonParams: `{"epsilonExploreNeg": -0.1}`,
			expectErr:  true,
		},
		{
			name:       "affinityGateTau out of (0,1]",
			pluginName: "tau-invalid",
			jsonParams: `{"affinityGateTau": 1.5}`,
			expectErr:  true,
		},
		{
			name:       "affinityGateTauGlobal <= 0",
			pluginName: "tau-global-zero",
			jsonParams: `{"affinityGateTauGlobal": 0}`,
			expectErr:  true,
		},
		{
			name:       "multiple validation errors",
			pluginName: "multi-error",
			jsonParams: `{
				"samplingMean": -1,
				"maxSampledTokens": 0,
				"epsilonExploreSticky": 2.0,
				"headroomSelectionStrategy": "unknown"
			}`,
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handle := utils.NewTestHandle(context.Background())
			rawParams := json.RawMessage(tt.jsonParams)
			plugin, err := SLOAwareRouterFactory(tt.pluginName, rawParams, handle)

			if tt.expectErr {
				assert.Error(t, err)
				assert.Nil(t, plugin)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, plugin)
			}
		})
	}
}

func TestSLOAwareRouterFactoryInvalidJSON(t *testing.T) {
	invalidTests := []struct {
		name       string
		jsonParams string
	}{
		{
			name:       "malformed JSON",
			jsonParams: `{"samplingMean": 100.0, "maxSampledTokens":`, // incomplete
		},
		{
			name:       "samplingMean as string",
			jsonParams: `{"samplingMean": "100"}`,
		},
		{
			name:       "maxSampledTokens as float",
			jsonParams: `{"maxSampledTokens": 20.5}`,
		},
		{
			name:       "headroomSelectionStrategy as number",
			jsonParams: `{"headroomSelectionStrategy": 123}`,
		},
	}

	for _, tt := range invalidTests {
		t.Run(tt.name, func(t *testing.T) {
			handle := utils.NewTestHandle(context.Background())
			rawParams := json.RawMessage(tt.jsonParams)
			plugin, err := SLOAwareRouterFactory("test", rawParams, handle)

			assert.Error(t, err)
			assert.Nil(t, plugin)
		})
	}
}
