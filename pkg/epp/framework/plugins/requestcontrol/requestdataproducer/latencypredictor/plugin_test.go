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

package latencypredictor

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"

	reqcommon "sigs.k8s.io/gateway-api-inference-extension/pkg/common/request"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkrh "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
	fwksched "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
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
	key := fmt.Sprintf("%.1f", request.KVCachePercentage)
	if pred, ok := m.predictions[key]; ok {
		return pred, nil
	}
	return &latencypredictor.PredictionResponse{TTFT: 0.5, TPOT: 0.03}, nil
}

func (m *mockPredictor) PredictBulk(ctx context.Context, requests []latencypredictor.PredictionRequest) (*latencypredictor.BulkPredictionResponse, error) {
	if m.err != nil {
		return nil, m.err
	}
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

func createTestEndpoint(name string, kvCacheUsage float64, runningRequestsSize, waitingQueueSize int) fwksched.Endpoint {
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
		NamespacedName: types.NamespacedName{
			Name:      name,
			Namespace: "default",
		}},
		&fwkdl.Metrics{
			KVCacheUsagePercent: kvCacheUsage,
			RunningRequestsSize: runningRequestsSize,
			WaitingQueueSize:    waitingQueueSize,
		},
		nil,
	)
}

func createTestLLMRequest(reqID string, ttftSLO, tpotSLO float64) *fwkrh.LLMRequest {
	return createTestLLMRequestWithBody(reqID, ttftSLO, tpotSLO, &fwkrh.RequestBody{
		Completions: &fwkrh.CompletionsRequest{
			Prompt: "test prompt",
		},
	})
}

func createTestChatCompletionsLLMRequest(reqID string, ttftSLO, tpotSLO float64) *fwkrh.LLMRequest {
	return createTestLLMRequestWithBody(reqID, ttftSLO, tpotSLO, &fwkrh.RequestBody{
		ChatCompletions: &fwkrh.ChatCompletionsRequest{
			Messages: []fwkrh.Message{
				{Role: "system", Content: fwkrh.Content{Raw: "You are a helpful assistant."}},
				{Role: "user", Content: fwkrh.Content{Raw: "Tell me a joke."}},
			},
		},
	})
}

func createTestLLMRequestWithBody(reqID string, ttftSLO, tpotSLO float64, body *fwkrh.RequestBody) *fwkrh.LLMRequest {
	headers := make(map[string]string)
	headers[reqcommon.RequestIdHeaderKey] = reqID
	if ttftSLO > 0 {
		headers["x-ttft-slo"] = fmt.Sprintf("%f", ttftSLO)
	}
	if tpotSLO > 0 {
		headers["x-avg-tpot-slo"] = fmt.Sprintf("%f", tpotSLO)
	}

	return &fwkrh.LLMRequest{
		Headers: headers,
		Body:    body,
	}
}

func TestPredictedLatency_TypedName(t *testing.T) {
	predictor := &mockPredictor{}
	cfg := DefaultConfig
	router := NewPredictedLatency(cfg, predictor)

	tn := router.TypedName()
	assert.Equal(t, "latency-predictor-producer", tn.Type, "Type should be latency-predictor")
	assert.Equal(t, "latency-predictor-producer", tn.Name, "Default name should be latency-predictor")
}

func TestPredictedLatency_WithName(t *testing.T) {
	predictor := &mockPredictor{}
	cfg := DefaultConfig
	router := NewPredictedLatency(cfg, predictor)

	customName := "custom-router"
	router = router.WithName(customName)

	tn := router.TypedName()
	assert.Equal(t, "latency-predictor-producer", tn.Type, "Type should remain latency-predictor")
	assert.Equal(t, customName, tn.Name, "Name should be updated to custom name")
}

func TestPredictedLatency_GetPodRunningRequestCount(t *testing.T) {
	tests := []struct {
		name          string
		setupRequests func(*PredictedLatency, fwksched.Endpoint)
		expectedCount int
	}{
		{
			name:          "No running requests",
			setupRequests: func(r *PredictedLatency, p fwksched.Endpoint) {},
			expectedCount: 0,
		},
		{
			name: "One running request",
			setupRequests: func(r *PredictedLatency, p fwksched.Endpoint) {
				podName := types.NamespacedName{
					Name:      p.GetMetadata().NamespacedName.Name,
					Namespace: p.GetMetadata().NamespacedName.Namespace,
				}
				queue := newRequestPriorityQueue()
				queue.Add("req1", 0.04)
				r.runningRequestLists.Store(podName, queue)
			},
			expectedCount: 1,
		},
		{
			name: "Multiple running requests",
			setupRequests: func(r *PredictedLatency, p fwksched.Endpoint) {
				endpointName := types.NamespacedName{
					Name:      p.GetMetadata().NamespacedName.Name,
					Namespace: p.GetMetadata().NamespacedName.Namespace,
				}
				queue := newRequestPriorityQueue()
				queue.Add("req1", 0.04)
				queue.Add("req2", 0.03)
				queue.Add("req3", 0.05)
				r.runningRequestLists.Store(endpointName, queue)
			},
			expectedCount: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			predictor := &mockPredictor{}
			cfg := DefaultConfig
			router := NewPredictedLatency(cfg, predictor)
			pod := createTestEndpoint("test-pod", 0.5, 2, 1)

			tt.setupRequests(router, pod)

			count := router.getEndpointRunningRequestCount(pod)
			assert.Equal(t, tt.expectedCount, count, "Running request count should match expected")
		})
	}
}

func TestPredictedLatency_GetPodMinTPOTSLO(t *testing.T) {
	tests := []struct {
		name          string
		setupRequests func(*PredictedLatency, fwksched.Endpoint)
		expectedSLO   float64
	}{
		{
			name:          "No running requests",
			setupRequests: func(r *PredictedLatency, p fwksched.Endpoint) {},
			expectedSLO:   0.0,
		},
		{
			name: "One running request",
			setupRequests: func(r *PredictedLatency, e fwksched.Endpoint) {
				endpointName := types.NamespacedName{
					Name:      e.GetMetadata().NamespacedName.Name,
					Namespace: e.GetMetadata().NamespacedName.Namespace,
				}
				queue := newRequestPriorityQueue()
				queue.Add("req1", 0.04)
				r.runningRequestLists.Store(endpointName, queue)
			},
			expectedSLO: 0.04,
		},
		{
			name: "Multiple running requests - should return minimum",
			setupRequests: func(r *PredictedLatency, e fwksched.Endpoint) {
				endpointName := types.NamespacedName{
					Name:      e.GetMetadata().NamespacedName.Name,
					Namespace: e.GetMetadata().NamespacedName.Namespace,
				}
				queue := newRequestPriorityQueue()
				queue.Add("req1", 0.05)
				queue.Add("req2", 0.03) // This is the minimum
				queue.Add("req3", 0.04)
				r.runningRequestLists.Store(endpointName, queue)
			},
			expectedSLO: 0.03,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			predictor := &mockPredictor{}
			cfg := DefaultConfig
			router := NewPredictedLatency(cfg, predictor)
			pod := createTestEndpoint("test-pod", 0.5, 2, 1)

			tt.setupRequests(router, pod)

			minSLO := router.getEndpointMinTPOTSLO(pod)
			assert.InDelta(t, tt.expectedSLO, minSLO, 0.0001, "Min TPOT SLO should match expected")
		})
	}
}

func TestPredictedLatencyFactory(t *testing.T) {
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
				"maxDecodeTokenSamplesForPrediction": 30,
				"sloBufferFactor": 1.2
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
			name:       "invalid samplingMean <= 0",
			pluginName: "bad-sampling-mean",
			jsonParams: `{"samplingMean": -1.0}`,
			expectErr:  true,
		},
		{
			name:       "invalid maxSampledTokens < 0",
			pluginName: "bad-max-tokens",
			jsonParams: `{"maxDecodeTokenSamplesForPrediction": -1}`,
			expectErr:  true,
		},
		{
			name:       "invalid sloBufferFactor <= 0",
			pluginName: "bad-buffer",
			jsonParams: `{"sloBufferFactor": 0}`,
			expectErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handle := utils.NewTestHandle(context.Background())
			rawParams := json.RawMessage(tt.jsonParams)
			plugin, err := PredictedLatencyFactory(tt.pluginName, rawParams, handle)

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

func TestPredictedLatencyFactoryInvalidJSON(t *testing.T) {
	invalidTests := []struct {
		name       string
		jsonParams string
	}{
		{
			name:       "malformed JSON",
			jsonParams: `{"samplingMean": 100.0, "maxDecodeTokenSamplesForPrediction":`, // incomplete
		},
		{
			name:       "samplingMean as string",
			jsonParams: `{"samplingMean": "100"}`,
		},
		{
			name:       "maxSampledTokens as float",
			jsonParams: `{"maxDecodeTokenSamplesForPrediction": 20.5}`,
		},
	}

	for _, tt := range invalidTests {
		t.Run(tt.name, func(t *testing.T) {
			handle := utils.NewTestHandle(context.Background())
			rawParams := json.RawMessage(tt.jsonParams)
			plugin, err := PredictedLatencyFactory("test", rawParams, handle)

			assert.Error(t, err)
			assert.Nil(t, plugin)
		})
	}
}

func TestSloContextStoreEviction(t *testing.T) {
	config := DefaultConfig
	config.ContextTTL = 100 * time.Millisecond
	pl := NewPredictedLatency(config, nil)

	requestID := "test-req-id"
	endpointName := types.NamespacedName{Name: "test-model", Namespace: "default"}

	req := &fwkrh.LLMRequest{
		Headers: map[string]string{
			reqcommon.RequestIdHeaderKey: requestID,
		},
	}

	metadata := &fwkdl.EndpointMetadata{
		NamespacedName: endpointName,
	}

	sloCtx := newPredictedLatencyContext(req)
	sloCtx.targetMetadata = metadata
	sloCtx.avgTPOTSLO = 0.05

	pl.setPredictedLatencyContextForRequest(req, sloCtx)

	queue := newRequestPriorityQueue()
	queue.Add(requestID, sloCtx.avgTPOTSLO)
	pl.runningRequestLists.Store(endpointName, queue)

	assert.True(t, queue.Contains(requestID), "Request should be in queue initially")
	item := pl.sloContextStore.Get(requestID)
	assert.NotNil(t, item, "Item should be in cache initially")

	time.Sleep(300 * time.Millisecond)
	item = pl.sloContextStore.Get(requestID)
	assert.Nil(t, item, "Item should have been evicted from cache")
	assert.False(t, queue.Contains(requestID), "Request should be removed from queue via OnEviction")
}
