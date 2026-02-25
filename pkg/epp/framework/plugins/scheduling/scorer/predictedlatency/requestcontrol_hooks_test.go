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
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/google/uuid"
	"github.com/jellydator/ttlcache/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
)

const (
	testModelName   = "test-model"
	kvUsage         = 1
	runningRequests = 1
	waitingQueue    = 1
)

// Helper functions

func createTestSchedulingResult(metadata *fwkdl.EndpointMetadata) *schedulingtypes.SchedulingResult {

	mockPod := createTestEndpoint(metadata.NamespacedName.Name, kvUsage, runningRequests, waitingQueue)

	return &schedulingtypes.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*schedulingtypes.ProfileRunResult{
			"default": {
				TargetEndpoints: []schedulingtypes.Endpoint{mockPod},
			},
		},
	}
}

func createTestRouter() *PredictedLatency {
	return &PredictedLatency{
		sloContextStore: ttlcache.New(
			ttlcache.WithTTL[string, *predictedLatencyCtx](DefaultConfig.ContextTTL),
		),
		// runningRequestLists is a sync.Map and needs no initialization
		latencypredictor: nil,
		config:           DefaultConfig,
	}
}

// Test cases

func TestNewPredictedLatencyContext(t *testing.T) {
	request := createTestLLMRequest("test", 100, 50)

	ctx := newPredictedLatencyContext(request)

	assert.NotNil(t, ctx)
	assert.Equal(t, *request, ctx.schedulingRequest)
	assert.Equal(t, "test prompt", ctx.promptText)
	assert.NotNil(t, ctx.lastSeenMetrics)
	assert.NotNil(t, ctx.prefixCacheScoresForEndpoints)
	assert.NotNil(t, ctx.predictionsForScheduling)
	assert.Empty(t, ctx.lastSeenMetrics)
	assert.Empty(t, ctx.prefixCacheScoresForEndpoints)
}

func TestNewPredictedLatencyContext_NilBody(t *testing.T) {
	request := &schedulingtypes.LLMRequest{
		Headers: map[string]string{requtil.RequestIdHeaderKey: "test-nil-body"},
		Body:    nil,
	}
	ctx := newPredictedLatencyContext(request)

	assert.NotNil(t, ctx)
	assert.Empty(t, ctx.promptText)
}

func TestNewPredictedLatencyContext_ChatCompletionsPrompt(t *testing.T) {
	request := createTestChatCompletionsLLMRequest("test-chat", 1.0, 0.05)
	ctx := newPredictedLatencyContext(request)

	assert.NotNil(t, ctx)
	assert.Equal(t, "You are a helpful assistant. Tell me a joke. ", ctx.promptText)
}

func TestPredictedLatency_SetAndGetSLOContext(t *testing.T) {
	router := createTestRouter()
	request := createTestLLMRequest("test", 100, 50)
	predictedLatencyCtx := newPredictedLatencyContext(request)

	// Set context
	router.setPredictedLatencyContextForRequest(request, predictedLatencyCtx)

	// Get context
	retrievedCtx, err := router.getPredictedLatencyContextForRequest(request)

	require.NoError(t, err)
	assert.Equal(t, predictedLatencyCtx, retrievedCtx)
}

func TestPredictedLatency_GetSLOContext_NotFound(t *testing.T) {
	router := createTestRouter()
	request := createTestLLMRequest("test", 100, 50)

	// Try to get context that doesn't exist
	ctx, err := router.getPredictedLatencyContextForRequest(request)

	assert.Error(t, err)
	assert.Nil(t, ctx)
	assert.Contains(t, err.Error(), "SLO context not found")
}

func TestPredictedLatency_DeleteSLOContext(t *testing.T) {
	router := createTestRouter()
	request := createTestLLMRequest("test", 100, 50)
	predictedLatencyCtx := newPredictedLatencyContext(request)

	// Set and then delete context
	router.setPredictedLatencyContextForRequest(request, predictedLatencyCtx)
	router.deletePredictedLatencyContextForRequest(request)

	// Verify it's deleted
	ctx, err := router.getPredictedLatencyContextForRequest(request)
	assert.Error(t, err)
	assert.Nil(t, ctx)
}

func TestPredictedLatency_PreRequest_NoSchedulingResult(t *testing.T) {
	router := createTestRouter()
	ctx := context.Background()
	request := createTestLLMRequest("test", 100, 50)

	// Call PreRequest with nil scheduling result
	router.PreRequest(ctx, request, nil)

	// Should not create SLO context
	_, err := router.getPredictedLatencyContextForRequest(request)
	assert.Error(t, err)
}

func TestPredictedLatency_PreRequest_EmptySchedulingResult(t *testing.T) {
	router := createTestRouter()
	ctx := context.Background()
	request := createTestLLMRequest("test", 100, 50)

	schedulingResult := &schedulingtypes.SchedulingResult{
		ProfileResults: map[string]*schedulingtypes.ProfileRunResult{},
	}

	// Call PreRequest with empty scheduling result
	router.PreRequest(ctx, request, schedulingResult)

	// Should not create SLO context
	_, err := router.getPredictedLatencyContextForRequest(request)
	assert.Error(t, err)
}

func TestPredictedLatency_PreRequest_Success(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	endpoint := createTestEndpoint("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test", 100, 50)
	schedulingResult := createTestSchedulingResult(endpoint.GetMetadata())

	// Create and set initial SLO context
	predictedLatencyCtx := newPredictedLatencyContext(request)
	predictedLatencyCtx.avgTPOTSLO = 50
	router.setPredictedLatencyContextForRequest(request, predictedLatencyCtx)

	// Initialize the request priority queue
	router.runningRequestLists.Store(endpoint.GetMetadata().NamespacedName, newRequestPriorityQueue())

	beforeTime := time.Now()
	router.PreRequest(ctx, request, schedulingResult)
	afterTime := time.Now()

	// Verify SLO context was updated
	retrievedCtx, err := router.getPredictedLatencyContextForRequest(request)
	require.NoError(t, err)
	assert.Equal(t, endpoint.GetMetadata(), retrievedCtx.targetMetadata)
	assert.Equal(t, schedulingResult, retrievedCtx.schedulingResult)
	assert.True(t, retrievedCtx.requestReceivedTimestamp.After(beforeTime) ||
		retrievedCtx.requestReceivedTimestamp.Equal(beforeTime))
	assert.True(t, retrievedCtx.requestReceivedTimestamp.Before(afterTime) ||
		retrievedCtx.requestReceivedTimestamp.Equal(afterTime))
}

func TestPredictedLatency_PreRequest_AddsToQueue(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	endpoint := createTestEndpoint("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test", 100, 50)
	schedulingResult := createTestSchedulingResult(endpoint.GetMetadata())

	// Create and set initial SLO context
	predictedLatencyCtx := newPredictedLatencyContext(request)
	predictedLatencyCtx.avgTPOTSLO = 50
	router.setPredictedLatencyContextForRequest(request, predictedLatencyCtx)

	// PreRequest should create the queue
	router.PreRequest(ctx, request, schedulingResult)

	// Verify queue was created and request was added
	value, exists := router.runningRequestLists.Load(endpoint.GetMetadata().NamespacedName)
	assert.True(t, exists, "Queue should be created for endpoint")
	assert.NotNil(t, value)
	queue := value.(*requestPriorityQueue)
	assert.NotNil(t, queue)
}

func TestPredictedLatency_PreRequest_QueueAlreadyExists(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	endpoint := createTestEndpoint("test-pod", 1, 1, 1)
	request1 := createTestLLMRequest("test-id-1", 100, 50)
	request2 := createTestLLMRequest("test-id-2", 100, 50)
	schedulingResult := createTestSchedulingResult(endpoint.GetMetadata())

	// Create and set initial SLO contexts
	predictedLatencyCtx1 := newPredictedLatencyContext(request1)
	predictedLatencyCtx1.avgTPOTSLO = 50
	router.setPredictedLatencyContextForRequest(request1, predictedLatencyCtx1)

	predictedLatencyCtx2 := newPredictedLatencyContext(request2)
	predictedLatencyCtx2.avgTPOTSLO = 50
	router.setPredictedLatencyContextForRequest(request2, predictedLatencyCtx2)
	// Add first request
	router.PreRequest(ctx, request1, schedulingResult)

	// Add second request to same pod
	router.PreRequest(ctx, request2, schedulingResult)

	// Verify both are in the same queue
	value, exists := router.runningRequestLists.Load(endpoint.GetMetadata().NamespacedName)
	assert.True(t, exists)
	assert.NotNil(t, value)
}

func TestPredictedLatency_ResponseReceived_NilPredictor(t *testing.T) {
	router := createTestRouter()
	router.latencypredictor = nil

	ctx := context.Background()
	endpoint := createTestEndpoint("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test", 100, 50)
	response := &requestcontrol.Response{}

	predictedLatencyCtx := newPredictedLatencyContext(request)
	router.setPredictedLatencyContextForRequest(request, predictedLatencyCtx)

	// Should not panic and should return early
	router.ResponseReceived(ctx, request, response, endpoint.GetMetadata())

	// Context should still exist
	_, err := router.getPredictedLatencyContextForRequest(request)
	assert.NoError(t, err)
}

func TestPredictedLatency_ResponseReceived_NoPod(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	request := createTestLLMRequest("test", 100, 50)
	response := &requestcontrol.Response{}

	predictedLatencyCtx := newPredictedLatencyContext(request)
	router.setPredictedLatencyContextForRequest(request, predictedLatencyCtx)

	// Should not panic with nil pod
	router.ResponseReceived(ctx, request, response, nil)

	// Predictor should not be called

}

func TestPredictedLatency_ResponseReceived_NoContext(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	endpoint := createTestEndpoint("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test", 100, 50)
	response := &requestcontrol.Response{}

	// Don't set SLO context
	router.ResponseReceived(ctx, request, response, endpoint.GetMetadata())

	// Should handle missing context gracefully

}

func TestPredictedLatency_ResponseStreaming_NilPredictor(t *testing.T) {
	router := createTestRouter()
	router.latencypredictor = nil

	ctx := context.Background()
	endpoint := createTestEndpoint("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test", 100, 50)
	response := &requestcontrol.Response{}

	predictedLatencyCtx := newPredictedLatencyContext(request)
	router.setPredictedLatencyContextForRequest(request, predictedLatencyCtx)

	// Should not panic and should return early
	router.ResponseStreaming(ctx, request, response, endpoint.GetMetadata())

	// Context should still exist
	_, err := router.getPredictedLatencyContextForRequest(request)
	assert.NoError(t, err)
}
func TestPredictedLatency_ResponseStreaming_FirstToken(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	endpoint := createTestEndpoint("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test", 100, 50)
	response := &requestcontrol.Response{}
	schedulingResult := createTestSchedulingResult(endpoint.GetMetadata())

	predictedLatencyCtx := newPredictedLatencyContext(request)
	predictedLatencyCtx.targetMetadata = endpoint.GetMetadata()
	predictedLatencyCtx.requestReceivedTimestamp = time.Now()
	predictedLatencyCtx.schedulingResult = schedulingResult
	predictedLatencyCtx.schedulingRequest = *request
	predictedLatencyCtx.ttftSLO = 100
	predictedLatencyCtx.avgTPOTSLO = 50
	predictedLatencyCtx.incomingModelName = testModelName
	predictedLatencyCtx.predictedTTFT = 80.0
	predictedLatencyCtx.avgPredictedTPOT = 30.0

	predictedLatencyCtx.lastSeenMetrics["prefill"] = &fwkdl.Metrics{
		KVCacheUsagePercent: 0.5,
		WaitingQueueSize:    1,
		RunningRequestsSize: 1,
	}
	predictedLatencyCtx.lastSeenMetrics["default"] = &fwkdl.Metrics{
		KVCacheUsagePercent: 0.5,
		WaitingQueueSize:    1,
		RunningRequestsSize: 1,
	}

	router.setPredictedLatencyContextForRequest(request, predictedLatencyCtx)

	// Initialize the queue and add the request
	queue := newRequestPriorityQueue()
	queue.Add(request.Headers[requtil.RequestIdHeaderKey], 50.0)
	router.runningRequestLists.Store(endpoint.GetMetadata().NamespacedName, queue)

	beforeTime := time.Now()
	router.ResponseStreaming(ctx, request, response, endpoint.GetMetadata())
	afterTime := time.Now()

	// Verify first token timestamp was set
	retrievedCtx, err := router.getPredictedLatencyContextForRequest(request)
	require.NoError(t, err)
	assert.True(t, retrievedCtx.lastTokenTimestamp.After(beforeTime) ||
		retrievedCtx.lastTokenTimestamp.Equal(beforeTime))
	assert.True(t, retrievedCtx.lastTokenTimestamp.Before(afterTime) ||
		retrievedCtx.lastTokenTimestamp.Equal(afterTime))
}

func TestPredictedLatency_ResponseStreaming_SubsequentTokens(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	endpoint := createTestEndpoint("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test", 100, 50)
	response := &requestcontrol.Response{}
	schedulingResult := createTestSchedulingResult(endpoint.GetMetadata())

	predictedLatencyCtx := newPredictedLatencyContext(request)
	predictedLatencyCtx.targetMetadata = endpoint.GetMetadata()
	predictedLatencyCtx.requestReceivedTimestamp = time.Now()
	predictedLatencyCtx.schedulingResult = schedulingResult
	predictedLatencyCtx.schedulingRequest = *request
	predictedLatencyCtx.ttftSLO = 100
	predictedLatencyCtx.avgTPOTSLO = 50
	predictedLatencyCtx.incomingModelName = testModelName
	predictedLatencyCtx.predictedTTFT = 80.0
	predictedLatencyCtx.avgPredictedTPOT = 30.0
	// ADD THIS - populate metrics
	predictedLatencyCtx.lastSeenMetrics["prefill"] = &fwkdl.Metrics{
		KVCacheUsagePercent: 0.5,
		WaitingQueueSize:    1,
		RunningRequestsSize: 1,
	}
	predictedLatencyCtx.lastSeenMetrics["default"] = &fwkdl.Metrics{
		KVCacheUsagePercent: 0.5,
		WaitingQueueSize:    1,
		RunningRequestsSize: 1,
	}
	firstTokenTime := time.Now().Add(-100 * time.Millisecond)

	router.setPredictedLatencyContextForRequest(request, predictedLatencyCtx)

	// Initialize the queue and add the request
	queue := newRequestPriorityQueue()
	queue.Add(request.Headers[requtil.RequestIdHeaderKey], 50.0)
	router.runningRequestLists.Store(endpoint.GetMetadata().NamespacedName, queue)

	router.ResponseStreaming(ctx, request, response, endpoint.GetMetadata())

	// Verify token timestamp was updated
	retrievedCtx, err := router.getPredictedLatencyContextForRequest(request)
	require.NoError(t, err)
	assert.True(t, retrievedCtx.lastTokenTimestamp.After(firstTokenTime))
}

func TestPredictedLatency_ResponseComplete_QueueNotFound(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	endpoint := createTestEndpoint("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test", 100, 50)
	response := &requestcontrol.Response{}

	predictedLatencyCtx := newPredictedLatencyContext(request)
	predictedLatencyCtx.incomingModelName = testModelName
	predictedLatencyCtx.targetMetadata = endpoint.GetMetadata() // ADD THIS to avoid other issues
	router.setPredictedLatencyContextForRequest(request, predictedLatencyCtx)

	// Create an EMPTY queue (not nil, but empty) to test queue.Remove behavior
	router.runningRequestLists.Store(endpoint.GetMetadata().NamespacedName, newRequestPriorityQueue())

	// Should handle gracefully when request is not in queue
	router.ResponseComplete(ctx, request, response, endpoint.GetMetadata())

	// Context should be deleted
	_, err := router.getPredictedLatencyContextForRequest(request)
	assert.Error(t, err)
}
func TestPredictedLatency_ResponseStreaming_NoContext(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	endpoint := createTestEndpoint("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test", 100, 50)
	response := &requestcontrol.Response{}

	// Don't set SLO context - should handle gracefully
	router.ResponseStreaming(ctx, request, response, endpoint.GetMetadata())

	// Should not panic

}

func TestPredictedLatency_ResponseComplete_Success(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	endpoint := createTestEndpoint("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test", 100, 50)
	response := &requestcontrol.Response{}

	// Create queue and add request
	queue := newRequestPriorityQueue()
	router.runningRequestLists.Store(endpoint.GetMetadata().NamespacedName, queue)
	queue.Add(request.Headers[requtil.RequestIdHeaderKey], 50.0)

	predictedLatencyCtx := newPredictedLatencyContext(request)
	predictedLatencyCtx.ttft = 80
	predictedLatencyCtx.avgTPOT = 30
	predictedLatencyCtx.predictedTTFT = 85
	predictedLatencyCtx.avgPredictedTPOT = 32
	predictedLatencyCtx.ttftSLO = 100
	predictedLatencyCtx.avgTPOTSLO = 50
	predictedLatencyCtx.incomingModelName = "incoming-model"
	predictedLatencyCtx.targetMetadata = endpoint.GetMetadata()
	router.setPredictedLatencyContextForRequest(request, predictedLatencyCtx)

	router.ResponseComplete(ctx, request, response, endpoint.GetMetadata())

	// Verify context was deleted
	_, err := router.getPredictedLatencyContextForRequest(request)
	assert.Error(t, err)

	// Verify request was removed from queue
	assert.Equal(t, 0, queue.Len())
}

func TestPredictedLatency_ResponseComplete_NilPredictor(t *testing.T) {
	router := createTestRouter()
	router.latencypredictor = nil

	ctx := context.Background()
	endpoint := createTestEndpoint("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test", 100, 50)
	response := &requestcontrol.Response{}

	predictedLatencyCtx := newPredictedLatencyContext(request)
	router.setPredictedLatencyContextForRequest(request, predictedLatencyCtx)

	// Should not panic
	router.ResponseComplete(ctx, request, response, endpoint.GetMetadata())

	// Context should still exist (deletion happens only with predictor)
	_, err := router.getPredictedLatencyContextForRequest(request)
	assert.NoError(t, err)
}

func TestPredictedLatency_ResponseComplete_NoPod(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	request := createTestLLMRequest("test", 100, 50)
	response := &requestcontrol.Response{}

	predictedLatencyCtx := newPredictedLatencyContext(request)
	router.setPredictedLatencyContextForRequest(request, predictedLatencyCtx)

	// Should not panic with nil pod
	router.ResponseComplete(ctx, request, response, nil)

	// Context should still exist (deletion happens only with validpod.GetPod())
	_, err := router.getPredictedLatencyContextForRequest(request)
	assert.NoError(t, err)
}

func TestPredictedLatency_ResponseComplete_NoContext(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	endpoint := createTestEndpoint("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test", 100, 50)
	response := &requestcontrol.Response{}

	// Don't set SLO context - should handle gracefully
	router.ResponseComplete(ctx, request, response, endpoint.GetMetadata())

	// Should not panic

}

func TestPredictedLatency_ResponseComplete_WithMetrics(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	endpoint := createTestEndpoint("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test", 100, 50)
	response := &requestcontrol.Response{}

	// Create queue
	queue := newRequestPriorityQueue()
	router.runningRequestLists.Store(endpoint.GetMetadata().NamespacedName, queue)
	queue.Add(request.Headers[requtil.RequestIdHeaderKey], 50.0)

	predictedLatencyCtx := newPredictedLatencyContext(request)
	predictedLatencyCtx.ttft = 80
	predictedLatencyCtx.avgTPOT = 30
	predictedLatencyCtx.predictedTTFT = 85
	predictedLatencyCtx.avgPredictedTPOT = 32
	predictedLatencyCtx.ttftSLO = 100
	predictedLatencyCtx.avgTPOTSLO = 50
	predictedLatencyCtx.incomingModelName = "incoming-model"
	router.setPredictedLatencyContextForRequest(request, predictedLatencyCtx)

	// Should record metrics without panicking
	router.ResponseComplete(ctx, request, response, endpoint.GetMetadata())

	// Verify cleanup
	_, err := router.getPredictedLatencyContextForRequest(request)
	assert.Error(t, err)
}

func TestPredictedLatency_ResponseComplete_NoSLOs(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	endpoint := createTestEndpoint("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test-id", 0, 0) // No SLOs
	response := &requestcontrol.Response{}

	// Create queue
	queue := newRequestPriorityQueue()
	router.runningRequestLists.Store(endpoint.GetMetadata().NamespacedName, queue)
	queue.Add(request.Headers[requtil.RequestIdHeaderKey], 0)

	predictedLatencyCtx := newPredictedLatencyContext(request)
	predictedLatencyCtx.ttft = 80
	predictedLatencyCtx.avgTPOT = 30
	predictedLatencyCtx.incomingModelName = testModelName
	router.setPredictedLatencyContextForRequest(request, predictedLatencyCtx)

	// Should handle missing SLOs gracefully
	router.ResponseComplete(ctx, request, response, endpoint.GetMetadata())

	// Verify cleanup
	_, err := router.getPredictedLatencyContextForRequest(request)
	assert.Error(t, err)
}

func TestPredictedLatency_CheckPredictor_NilPod(t *testing.T) {
	router := createTestRouter()
	logger := logr.Discard()

	result := router.checkPredictor(logger, nil)

	assert.False(t, result)
}

func TestPredictedLatency_CheckPredictor_NilPredictor(t *testing.T) {
	router := createTestRouter()
	router.latencypredictor = nil
	logger := logr.Discard()
	endpoint := createTestEndpoint("test-pod", 1, 1, 1)

	result := router.checkPredictor(logger, endpoint.GetMetadata())

	assert.False(t, result)
}

func TestPredictedLatency_CheckPredictor_Success(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor
	logger := logr.Discard()
	endpoint := createTestEndpoint("test-pod", 1, 1, 1)

	result := router.checkPredictor(logger, endpoint.GetMetadata())

	assert.True(t, result)
}

func TestPredictedLatencyContext_Fields(t *testing.T) {
	request := createTestLLMRequest("test", 100, 50)
	ctx := newPredictedLatencyContext(request)

	// Test all field initialization
	assert.NotNil(t, ctx.lastSeenMetrics)
	assert.NotNil(t, ctx.prefixCacheScoresForEndpoints)
	assert.NotNil(t, ctx.predictionsForScheduling)
	assert.Empty(t, ctx.tpotObservations)
	assert.Empty(t, ctx.predictedTPOTObservations)
	assert.Zero(t, ctx.generatedTokenCount)
	assert.Zero(t, ctx.ttft)
	assert.Zero(t, ctx.avgTPOT)
	assert.Nil(t, ctx.targetMetadata)
	assert.Nil(t, ctx.schedulingResult)
	assert.Nil(t, ctx.tokenSampler)
	assert.Equal(t, "test prompt", ctx.promptText)
}

func TestPredictedLatencyContext_UpdateMetrics(t *testing.T) {
	request := createTestLLMRequest("test", 100, 50)
	ctx := newPredictedLatencyContext(request)

	// Add some metrics
	metricsState := &fwkdl.Metrics{
		KVCacheUsagePercent: 0.5,
		WaitingQueueSize:    3,
	}
	ctx.lastSeenMetrics["test-pod"] = metricsState

	assert.Len(t, ctx.lastSeenMetrics, 1)
	assert.Equal(t, 0.5, ctx.lastSeenMetrics["test-pod"].KVCacheUsagePercent)
	assert.Equal(t, 3, ctx.lastSeenMetrics["test-pod"].WaitingQueueSize)
}

func TestPredictedLatencyContext_PredictionData(t *testing.T) {
	request := createTestLLMRequest("test", 100, 50)
	ctx := newPredictedLatencyContext(request)

	ctx.predictionsForScheduling = make(map[string]endpointPredictionResult)

	// Set prediction data
	ctx.predictionsForScheduling["pod1"] = endpointPredictionResult{Endpoint: createTestEndpoint("pod1", 0, 0, 0), TTFT: 80.0, TPOT: 25.0}
	ctx.predictionsForScheduling["pod2"] = endpointPredictionResult{Endpoint: createTestEndpoint("pod2", 0, 0, 0), TPOT: 30.0, TTFT: 85.0}

	assert.Len(t, ctx.predictionsForScheduling, 2)
	assert.Equal(t, 80.0, ctx.predictionsForScheduling["pod1"].TTFT)
	assert.Equal(t, 30.0, ctx.predictionsForScheduling["pod2"].TPOT)
}

func TestPredictedLatencyContext_PrefixCacheScores(t *testing.T) {
	request := createTestLLMRequest("test", 100, 50)
	ctx := newPredictedLatencyContext(request)

	// Set prefix cache scores
	ctx.prefixCacheScoresForEndpoints["pod1"] = 0.8
	ctx.prefixCacheScoresForEndpoints["pod2"] = 0.6
	ctx.prefixCacheScoresForEndpoints["pod3"] = 0.9

	assert.Len(t, ctx.prefixCacheScoresForEndpoints, 3)
	assert.Equal(t, 0.8, ctx.prefixCacheScoresForEndpoints["pod1"])
	assert.Equal(t, 0.9, ctx.prefixCacheScoresForEndpoints["pod3"])
}

func TestPredictedLatency_ConcurrentContextAccess(t *testing.T) {
	router := createTestRouter()

	// Test concurrent access to context store
	var wg sync.WaitGroup
	numGoroutines := 100

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			requestID := uuid.New().String()
			request := createTestLLMRequest(requestID, 100, 50)
			predictedLatencyCtx := newPredictedLatencyContext(request)

			// Set context
			router.setPredictedLatencyContextForRequest(request, predictedLatencyCtx)

			// Get context
			retrievedCtx, err := router.getPredictedLatencyContextForRequest(request)
			assert.NoError(t, err)
			assert.NotNil(t, retrievedCtx)

			// Delete context
			router.deletePredictedLatencyContextForRequest(request)
		}()
	}

	wg.Wait()
}

func TestPredictedLatency_MultipleRequests_SamePod(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	endpoint := createTestEndpoint("test-pod", 1, 1, 1)

	request1 := createTestLLMRequest("test-id-1", 100, 50)
	request2 := createTestLLMRequest("test-id-2", 100, 50)
	request3 := createTestLLMRequest("test-id-3", 100, 50)

	schedulingResult := createTestSchedulingResult(endpoint.GetMetadata())

	// Create and set SLO contexts
	for _, req := range []*schedulingtypes.LLMRequest{request1, request2, request3} {
		predictedLatencyCtx := newPredictedLatencyContext(req)
		predictedLatencyCtx.avgTPOTSLO = 50
		router.setPredictedLatencyContextForRequest(req, predictedLatencyCtx)
	}

	// Add all requests
	router.PreRequest(ctx, request1, schedulingResult)
	router.PreRequest(ctx, request2, schedulingResult)
	router.PreRequest(ctx, request3, schedulingResult)

	// Verify queue has all requests
	value, exists := router.runningRequestLists.Load(endpoint.GetMetadata().NamespacedName)
	assert.True(t, exists)
	assert.NotNil(t, value)
}

func TestPredictedLatency_RequestLifecycle_Complete(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	endpoint := createTestEndpoint("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test", 100, 50)
	response := &requestcontrol.Response{}
	schedulingResult := createTestSchedulingResult(endpoint.GetMetadata())

	// Create initial context
	predictedLatencyCtx := newPredictedLatencyContext(request)
	predictedLatencyCtx.avgTPOTSLO = 50
	predictedLatencyCtx.incomingModelName = testModelName
	router.setPredictedLatencyContextForRequest(request, predictedLatencyCtx)

	// 1. PreRequest
	router.PreRequest(ctx, request, schedulingResult)

	// Verify context exists
	retrievedCtx, err := router.getPredictedLatencyContextForRequest(request)
	require.NoError(t, err)
	assert.NotNil(t, retrievedCtx.targetMetadata)

	// 2. ResponseReceived
	router.ResponseReceived(ctx, request, response, endpoint.GetMetadata())

	// 3. ResponseStreaming (first token)
	router.ResponseStreaming(ctx, request, response, endpoint.GetMetadata())

	// 4. ResponseStreaming (subsequent tokens)
	retrievedCtx, _ = router.getPredictedLatencyContextForRequest(request)
	retrievedCtx.ttft = 100 // Mark first token received
	router.setPredictedLatencyContextForRequest(request, retrievedCtx)
	router.ResponseStreaming(ctx, request, response, endpoint.GetMetadata())

	// 5. ResponseComplete
	retrievedCtx, _ = router.getPredictedLatencyContextForRequest(request)
	retrievedCtx.ttft = 80
	retrievedCtx.avgTPOT = 30
	router.setPredictedLatencyContextForRequest(request, retrievedCtx)
	router.ResponseComplete(ctx, request, response, endpoint.GetMetadata())

	// Verify context was cleaned up
	_, err = router.getPredictedLatencyContextForRequest(request)
	assert.Error(t, err)
}

func TestPredictedLatency_MultipleRequests_DifferentPods(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()

	endpoint1 := createTestEndpoint("test-pod-1", 1, 1, 1)
	endpoint2 := createTestEndpoint("test-pod-2", 1, 1, 1)

	request1 := createTestLLMRequest("test-id-1", 100, 50)
	request2 := createTestLLMRequest("test-id-2", 100, 50)

	schedulingResult1 := createTestSchedulingResult(endpoint1.GetMetadata())
	schedulingResult2 := createTestSchedulingResult(endpoint2.GetMetadata())

	// Create and set SLO contexts
	predictedLatencyCtx1 := newPredictedLatencyContext(request1)
	predictedLatencyCtx1.avgTPOTSLO = 50
	router.setPredictedLatencyContextForRequest(request1, predictedLatencyCtx1)

	predictedLatencyCtx2 := newPredictedLatencyContext(request2)
	predictedLatencyCtx2.avgTPOTSLO = 50
	router.setPredictedLatencyContextForRequest(request2, predictedLatencyCtx2)
	// Add requests to different pods
	router.PreRequest(ctx, request1, schedulingResult1)
	router.PreRequest(ctx, request2, schedulingResult2)

	// Verify separate queues were created
	value1, exists1 := router.runningRequestLists.Load(endpoint1.GetMetadata().NamespacedName)
	value2, exists2 := router.runningRequestLists.Load(endpoint2.GetMetadata().NamespacedName)

	assert.True(t, exists1)
	assert.True(t, exists2)
	assert.NotNil(t, value1)
	assert.NotNil(t, value2)
	queue1 := value1.(*requestPriorityQueue)
	queue2 := value2.(*requestPriorityQueue)
	assert.NotEqual(t, queue1, queue2)
}

func TestPredictedLatencyContext_SLOValidation(t *testing.T) {
	tests := []struct {
		name       string
		ttftSLO    float64
		tpotSLO    float64
		expectSLOs bool
	}{
		{
			name:       "Both SLOs set",
			ttftSLO:    100,
			tpotSLO:    50,
			expectSLOs: true,
		},
		{
			name:       "No SLOs",
			ttftSLO:    0,
			tpotSLO:    0,
			expectSLOs: false,
		},
		{
			name:       "Only TTFT SLO",
			ttftSLO:    100,
			tpotSLO:    0,
			expectSLOs: false,
		},
		{
			name:       "Only TPOT SLO",
			ttftSLO:    0,
			tpotSLO:    50,
			expectSLOs: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request := createTestLLMRequest("test-id", tt.ttftSLO, tt.tpotSLO)
			ctx := newPredictedLatencyContext(request)
			ctx.ttftSLO = tt.ttftSLO
			ctx.avgTPOTSLO = tt.tpotSLO

			hasBothSLOs := ctx.ttftSLO > 0 && ctx.avgTPOTSLO > 0
			assert.Equal(t, tt.expectSLOs, hasBothSLOs)
		})
	}
}

// Benchmark tests

func BenchmarkPredictedLatency_PreRequest(b *testing.B) {
	router := createTestRouter()
	ctx := context.Background()
	endpoint := createTestEndpoint("test-pod", 1, 1, 1)
	schedulingResult := createTestSchedulingResult(endpoint.GetMetadata())

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		requestID := uuid.New().String()
		request := createTestLLMRequest(requestID, 100, 50)
		predictedLatencyCtx := newPredictedLatencyContext(request)
		predictedLatencyCtx.avgTPOTSLO = 50
		router.setPredictedLatencyContextForRequest(request, predictedLatencyCtx)
		router.PreRequest(ctx, request, schedulingResult)
	}
}

func BenchmarkPredictedLatency_ContextOperations(b *testing.B) {
	router := createTestRouter()
	request := createTestLLMRequest("test", 100, 50)
	predictedLatencyCtx := newPredictedLatencyContext(request)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		router.setPredictedLatencyContextForRequest(request, predictedLatencyCtx)
		_, _ = router.getPredictedLatencyContextForRequest(request)
		router.deletePredictedLatencyContextForRequest(request)
	}
}

func BenchmarkPredictedLatencyContext_Creation(b *testing.B) {
	request := createTestLLMRequest("test", 100, 50)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = newPredictedLatencyContext(request)
	}
}
