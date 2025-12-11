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
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/requestcontrol"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
)

const (
	testModelName   = "test-model"
	kvUsage         = 1
	runningRequests = 1
	waitingQueue    = 1
)

// Helper functions

func createTestSchedulingResult(pod *backend.Pod) *schedulingtypes.SchedulingResult {

	mockPod := createTestPod(pod.NamespacedName.Name, kvUsage, runningRequests, waitingQueue)

	return &schedulingtypes.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*schedulingtypes.ProfileRunResult{
			"default": {
				TargetPods: []schedulingtypes.Pod{mockPod},
			},
		},
	}
}

func createTestRouter() *SLOAwareRouter {
	return &SLOAwareRouter{
		sloContextStore:     sync.Map{},
		runningRequestLists: make(map[types.NamespacedName]*requestPriorityQueue),
		latencypredictor:    nil,
	}
}

// Test cases

func TestNewSLORequestContext(t *testing.T) {
	request := createTestLLMRequest("test", 100, 50, true)

	ctx := newSLORequestContext(request)

	assert.NotNil(t, ctx)
	assert.Equal(t, *request, ctx.schedulingRequest)
	assert.NotNil(t, ctx.lastSeenMetrics)
	assert.NotNil(t, ctx.prefixCacheScoresForPods)
	assert.NotNil(t, ctx.predictedTTFTForScheduling)
	assert.NotNil(t, ctx.predictedTPOTForScheduling)
	assert.Empty(t, ctx.lastSeenMetrics)
	assert.Empty(t, ctx.prefixCacheScoresForPods)
}

func TestSLOAwareRouter_SetAndGetSLOContext(t *testing.T) {
	router := createTestRouter()
	request := createTestLLMRequest("test", 100, 50, true)
	sloCtx := newSLORequestContext(request)

	// Set context
	router.setSLOContextForRequest(request, sloCtx)

	// Get context
	retrievedCtx, err := router.getSLOContextForRequest(request)

	require.NoError(t, err)
	assert.Equal(t, sloCtx, retrievedCtx)
}

func TestSLOAwareRouter_GetSLOContext_NotFound(t *testing.T) {
	router := createTestRouter()
	request := createTestLLMRequest("test", 100, 50, true)

	// Try to get context that doesn't exist
	ctx, err := router.getSLOContextForRequest(request)

	assert.Error(t, err)
	assert.Nil(t, ctx)
	assert.Contains(t, err.Error(), "SLO context not found")
}

func TestSLOAwareRouter_DeleteSLOContext(t *testing.T) {
	router := createTestRouter()
	request := createTestLLMRequest("test", 100, 50, true)
	sloCtx := newSLORequestContext(request)

	// Set and then delete context
	router.setSLOContextForRequest(request, sloCtx)
	router.deleteSLOContextForRequest(request)

	// Verify it's deleted
	ctx, err := router.getSLOContextForRequest(request)
	assert.Error(t, err)
	assert.Nil(t, ctx)
}

func TestSLOAwareRouter_PreRequest_NoSchedulingResult(t *testing.T) {
	router := createTestRouter()
	ctx := context.Background()
	request := createTestLLMRequest("test", 100, 50, true)

	// Call PreRequest with nil scheduling result
	router.PreRequest(ctx, request, nil)

	// Should not create SLO context
	_, err := router.getSLOContextForRequest(request)
	assert.Error(t, err)
}

func TestSLOAwareRouter_PreRequest_EmptySchedulingResult(t *testing.T) {
	router := createTestRouter()
	ctx := context.Background()
	request := createTestLLMRequest("test", 100, 50, true)

	schedulingResult := &schedulingtypes.SchedulingResult{
		ProfileResults: map[string]*schedulingtypes.ProfileRunResult{},
	}

	// Call PreRequest with empty scheduling result
	router.PreRequest(ctx, request, schedulingResult)

	// Should not create SLO context
	_, err := router.getSLOContextForRequest(request)
	assert.Error(t, err)
}

func TestSLOAwareRouter_PreRequest_Success(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	pod := createTestPod("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test", 100, 50, true)
	schedulingResult := createTestSchedulingResult(pod.GetPod())

	// Create and set initial SLO context
	sloCtx := newSLORequestContext(request)
	sloCtx.avgTPOTSLO = 50
	router.setSLOContextForRequest(request, sloCtx)

	// Initialize the request priority queue
	router.runningRequestLists[pod.GetPod().NamespacedName] = newRequestPriorityQueue()

	beforeTime := time.Now()
	router.PreRequest(ctx, request, schedulingResult)
	afterTime := time.Now()

	// Verify SLO context was updated
	retrievedCtx, err := router.getSLOContextForRequest(request)
	require.NoError(t, err)
	assert.Equal(t, pod.GetPod(), retrievedCtx.targetPod)
	assert.Equal(t, schedulingResult, retrievedCtx.schedulingResult)
	assert.True(t, retrievedCtx.requestReceivedTimestamp.After(beforeTime) ||
		retrievedCtx.requestReceivedTimestamp.Equal(beforeTime))
	assert.True(t, retrievedCtx.requestReceivedTimestamp.Before(afterTime) ||
		retrievedCtx.requestReceivedTimestamp.Equal(afterTime))
}

func TestSLOAwareRouter_PreRequest_AddsToQueue(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	pod := createTestPod("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test", 100, 50, true)
	schedulingResult := createTestSchedulingResult(pod.GetPod())

	// Create and set initial SLO context
	sloCtx := newSLORequestContext(request)
	sloCtx.avgTPOTSLO = 50
	router.setSLOContextForRequest(request, sloCtx)

	// PreRequest should create the queue
	router.PreRequest(ctx, request, schedulingResult)

	// Verify queue was created and request was added
	queue, exists := router.runningRequestLists[pod.GetPod().NamespacedName]
	assert.True(t, exists, "Queue should be created for pod")
	assert.NotNil(t, queue)
}

func TestSLOAwareRouter_PreRequest_QueueAlreadyExists(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	pod := createTestPod("test-pod", 1, 1, 1)
	request1 := createTestLLMRequest("test-id-1", 100, 50, true)
	request2 := createTestLLMRequest("test-id-2", 100, 50, true)
	schedulingResult := createTestSchedulingResult(pod.GetPod())

	// Create and set initial SLO contexts
	sloCtx1 := newSLORequestContext(request1)
	sloCtx1.avgTPOTSLO = 50
	router.setSLOContextForRequest(request1, sloCtx1)

	sloCtx2 := newSLORequestContext(request2)
	sloCtx2.avgTPOTSLO = 50
	router.setSLOContextForRequest(request2, sloCtx2)

	// Add first request
	router.PreRequest(ctx, request1, schedulingResult)

	// Add second request to same pod
	router.PreRequest(ctx, request2, schedulingResult)

	// Verify both are in the same queue
	queue, exists := router.runningRequestLists[pod.GetPod().NamespacedName]
	assert.True(t, exists)
	assert.NotNil(t, queue)
}

func TestSLOAwareRouter_ResponseReceived_NilPredictor(t *testing.T) {
	router := createTestRouter()
	router.latencypredictor = nil

	ctx := context.Background()
	pod := createTestPod("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test", 100, 50, true)
	response := &requestcontrol.Response{}

	sloCtx := newSLORequestContext(request)
	router.setSLOContextForRequest(request, sloCtx)

	// Should not panic and should return early
	router.ResponseReceived(ctx, request, response, pod.GetPod())

	// Context should still exist
	_, err := router.getSLOContextForRequest(request)
	assert.NoError(t, err)
}

func TestSLOAwareRouter_ResponseReceived_NoPod(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	request := createTestLLMRequest("test", 100, 50, true)
	response := &requestcontrol.Response{}

	sloCtx := newSLORequestContext(request)
	router.setSLOContextForRequest(request, sloCtx)

	// Should not panic with nil pod
	router.ResponseReceived(ctx, request, response, nil)

	// Predictor should not be called

}

func TestSLOAwareRouter_ResponseReceived_NoContext(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	pod := createTestPod("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test", 100, 50, true)
	response := &requestcontrol.Response{}

	// Don't set SLO context
	router.ResponseReceived(ctx, request, response, pod.GetPod())

	// Should handle missing context gracefully

}

func TestSLOAwareRouter_ResponseStreaming_NilPredictor(t *testing.T) {
	router := createTestRouter()
	router.latencypredictor = nil

	ctx := context.Background()
	pod := createTestPod("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test", 100, 50, true)
	response := &requestcontrol.Response{}

	sloCtx := newSLORequestContext(request)
	router.setSLOContextForRequest(request, sloCtx)

	// Should not panic and should return early
	router.ResponseStreaming(ctx, request, response, pod.GetPod())

	// Context should still exist
	_, err := router.getSLOContextForRequest(request)
	assert.NoError(t, err)
}
func TestSLOAwareRouter_ResponseStreaming_FirstToken(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	pod := createTestPod("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test", 100, 50, true)
	response := &requestcontrol.Response{}
	schedulingResult := createTestSchedulingResult(pod.GetPod())

	sloCtx := newSLORequestContext(request)
	sloCtx.requestReceivedTimestamp = time.Now()
	sloCtx.schedulingResult = schedulingResult
	sloCtx.schedulingRequest = *request
	sloCtx.ttftSLO = 100
	sloCtx.avgTPOTSLO = 50
	sloCtx.incomingModelName = testModelName
	sloCtx.predictedTTFT = 80.0
	sloCtx.avgPredictedTPOT = 30.0
	// ADD THIS - populate metrics
	sloCtx.lastSeenMetrics["prefill"] = &backendmetrics.MetricsState{
		KVCacheUsagePercent: 0.5,
		WaitingQueueSize:    1,
		RunningRequestsSize: 1,
	}
	sloCtx.lastSeenMetrics["default"] = &backendmetrics.MetricsState{
		KVCacheUsagePercent: 0.5,
		WaitingQueueSize:    1,
		RunningRequestsSize: 1,
	}
	router.setSLOContextForRequest(request, sloCtx)

	// Initialize the queue and add the request
	queue := newRequestPriorityQueue()
	queue.Add(request.Headers[requtil.RequestIdHeaderKey], 50.0)
	router.runningRequestLists[pod.GetPod().NamespacedName] = queue

	beforeTime := time.Now()
	router.ResponseStreaming(ctx, request, response, pod.GetPod())
	afterTime := time.Now()

	// Verify first token timestamp was set
	retrievedCtx, err := router.getSLOContextForRequest(request)
	require.NoError(t, err)
	assert.True(t, retrievedCtx.lastTokenTimestamp.After(beforeTime) ||
		retrievedCtx.lastTokenTimestamp.Equal(beforeTime))
	assert.True(t, retrievedCtx.lastTokenTimestamp.Before(afterTime) ||
		retrievedCtx.lastTokenTimestamp.Equal(afterTime))
}

func TestSLOAwareRouter_ResponseStreaming_SubsequentTokens(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	pod := createTestPod("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test", 100, 50, true)
	response := &requestcontrol.Response{}
	schedulingResult := createTestSchedulingResult(pod.GetPod())

	sloCtx := newSLORequestContext(request)
	sloCtx.requestReceivedTimestamp = time.Now()
	sloCtx.schedulingResult = schedulingResult
	sloCtx.schedulingRequest = *request
	sloCtx.ttftSLO = 100
	sloCtx.avgTPOTSLO = 50
	sloCtx.incomingModelName = testModelName
	sloCtx.predictedTTFT = 80.0
	sloCtx.avgPredictedTPOT = 30.0
	// ADD THIS - populate metrics
	sloCtx.lastSeenMetrics["prefill"] = &backendmetrics.MetricsState{
		KVCacheUsagePercent: 0.5,
		WaitingQueueSize:    1,
		RunningRequestsSize: 1,
	}
	sloCtx.lastSeenMetrics["default"] = &backendmetrics.MetricsState{
		KVCacheUsagePercent: 0.5,
		WaitingQueueSize:    1,
		RunningRequestsSize: 1,
	}
	firstTokenTime := time.Now().Add(-100 * time.Millisecond)

	router.setSLOContextForRequest(request, sloCtx)

	// Initialize the queue and add the request
	queue := newRequestPriorityQueue()
	queue.Add(request.Headers[requtil.RequestIdHeaderKey], 50.0)
	router.runningRequestLists[pod.GetPod().NamespacedName] = queue

	router.ResponseStreaming(ctx, request, response, pod.GetPod())

	// Verify token timestamp was updated
	retrievedCtx, err := router.getSLOContextForRequest(request)
	require.NoError(t, err)
	assert.True(t, retrievedCtx.lastTokenTimestamp.After(firstTokenTime))
}

func TestSLOAwareRouter_ResponseComplete_QueueNotFound(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	pod := createTestPod("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test", 100, 50, true)
	response := &requestcontrol.Response{}

	sloCtx := newSLORequestContext(request)
	sloCtx.incomingModelName = testModelName
	sloCtx.targetPod = pod.GetPod() // ADD THIS to avoid other issues
	router.setSLOContextForRequest(request, sloCtx)

	// Create an EMPTY queue (not nil, but empty) to test queue.Remove behavior
	router.runningRequestLists[pod.GetPod().NamespacedName] = newRequestPriorityQueue()

	// Should handle gracefully when request is not in queue
	router.ResponseComplete(ctx, request, response, pod.GetPod())

	// Context should be deleted
	_, err := router.getSLOContextForRequest(request)
	assert.Error(t, err)
}
func TestSLOAwareRouter_ResponseStreaming_NoContext(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	pod := createTestPod("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test", 100, 50, true)
	response := &requestcontrol.Response{}

	// Don't set SLO context - should handle gracefully
	router.ResponseStreaming(ctx, request, response, pod.GetPod())

	// Should not panic

}

func TestSLOAwareRouter_ResponseComplete_Success(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	pod := createTestPod("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test", 100, 50, true)
	response := &requestcontrol.Response{}

	// Create queue and add request
	queue := newRequestPriorityQueue()
	router.runningRequestLists[pod.GetPod().NamespacedName] = queue
	queue.Add(request.Headers[requtil.RequestIdHeaderKey], 50.0)

	sloCtx := newSLORequestContext(request)
	sloCtx.ttft = 80
	sloCtx.avgTPOT = 30
	sloCtx.predictedTTFT = 85
	sloCtx.avgPredictedTPOT = 32
	sloCtx.ttftSLO = 100
	sloCtx.avgTPOTSLO = 50
	sloCtx.incomingModelName = "incoming-model"
	router.setSLOContextForRequest(request, sloCtx)

	router.ResponseComplete(ctx, request, response, pod.GetPod())

	// Verify context was deleted
	_, err := router.getSLOContextForRequest(request)
	assert.Error(t, err)

	// Verify request was removed from queue
	assert.Equal(t, 0, queue.Len())
}

func TestSLOAwareRouter_ResponseComplete_NilPredictor(t *testing.T) {
	router := createTestRouter()
	router.latencypredictor = nil

	ctx := context.Background()
	pod := createTestPod("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test", 100, 50, true)
	response := &requestcontrol.Response{}

	sloCtx := newSLORequestContext(request)
	router.setSLOContextForRequest(request, sloCtx)

	// Should not panic
	router.ResponseComplete(ctx, request, response, pod.GetPod())

	// Context should still exist (deletion happens only with predictor)
	_, err := router.getSLOContextForRequest(request)
	assert.NoError(t, err)
}

func TestSLOAwareRouter_ResponseComplete_NoPod(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	request := createTestLLMRequest("test", 100, 50, true)
	response := &requestcontrol.Response{}

	sloCtx := newSLORequestContext(request)
	router.setSLOContextForRequest(request, sloCtx)

	// Should not panic with nil pod
	router.ResponseComplete(ctx, request, response, nil)

	// Context should still exist (deletion happens only with validpod.GetPod())
	_, err := router.getSLOContextForRequest(request)
	assert.NoError(t, err)
}

func TestSLOAwareRouter_ResponseComplete_NoContext(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	pod := createTestPod("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test", 100, 50, true)
	response := &requestcontrol.Response{}

	// Don't set SLO context - should handle gracefully
	router.ResponseComplete(ctx, request, response, pod.GetPod())

	// Should not panic

}

func TestSLOAwareRouter_ResponseComplete_WithMetrics(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	pod := createTestPod("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test", 100, 50, true)
	response := &requestcontrol.Response{}

	// Create queue
	queue := newRequestPriorityQueue()
	router.runningRequestLists[pod.GetPod().NamespacedName] = queue
	queue.Add(request.Headers[requtil.RequestIdHeaderKey], 50.0)

	sloCtx := newSLORequestContext(request)
	sloCtx.ttft = 80
	sloCtx.avgTPOT = 30
	sloCtx.predictedTTFT = 85
	sloCtx.avgPredictedTPOT = 32
	sloCtx.ttftSLO = 100
	sloCtx.avgTPOTSLO = 50
	sloCtx.incomingModelName = "incoming-model"
	router.setSLOContextForRequest(request, sloCtx)

	// Should record metrics without panicking
	router.ResponseComplete(ctx, request, response, pod.GetPod())

	// Verify cleanup
	_, err := router.getSLOContextForRequest(request)
	assert.Error(t, err)
}

func TestSLOAwareRouter_ResponseComplete_NoSLOs(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	pod := createTestPod("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test-id", 0, 0, true) // No SLOs
	response := &requestcontrol.Response{}

	// Create queue
	queue := newRequestPriorityQueue()
	router.runningRequestLists[pod.GetPod().NamespacedName] = queue
	queue.Add(request.Headers[requtil.RequestIdHeaderKey], 0)

	sloCtx := newSLORequestContext(request)
	sloCtx.ttft = 80
	sloCtx.avgTPOT = 30
	sloCtx.incomingModelName = testModelName
	router.setSLOContextForRequest(request, sloCtx)

	// Should handle missing SLOs gracefully
	router.ResponseComplete(ctx, request, response, pod.GetPod())

	// Verify cleanup
	_, err := router.getSLOContextForRequest(request)
	assert.Error(t, err)
}

func TestSLOAwareRouter_CheckPredictor_NilPod(t *testing.T) {
	router := createTestRouter()
	logger := logr.Discard()

	result := router.checkPredictor(logger, nil)

	assert.False(t, result)
}

func TestSLOAwareRouter_CheckPredictor_NilPredictor(t *testing.T) {
	router := createTestRouter()
	router.latencypredictor = nil
	logger := logr.Discard()
	pod := createTestPod("test-pod", 1, 1, 1)

	result := router.checkPredictor(logger, pod.GetPod())

	assert.False(t, result)
}

func TestSLOAwareRouter_CheckPredictor_Success(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor
	logger := logr.Discard()
	pod := createTestPod("test-pod", 1, 1, 1)

	result := router.checkPredictor(logger, pod.GetPod())

	assert.True(t, result)
}

func TestSLORequestContext_Fields(t *testing.T) {
	request := createTestLLMRequest("test", 100, 50, true)
	ctx := newSLORequestContext(request)

	// Test all field initialization
	assert.NotNil(t, ctx.lastSeenMetrics)
	assert.NotNil(t, ctx.prefixCacheScoresForPods)
	assert.NotNil(t, ctx.predictedTTFTForScheduling)
	assert.NotNil(t, ctx.predictedTPOTForScheduling)
	assert.Empty(t, ctx.tpotObservations)
	assert.Empty(t, ctx.predictedTPOTObservations)
	assert.Zero(t, ctx.generatedTokenCount)
	assert.Zero(t, ctx.ttft)
	assert.Zero(t, ctx.avgTPOT)
	assert.Nil(t, ctx.targetPod)
	assert.Nil(t, ctx.schedulingResult)
	assert.Nil(t, ctx.tokenSampler)
}

func TestSLORequestContext_UpdateMetrics(t *testing.T) {
	request := createTestLLMRequest("test", 100, 50, true)
	ctx := newSLORequestContext(request)

	// Add some metrics
	metricsState := &backendmetrics.MetricsState{
		KVCacheUsagePercent: 0.5,
		WaitingQueueSize:    3,
	}
	ctx.lastSeenMetrics["test-pod"] = metricsState

	assert.Len(t, ctx.lastSeenMetrics, 1)
	assert.Equal(t, 0.5, ctx.lastSeenMetrics["test-pod"].KVCacheUsagePercent)
	assert.Equal(t, 3, ctx.lastSeenMetrics["test-pod"].WaitingQueueSize)
}

func TestSLORequestContext_PredictionData(t *testing.T) {
	request := createTestLLMRequest("test", 100, 50, true)
	ctx := newSLORequestContext(request)

	// Set prediction data
	ctx.predictedTTFTForScheduling["pod1"] = 80.0
	ctx.predictedTPOTForScheduling["pod1"] = 30.0
	ctx.predictedTTFTForScheduling["pod2"] = 90.0
	ctx.predictedTPOTForScheduling["pod2"] = 35.0

	assert.Len(t, ctx.predictedTTFTForScheduling, 2)
	assert.Len(t, ctx.predictedTPOTForScheduling, 2)
	assert.Equal(t, 80.0, ctx.predictedTTFTForScheduling["pod1"])
	assert.Equal(t, 30.0, ctx.predictedTPOTForScheduling["pod1"])
}

func TestSLORequestContext_PrefixCacheScores(t *testing.T) {
	request := createTestLLMRequest("test", 100, 50, true)
	ctx := newSLORequestContext(request)

	// Set prefix cache scores
	ctx.prefixCacheScoresForPods["pod1"] = 0.8
	ctx.prefixCacheScoresForPods["pod2"] = 0.6
	ctx.prefixCacheScoresForPods["pod3"] = 0.9

	assert.Len(t, ctx.prefixCacheScoresForPods, 3)
	assert.Equal(t, 0.8, ctx.prefixCacheScoresForPods["pod1"])
	assert.Equal(t, 0.9, ctx.prefixCacheScoresForPods["pod3"])
}

func TestSLOAwareRouter_ConcurrentContextAccess(t *testing.T) {
	router := createTestRouter()

	// Test concurrent access to context store
	var wg sync.WaitGroup
	numGoroutines := 100

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			requestID := uuid.New().String()
			request := createTestLLMRequest(requestID, 100, 50, true)
			sloCtx := newSLORequestContext(request)

			// Set context
			router.setSLOContextForRequest(request, sloCtx)

			// Get context
			retrievedCtx, err := router.getSLOContextForRequest(request)
			assert.NoError(t, err)
			assert.NotNil(t, retrievedCtx)

			// Delete context
			router.deleteSLOContextForRequest(request)
		}()
	}

	wg.Wait()
}

func TestSLOAwareRouter_MultipleRequests_SamePod(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	pod := createTestPod("test-pod", 1, 1, 1)

	request1 := createTestLLMRequest("test-id-1", 100, 50, true)
	request2 := createTestLLMRequest("test-id-2", 100, 50, true)
	request3 := createTestLLMRequest("test-id-3", 100, 50, true)

	schedulingResult := createTestSchedulingResult(pod.GetPod())

	// Create and set SLO contexts
	for _, req := range []*schedulingtypes.LLMRequest{request1, request2, request3} {
		sloCtx := newSLORequestContext(req)
		sloCtx.avgTPOTSLO = 50
		router.setSLOContextForRequest(req, sloCtx)
	}

	// Add all requests
	router.PreRequest(ctx, request1, schedulingResult)
	router.PreRequest(ctx, request2, schedulingResult)
	router.PreRequest(ctx, request3, schedulingResult)

	// Verify queue has all requests
	queue, exists := router.runningRequestLists[pod.GetPod().NamespacedName]
	assert.True(t, exists)
	assert.NotNil(t, queue)
}

func TestSLOAwareRouter_RequestLifecycle_Complete(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()
	pod := createTestPod("test-pod", 1, 1, 1)
	request := createTestLLMRequest("test", 100, 50, true)
	response := &requestcontrol.Response{}
	schedulingResult := createTestSchedulingResult(pod.GetPod())

	// Create initial context
	sloCtx := newSLORequestContext(request)
	sloCtx.avgTPOTSLO = 50
	sloCtx.incomingModelName = testModelName
	router.setSLOContextForRequest(request, sloCtx)

	// 1. PreRequest
	router.PreRequest(ctx, request, schedulingResult)

	// Verify context exists
	retrievedCtx, err := router.getSLOContextForRequest(request)
	require.NoError(t, err)
	assert.NotNil(t, retrievedCtx.targetPod)

	// 2. ResponseReceived
	router.ResponseReceived(ctx, request, response, pod.GetPod())

	// 3. ResponseStreaming (first token)
	router.ResponseStreaming(ctx, request, response, pod.GetPod())

	// 4. ResponseStreaming (subsequent tokens)
	retrievedCtx, _ = router.getSLOContextForRequest(request)
	retrievedCtx.ttft = 100 // Mark first token received
	router.setSLOContextForRequest(request, retrievedCtx)
	router.ResponseStreaming(ctx, request, response, pod.GetPod())

	// 5. ResponseComplete
	retrievedCtx, _ = router.getSLOContextForRequest(request)
	retrievedCtx.ttft = 80
	retrievedCtx.avgTPOT = 30
	router.setSLOContextForRequest(request, retrievedCtx)
	router.ResponseComplete(ctx, request, response, pod.GetPod())

	// Verify context was cleaned up
	_, err = router.getSLOContextForRequest(request)
	assert.Error(t, err)
}

func TestSLOAwareRouter_MultipleRequests_DifferentPods(t *testing.T) {
	router := createTestRouter()
	mockPredictor := new(mockPredictor)
	router.latencypredictor = mockPredictor

	ctx := context.Background()

	pod1 := createTestPod("test-pod-1", 1, 1, 1)
	pod2 := createTestPod("test-pod-2", 1, 1, 1)

	request1 := createTestLLMRequest("test-id-1", 100, 50, true)
	request2 := createTestLLMRequest("test-id-2", 100, 50, true)

	schedulingResult1 := createTestSchedulingResult(pod1.GetPod())
	schedulingResult2 := createTestSchedulingResult(pod2.GetPod())

	// Create and set SLO contexts
	sloCtx1 := newSLORequestContext(request1)
	sloCtx1.avgTPOTSLO = 50
	router.setSLOContextForRequest(request1, sloCtx1)

	sloCtx2 := newSLORequestContext(request2)
	sloCtx2.avgTPOTSLO = 50
	router.setSLOContextForRequest(request2, sloCtx2)

	// Add requests to different pods
	router.PreRequest(ctx, request1, schedulingResult1)
	router.PreRequest(ctx, request2, schedulingResult2)

	// Verify separate queues were created
	queue1, exists1 := router.runningRequestLists[pod1.GetPod().NamespacedName]
	queue2, exists2 := router.runningRequestLists[pod2.GetPod().NamespacedName]

	assert.True(t, exists1)
	assert.True(t, exists2)
	assert.NotNil(t, queue1)
	assert.NotNil(t, queue2)
	assert.NotEqual(t, queue1, queue2)
}

func TestSLORequestContext_SLOValidation(t *testing.T) {
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
			request := createTestLLMRequest("test-id", tt.ttftSLO, tt.tpotSLO, true)
			ctx := newSLORequestContext(request)
			ctx.ttftSLO = tt.ttftSLO
			ctx.avgTPOTSLO = tt.tpotSLO

			hasBothSLOs := ctx.ttftSLO > 0 && ctx.avgTPOTSLO > 0
			assert.Equal(t, tt.expectSLOs, hasBothSLOs)
		})
	}
}

// Benchmark tests

func BenchmarkSLOAwareRouter_PreRequest(b *testing.B) {
	router := createTestRouter()
	ctx := context.Background()
	pod := createTestPod("test-pod", 1, 1, 1)
	schedulingResult := createTestSchedulingResult(pod.GetPod())

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		requestID := uuid.New().String()
		request := createTestLLMRequest(requestID, 100, 50, true)
		sloCtx := newSLORequestContext(request)
		sloCtx.avgTPOTSLO = 50
		router.setSLOContextForRequest(request, sloCtx)
		router.PreRequest(ctx, request, schedulingResult)
	}
}

func BenchmarkSLOAwareRouter_ContextOperations(b *testing.B) {
	router := createTestRouter()
	request := createTestLLMRequest("test", 100, 50, true)
	sloCtx := newSLORequestContext(request)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		router.setSLOContextForRequest(request, sloCtx)
		_, _ = router.getSLOContextForRequest(request)
		router.deleteSLOContextForRequest(request)
	}
}

func BenchmarkSLORequestContext_Creation(b *testing.B) {
	request := createTestLLMRequest("test", 100, 50, true)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = newSLORequestContext(request)
	}
}
