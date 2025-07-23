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

package requestcontrol

import (
	"context"
	"errors"

	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	k8stypes "k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/handlers"
	latencypredictor "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/latencypredictorasync"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
	testutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/testing"
)

// --- Mocks ---

type mockSaturationDetector struct {
	isSaturated bool
}

func (m *mockSaturationDetector) IsSaturated(_ context.Context) bool {
	return m.isSaturated
}

// Updated mock scheduler to handle the new Schedule method signature
type mockScheduler struct {
	scheduleResults *schedulingtypes.SchedulingResult
	scheduleErr     error
}

// GetCycleState implements Scheduler.
func (m *mockScheduler) GetCycleState() *schedulingtypes.CycleState {
	panic("unimplemented")
}

// Updated Schedule method to return two values: result, error
func (m *mockScheduler) Schedule(_ context.Context, _ *schedulingtypes.LLMRequest, _ []schedulingtypes.Pod) (*schedulingtypes.SchedulingResult, error) {
	// If no raw results are set, create default ones based on the schedule results
	if m.scheduleResults != nil && m.scheduleResults.AllProfileRunResults == nil {
		m.scheduleResults.AllProfileRunResults = make(map[string]*schedulingtypes.ProfileRunResult)
		// Copy the schedule results as raw results for testing
		for profileName, profileResult := range m.scheduleResults.ProfileResults {
			if profileResult != nil {
				// Create a copy of the profile result for AllProfileRunResults
				allProfileResult := &schedulingtypes.ProfileRunResult{
					TargetPods: append([]schedulingtypes.Pod{}, profileResult.TargetPods...),
					RawScores:  make(map[string]map[schedulingtypes.Pod]float64),
				}

				// Add prefix-cache scores for testing
				if len(profileResult.TargetPods) > 0 {
					allProfileResult.RawScores["prefix-cache"] = make(map[schedulingtypes.Pod]float64)
					for _, pod := range profileResult.TargetPods {
						allProfileResult.RawScores["prefix-cache"][pod] = 0.8 // Default 80% prefix cache score
					}
				}

				// Copy any existing raw scores if they exist
				for scorerType, podScores := range profileResult.RawScores {
					if allProfileResult.RawScores[scorerType] == nil {
						allProfileResult.RawScores[scorerType] = make(map[schedulingtypes.Pod]float64)
					}
					for pod, score := range podScores {
						allProfileResult.RawScores[scorerType][pod] = score
					}
				}

				m.scheduleResults.AllProfileRunResults[profileName] = allProfileResult
			}
		}
	}

	return m.scheduleResults, m.scheduleErr
}

// Helper method to set raw results for testing
func (m *mockScheduler) SetRawResults(rawResults map[string]*schedulingtypes.ProfileRunResult) {
	if m.scheduleResults == nil {
		m.scheduleResults = &schedulingtypes.SchedulingResult{}
	}
	m.scheduleResults.AllProfileRunResults = rawResults
}

// mockPredictor implements the Predictor interface for testing.
type mockPredictor struct {
	PredictFunc         func(ctx context.Context, req latencypredictor.PredictionRequest) (*latencypredictor.PredictionResponse, error)
	trainingSamples     []latencypredictor.TrainingEntry
	addSampleShouldFail bool
}

var _ latencypredictor.PredictorInterface = &mockPredictor{}

func (m *mockPredictor) Predict(ctx context.Context, req latencypredictor.PredictionRequest) (*latencypredictor.PredictionResponse, error) {
	if m.PredictFunc != nil {
		return m.PredictFunc(ctx, req)
	}
	return nil, errors.New("PredictFunc not implemented")
}

func (m *mockPredictor) AddTrainingDataBulk(entry []latencypredictor.TrainingEntry) error {
	if m.addSampleShouldFail {
		return errors.New("failed to add sample")
	}
	m.trainingSamples = append(m.trainingSamples, entry...)
	return nil
}

func TestDirector_HandleRequest(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	// --- Setup common objects ---
	model := "food-review"
	modelSheddable := "food-review-sheddable"
	modelWithResolvedTarget := "food-review-resolve"

	// InferenceModel definitions
	imFoodReview := testutil.MakeInferenceModel("imFoodReview").
		CreationTimestamp(metav1.Unix(1000, 0)).
		ModelName(model).
		Criticality(v1alpha2.Critical).
		ObjRef()
	imFoodReviewSheddable := testutil.MakeInferenceModel("imFoodReviewSheddable").
		CreationTimestamp(metav1.Unix(1000, 0)).
		ModelName(modelSheddable).
		Criticality(v1alpha2.Sheddable).
		ObjRef()
	imFoodReviewResolve := testutil.MakeInferenceModel("imFoodReviewResolve").
		CreationTimestamp(metav1.Unix(1000, 0)).
		ModelName(modelWithResolvedTarget).
		Criticality(v1alpha2.Standard).
		TargetModel("resolved-target-model-A").
		ObjRef()

	// Datastore setup
	pmf := backendmetrics.NewPodMetricsFactory(&backendmetrics.FakePodMetricsClient{}, time.Second)
	ds := datastore.NewDatastore(t.Context(), pmf)
	ds.ModelSetIfOlder(imFoodReview)
	ds.ModelSetIfOlder(imFoodReviewResolve)
	ds.ModelSetIfOlder(imFoodReviewSheddable)

	pool := &v1alpha2.InferencePool{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pool", Namespace: "default"},
		Spec: v1alpha2.InferencePoolSpec{
			TargetPortNumber: int32(8000),
			Selector: map[v1alpha2.LabelKey]v1alpha2.LabelValue{
				"app": "inference",
			},
		},
	}

	// Pod setup
	testPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod1",
			Namespace: "default",
			Labels:    map[string]string{"app": "inference"},
		},
		Status: corev1.PodStatus{
			PodIP:      "192.168.1.100",
			Phase:      corev1.PodRunning,
			Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}},
		},
	}
	scheme := runtime.NewScheme()
	_ = clientgoscheme.AddToScheme(scheme)
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()
	if err := ds.PoolSet(ctx, fakeClient, pool); err != nil {
		t.Fatalf("Error while setting inference pool: %v", err)
	}
	ds.PodUpdateOrAddIfNotExist(testPod)

	// Updated defaultSuccessfulScheduleResults to include AllProfileRunResults
	defaultSuccessfulScheduleResults := &schedulingtypes.SchedulingResult{
		ProfileResults: map[string]*schedulingtypes.ProfileRunResult{
			"testProfile": {
				TargetPods: []schedulingtypes.Pod{
					&schedulingtypes.ScoredPod{
						Pod: &schedulingtypes.PodMetrics{
							Pod: &backend.Pod{
								Address:        "192.168.1.100",
								NamespacedName: k8stypes.NamespacedName{Name: "pod1", Namespace: "default"},
								RunningRequests: &backend.RequestPriorityQueue{}, // Add empty queue
								Labels:          map[string]string{"app": "inference"},
							},
						},
					},
				},
			},
		},
		PrimaryProfileName: "testProfile",
		// Add AllProfileRunResults to fix the GetTargetPodForProfile function
		AllProfileRunResults: map[string]*schedulingtypes.ProfileRunResult{
			"testProfile": {
				TargetPods: []schedulingtypes.Pod{
					&schedulingtypes.ScoredPod{
						Pod: &schedulingtypes.PodMetrics{
							Pod: &backend.Pod{
								Address:        "192.168.1.100",
								NamespacedName: k8stypes.NamespacedName{Name: "pod1", Namespace: "default"},
								RunningRequests: &backend.RequestPriorityQueue{}, // Add empty queue
								Labels:          map[string]string{"app": "inference"},
							},
						},
					},
				},
				RawScores: map[string]map[schedulingtypes.Pod]float64{
					"prefix-cache": {
						&schedulingtypes.ScoredPod{
							Pod: &schedulingtypes.PodMetrics{
								Pod: &backend.Pod{
									Address:        "192.168.1.100",
									NamespacedName: k8stypes.NamespacedName{Name: "pod1", Namespace: "default"},
									RunningRequests: &backend.RequestPriorityQueue{}, // Add empty queue
									Labels:          map[string]string{"app": "inference"},
								},
							},
						}: 0.8, // 80% prefix cache score
					},
				},
			},
		},
	}

	tests := []struct {
		name                   string
		reqBodyMap             map[string]any
		mockSaturationDetector *mockSaturationDetector
		schedulerMockSetup     func(m *mockScheduler)
		predictorMockSetup     func(m *mockPredictor)   // NEW: Add predictor setup
		wantErrCode            string                   // Expected errutil code string
		wantReqCtx             *handlers.RequestContext // Fields to check in the returned RequestContext
		wantMutatedBodyModel   string                   // Expected model in reqCtx.Request.Body after PostDispatch
	}{
		{
			name: "successful completions request (critical, saturation ignored)",
			reqBodyMap: map[string]any{
				"model":  model,
				"prompt": "critical prompt",
			},
			mockSaturationDetector: &mockSaturationDetector{isSaturated: true},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
			},
			wantReqCtx: &handlers.RequestContext{
				Model:               model,
				ResolvedTargetModel: model,
				TargetPod: &backend.Pod{
					NamespacedName:  types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:         "192.168.1.100",
					Labels:          map[string]string{"app": "inference"},
					RunningRequests: &backend.RequestPriorityQueue{}, // Empty but initialized
				},
				TargetEndpoint: "192.168.1.100:8000",
			},
			wantMutatedBodyModel: model,
		},
		{
			name: "successful request with prediction-based filtering (with SLOs)",
			reqBodyMap: map[string]any{
				"model":  model,
				"prompt": "test prompt",
			},
			mockSaturationDetector: &mockSaturationDetector{isSaturated: false},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
			},
			predictorMockSetup: func(m *mockPredictor) {
				// Mock prediction that meets SLOs
				m.PredictFunc = func(ctx context.Context, req latencypredictor.PredictionRequest) (*latencypredictor.PredictionResponse, error) {
					return &latencypredictor.PredictionResponse{
						TTFT: 80.0, // Below SLO of 100
						TPOT: 40.0, // Below SLO of 50
					}, nil
				}
			},
			wantReqCtx: &handlers.RequestContext{
				Model:               model,
				ResolvedTargetModel: model,
				TargetPod: &backend.Pod{
					NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:        "192.168.1.100",
					Labels:          map[string]string{"app": "inference"},
				},
				TargetEndpoint: "192.168.1.100:8000",
			},
			wantMutatedBodyModel: model,
		},
		{
			name: "non-critical request dropped due to prediction SLO violation",
			reqBodyMap: map[string]any{
				"model":  modelSheddable,
				"prompt": "test prompt",
			},
			mockSaturationDetector: &mockSaturationDetector{isSaturated: false},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
			},
			predictorMockSetup: func(m *mockPredictor) {
				// Mock prediction that violates SLOs
				m.PredictFunc = func(ctx context.Context, req latencypredictor.PredictionRequest) (*latencypredictor.PredictionResponse, error) {
					return &latencypredictor.PredictionResponse{
						TTFT: 150.0, // Above SLO of 100
						TPOT: 80.0,  // Above SLO of 50
					}, nil
				}
			},
			wantErrCode: errutil.InferencePoolResourceExhausted,
		},
		{
			name: "critical request succeeds despite prediction SLO violation",
			reqBodyMap: map[string]any{
				"model":  model, // Critical model
				"prompt": "test prompt",
			},
			mockSaturationDetector: &mockSaturationDetector{isSaturated: false},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
			},
			predictorMockSetup: func(m *mockPredictor) {
				// Mock prediction that violates SLOs
				m.PredictFunc = func(ctx context.Context, req latencypredictor.PredictionRequest) (*latencypredictor.PredictionResponse, error) {
					return &latencypredictor.PredictionResponse{
						TTFT: 150.0, // Above SLO of 100
						TPOT: 80.0,  // Above SLO of 50
					}, nil
				}
			},
			wantReqCtx: &handlers.RequestContext{
				Model:               model,
				ResolvedTargetModel: model,
				TargetPod: &backend.Pod{
					NamespacedName:  types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:         "192.168.1.100",
					Labels:          map[string]string{"app": "inference"},
					RunningRequests: &backend.RequestPriorityQueue{}, // Empty but initialized
				},
				TargetEndpoint: "192.168.1.100:8000",
			},
			wantMutatedBodyModel: model,
		},
		{
			name: "successful chat completions request (critical, saturation ignored)",
			reqBodyMap: map[string]any{
				"model": model,
				"messages": []any{
					map[string]any{
						"role":    "user",
						"content": "critical prompt",
					},
				},
			},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
			},
			wantReqCtx: &handlers.RequestContext{
				Model:               model,
				ResolvedTargetModel: model,
				TargetPod: &backend.Pod{
					NamespacedName:  types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:         "192.168.1.100",
					Labels:          map[string]string{"app": "inference"},
					RunningRequests: &backend.RequestPriorityQueue{}, // Empty but initialized
				},
				TargetEndpoint: "192.168.1.100:8000",
			},
			wantMutatedBodyModel: model,
		},
		{
			name: "successful completions request (sheddable, not saturated)",
			reqBodyMap: map[string]any{
				"model":  modelSheddable,
				"prompt": "sheddable prompt",
			},
			mockSaturationDetector: &mockSaturationDetector{isSaturated: false},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
			},
			wantReqCtx: &handlers.RequestContext{
				Model:               modelSheddable,
				ResolvedTargetModel: modelSheddable,
				TargetPod: &backend.Pod{
					NamespacedName:  types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:         "192.168.1.100",
					Labels:          map[string]string{"app": "inference"},
					RunningRequests: &backend.RequestPriorityQueue{}, // Empty but initialized
				},
				TargetEndpoint: "192.168.1.100:8000",
			},
			wantMutatedBodyModel: modelSheddable,
		},
		{
			name: "successful request with target model resolution",
			reqBodyMap: map[string]any{
				"model":  modelWithResolvedTarget,
				"prompt": "prompt for target resolution",
			},
			mockSaturationDetector: &mockSaturationDetector{isSaturated: false},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
			},
			wantReqCtx: &handlers.RequestContext{
				Model:               modelWithResolvedTarget,
				ResolvedTargetModel: "resolved-target-model-A",
				TargetPod: &backend.Pod{
					NamespacedName:  types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:         "192.168.1.100",
					Labels:          map[string]string{"app": "inference"},
					RunningRequests: &backend.RequestPriorityQueue{}, // Empty but initialized
				},
				TargetEndpoint: "192.168.1.100:8000",
			},
			wantMutatedBodyModel: "resolved-target-model-A",
		},
		{
			name: "nonexistent target defined, use default inference model",
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
			},
			wantReqCtx: &handlers.RequestContext{
				Model:               "food-review-1",
				ResolvedTargetModel: "food-review-1",
				TargetPod: &backend.Pod{
					NamespacedName:  types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:         "192.168.1.100",
					Labels:          map[string]string{"app": "inference"},
					RunningRequests: &backend.RequestPriorityQueue{}, // Empty but initialized
				},
				TargetEndpoint: "192.168.1.100:8000",
			},
			wantMutatedBodyModel: "food-review-1",
			reqBodyMap: map[string]any{
				"model":  "food-review-1",
				"prompt": "test prompt",
			},
			mockSaturationDetector: &mockSaturationDetector{isSaturated: false},
		},
		{
			name: "request dropped (sheddable, saturated)",
			reqBodyMap: map[string]any{
				"model":  modelSheddable,
				"prompt": "sheddable prompt",
			},
			mockSaturationDetector: &mockSaturationDetector{isSaturated: true},
			wantErrCode:            errutil.InferencePoolResourceExhausted,
		},
		{
			name:                   "model not found, expect err",
			reqBodyMap:             map[string]any{"prompt": "p"},
			mockSaturationDetector: &mockSaturationDetector{isSaturated: false},
			wantErrCode:            errutil.BadRequest,
		},
		{
			name:        "prompt or messages not found, expect err",
			reqBodyMap:  map[string]any{"model": model},
			wantErrCode: errutil.BadRequest,
		},
		{
			name: "scheduler returns error",
			reqBodyMap: map[string]any{
				"model":  model,
				"prompt": "prompt that causes scheduler error",
			},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleErr = errors.New("simulated scheduler failure")
			},
			wantErrCode: errutil.InferencePoolResourceExhausted,
		},
		{
			name: "scheduler returns nil result and nil error",
			reqBodyMap: map[string]any{
				"model":  model,
				"prompt": "prompt for nil,nil scheduler return",
			},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = nil
				m.scheduleErr = nil
			},
			wantErrCode: errutil.InferencePoolResourceExhausted,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			mockSched := &mockScheduler{}
			if test.schedulerMockSetup != nil {
				test.schedulerMockSetup(mockSched)
			}

			// Setup predictor for tests that need SLO-based filtering
			var mockPred *mockPredictor
			var director *Director
			if test.predictorMockSetup != nil {
				mockPred = &mockPredictor{}
				test.predictorMockSetup(mockPred)
				director = NewDirectorWithConfig(ds, mockSched, test.mockSaturationDetector, NewConfig(), mockPred)
			} else {
				director = NewDirectorWithConfig(ds, mockSched, test.mockSaturationDetector, NewConfig(), nil)
			}

			reqCtx := &handlers.RequestContext{
				Request: &handlers.Request{
					// Create a copy of the map for each test run to avoid mutation issues.
					Body: make(map[string]any),
					Headers: map[string]string{
						requtil.RequestIdHeaderKey: "test-req-id-" + test.name, // Ensure a default request ID
					},
				},
			}

			// Add SLO headers for prediction tests
			if test.predictorMockSetup != nil {
				reqCtx.Request.Headers["ttft_slo"] = "100.0"    // 100ms TTFT SLO
				reqCtx.Request.Headers["avg_tpot_slo"] = "50.0" // 50ms TPOT SLO
			}

			// Deep copy the body map.
			for k, v := range test.reqBodyMap {
				reqCtx.Request.Body[k] = v
			}

			returnedReqCtx, err := director.HandleRequest(ctx, reqCtx)

			if test.wantErrCode != "" {
				assert.Error(t, err, "HandleRequest() should have returned an error")
				var e errutil.Error
				if assert.ErrorAs(t, err, &e, "Error should be of type errutil.Error") {
					assert.Equal(t, test.wantErrCode, e.Code, "Error code mismatch")
				}
				return
			}

			assert.NoError(t, err, "HandleRequest() returned unexpected error")

			if test.wantReqCtx != nil {
				assert.Equal(t, test.wantReqCtx.Model, returnedReqCtx.Model, "reqCtx.Model mismatch")
				assert.Equal(t, test.wantReqCtx.ResolvedTargetModel, returnedReqCtx.ResolvedTargetModel,
					"reqCtx.ResolvedTargetModel mismatch")
				if test.wantReqCtx != nil && test.wantReqCtx.TargetPod != nil {
					expected := test.wantReqCtx.TargetPod
					actual := returnedReqCtx.TargetPod

					assert.Equal(t, expected.NamespacedName, actual.NamespacedName, "NamespacedName mismatch")
					assert.Equal(t, expected.Address, actual.Address, "Address mismatch")
					assert.Equal(t, expected.Labels, actual.Labels, "Labels mismatch")
					// Skip RunningRequests comparison - it's not relevant to the test
				}
				assert.Equal(t, test.wantReqCtx.TargetEndpoint, returnedReqCtx.TargetEndpoint, "reqCtx.TargetEndpoint mismatch")
			}

			if test.wantMutatedBodyModel != "" {
				assert.NotNil(t, returnedReqCtx.Request.Body, "Expected mutated body, but reqCtx.Request.Body is nil")
				assert.Equal(t, test.wantMutatedBodyModel, returnedReqCtx.Request.Body["model"],
					"Mutated reqCtx.Request.Body model mismatch")
			}

			// Verify prediction context is populated when predictor is used
			if test.predictorMockSetup != nil && err == nil {
				assert.NotNil(t, returnedReqCtx.SchedulingRequest, "SchedulingRequest should be populated")
				// Predictions arrays may be populated depending on the specific test scenario
			}
		})
	}
}

// Add a specific test for the PredictionScorer
func TestDirector_HandleRequest_PredictionFiltering_Fixed(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	// Setup datastore and models (same as before)
	model := "food-review"
	modelSheddable := "food-review-sheddable"

	imFoodReview := testutil.MakeInferenceModel("imFoodReview").
		CreationTimestamp(metav1.Unix(1000, 0)).
		ModelName(model).
		Criticality(v1alpha2.Critical).
		ObjRef()
	imFoodReviewSheddable := testutil.MakeInferenceModel("imFoodReviewSheddable").
		CreationTimestamp(metav1.Unix(1000, 0)).
		ModelName(modelSheddable).
		Criticality(v1alpha2.Sheddable).
		ObjRef()

	pmf := backendmetrics.NewPodMetricsFactory(&backendmetrics.FakePodMetricsClient{}, time.Second)
	ds := datastore.NewDatastore(t.Context(), pmf)
	ds.ModelSetIfOlder(imFoodReview)
	ds.ModelSetIfOlder(imFoodReviewSheddable)

	pool := &v1alpha2.InferencePool{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pool", Namespace: "default"},
		Spec: v1alpha2.InferencePoolSpec{
			TargetPortNumber: int32(8000),
			Selector: map[v1alpha2.LabelKey]v1alpha2.LabelValue{
				"app": "inference",
			},
		},
	}

	testPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod1",
			Namespace: "default",
			Labels:    map[string]string{"app": "inference"},
		},
		Status: corev1.PodStatus{
			PodIP:      "192.168.1.100",
			Phase:      corev1.PodRunning,
			Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}},
		},
	}

	scheme := runtime.NewScheme()
	_ = clientgoscheme.AddToScheme(scheme)
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()
	if err := ds.PoolSet(ctx, fakeClient, pool); err != nil {
		t.Fatalf("Error while setting inference pool: %v", err)
	}
	ds.PodUpdateOrAddIfNotExist(testPod)

	defaultSuccessfulScheduleResults := &schedulingtypes.SchedulingResult{
		ProfileResults: map[string]*schedulingtypes.ProfileRunResult{
			"testProfile": {
				TargetPods: []schedulingtypes.Pod{
					&schedulingtypes.ScoredPod{
						Pod: &schedulingtypes.PodMetrics{
							Pod: &backend.Pod{
								Address:         "192.168.1.100",
								NamespacedName:  k8stypes.NamespacedName{Name: "pod1", Namespace: "default"},
								RunningRequests: &backend.RequestPriorityQueue{}, // Add empty queue
								Labels:          map[string]string{"app": "inference"},
							},
						},
					},
				},
			},
		},
		PrimaryProfileName: "testProfile",
		AllProfileRunResults: map[string]*schedulingtypes.ProfileRunResult{
			"testProfile": {
				TargetPods: []schedulingtypes.Pod{
					&schedulingtypes.ScoredPod{
						Pod: &schedulingtypes.PodMetrics{
							Pod: &backend.Pod{
								Address:         "192.168.1.100",
								NamespacedName:  k8stypes.NamespacedName{Name: "pod1", Namespace: "default"},
								RunningRequests: &backend.RequestPriorityQueue{}, // Add empty queue
								Labels:          map[string]string{"app": "inference"},
							},
						},
					},
				},
				RawScores: map[string]map[schedulingtypes.Pod]float64{
					"prefix-cache": {
						&schedulingtypes.ScoredPod{
							Pod: &schedulingtypes.PodMetrics{
								Pod: &backend.Pod{
									Address:         "192.168.1.100",
									NamespacedName:  k8stypes.NamespacedName{Name: "pod1", Namespace: "default"},
									RunningRequests: &backend.RequestPriorityQueue{}, // Add empty queue
								},
							},
						}: 0.8,
					},
				},
			},
		},
	}

	tests := []struct {
		name                   string
		reqBodyMap             map[string]any
		mockSaturationDetector *mockSaturationDetector
		schedulerMockSetup     func(m *mockScheduler)
		predictorMockSetup     func(m *mockPredictor)
		wantErrCode            string
		wantReqCtx             *handlers.RequestContext
		wantMutatedBodyModel   string
	}{
		{
			name: "non-critical request dropped due to prediction SLO violation",
			reqBodyMap: map[string]any{
				"model":  modelSheddable,
				"prompt": "test prompt",
			},
			mockSaturationDetector: &mockSaturationDetector{isSaturated: false},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
			},
			predictorMockSetup: func(m *mockPredictor) {
				// Mock prediction that violates SLOs
				m.PredictFunc = func(ctx context.Context, req latencypredictor.PredictionRequest) (*latencypredictor.PredictionResponse, error) {
					return &latencypredictor.PredictionResponse{
						TTFT: 150.0, // Above SLO of 100
						TPOT: 80.0,  // Above SLO of 50
					}, nil
				}
			},
			wantErrCode: errutil.InferencePoolResourceExhausted,
		},
		{
			name: "critical request succeeds despite prediction SLO violation",
			reqBodyMap: map[string]any{
				"model":  model, // Critical model
				"prompt": "test prompt",
			},
			mockSaturationDetector: &mockSaturationDetector{isSaturated: false},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
			},
			predictorMockSetup: func(m *mockPredictor) {
				// Mock prediction that violates SLOs
				m.PredictFunc = func(ctx context.Context, req latencypredictor.PredictionRequest) (*latencypredictor.PredictionResponse, error) {
					return &latencypredictor.PredictionResponse{
						TTFT: 150.0, // Above SLO of 100
						TPOT: 80.0,  // Above SLO of 50
					}, nil
				}
			},
			wantReqCtx: &handlers.RequestContext{
				Model:               model,
				ResolvedTargetModel: model,
				TargetPod: &backend.Pod{
					NamespacedName:  types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:         "192.168.1.100",
					RunningRequests: &backend.RequestPriorityQueue{}, // Add empty queue
					Labels:          map[string]string{"app": "inference"},
				},
				TargetEndpoint: "192.168.1.100:8000",
			},
			wantMutatedBodyModel: model,
		},
		{
			name: "scheduler returns nil result should handle gracefully",
			reqBodyMap: map[string]any{
				"model":  model,
				"prompt": "test prompt",
			},
			mockSaturationDetector: &mockSaturationDetector{isSaturated: false},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = nil
				m.scheduleErr = nil
			},
			wantErrCode: errutil.InferencePoolResourceExhausted, // Should be handled in applyPredictionScoring
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			mockSched := &mockScheduler{}
			if test.schedulerMockSetup != nil {
				test.schedulerMockSetup(mockSched)
			}

			var mockPred *mockPredictor
			var director *Director
			if test.predictorMockSetup != nil {
				mockPred = &mockPredictor{}
				test.predictorMockSetup(mockPred)
				director = NewDirectorWithConfig(ds, mockSched, test.mockSaturationDetector, NewConfig(), mockPred)
			} else {
				director = NewDirectorWithConfig(ds, mockSched, test.mockSaturationDetector, NewConfig(), nil)
			}

			reqCtx := &handlers.RequestContext{
				Request: &handlers.Request{
					Body: make(map[string]any),
					Headers: map[string]string{
						requtil.RequestIdHeaderKey: "test-req-id-" + test.name,
					},
				},
			}

			// Add SLO headers for prediction tests
			if test.predictorMockSetup != nil {
				reqCtx.Request.Headers["ttft_slo"] = "100.0"    // 100ms TTFT SLO
				reqCtx.Request.Headers["avg_tpot_slo"] = "50.0" // 50ms TPOT SLO
			}

			// Deep copy the body map
			for k, v := range test.reqBodyMap {
				reqCtx.Request.Body[k] = v
			}

			returnedReqCtx, err := director.HandleRequest(ctx, reqCtx)

			if test.wantErrCode != "" {
				assert.Error(t, err, "HandleRequest() should have returned an error")
				var e errutil.Error
				if assert.ErrorAs(t, err, &e, "Error should be of type errutil.Error") {
					assert.Equal(t, test.wantErrCode, e.Code, "Error code mismatch")
				}
				return
			}

			assert.NoError(t, err, "HandleRequest() returned unexpected error")

			if test.wantReqCtx != nil {
				assert.Equal(t, test.wantReqCtx.Model, returnedReqCtx.Model, "reqCtx.Model mismatch")
				assert.Equal(t, test.wantReqCtx.ResolvedTargetModel, returnedReqCtx.ResolvedTargetModel,
					"reqCtx.ResolvedTargetModel mismatch")
				if test.wantReqCtx != nil && test.wantReqCtx.TargetPod != nil {
					expected := test.wantReqCtx.TargetPod
					actual := returnedReqCtx.TargetPod

					assert.Equal(t, expected.NamespacedName, actual.NamespacedName, "NamespacedName mismatch")
					assert.Equal(t, expected.Address, actual.Address, "Address mismatch")
					assert.Equal(t, expected.Labels, actual.Labels, "Labels mismatch")
					// Skip RunningRequests comparison - it's not relevant to the test
				}
				assert.Equal(t, test.wantReqCtx.TargetEndpoint, returnedReqCtx.TargetEndpoint, "reqCtx.TargetEndpoint mismatch")
			}

			if test.wantMutatedBodyModel != "" {
				assert.NotNil(t, returnedReqCtx.Request.Body, "Expected mutated body, but reqCtx.Request.Body is nil")
				assert.Equal(t, test.wantMutatedBodyModel, returnedReqCtx.Request.Body["model"],
					"Mutated reqCtx.Request.Body model mismatch")
			}
		})
	}
}
