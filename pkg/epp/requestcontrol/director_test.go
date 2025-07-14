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
	"fmt"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
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
	scheduleResults    *schedulingtypes.SchedulingResult
	rawResults         map[string]*schedulingtypes.ProfileRunResult // Add raw results
	scheduleErr        error
}

// GetCycleState implements Scheduler.
func (m *mockScheduler) GetCycleState() *schedulingtypes.CycleState {
	panic("unimplemented")
}

// Updated Schedule method to return three values: result, rawResults, error
func (m *mockScheduler) Schedule(_ context.Context, _ *schedulingtypes.LLMRequest, _ []schedulingtypes.Pod) (*schedulingtypes.SchedulingResult, map[string]*schedulingtypes.ProfileRunResult, error) {
	// If no raw results are set, create default ones based on the schedule results
	rawResults := m.rawResults
	if rawResults == nil && m.scheduleResults != nil {
		rawResults = make(map[string]*schedulingtypes.ProfileRunResult)
		// Copy the schedule results as raw results for testing
		for profileName, profileResult := range m.scheduleResults.ProfileResults {
			if profileResult != nil {
				rawResults[profileName] = &schedulingtypes.ProfileRunResult{
					TargetPods: append([]schedulingtypes.Pod{}, profileResult.TargetPods...),
					RawScores:  make(map[string]map[schedulingtypes.Pod]float64),
				}
				// Copy raw scores if they exist
				for pod, score := range profileResult.RawScores {
					rawResults[profileName].RawScores[pod] = score
				}
			}
		}
	}
	
	return m.scheduleResults, rawResults, m.scheduleErr
}

// Helper method to set raw results for testing
func (m *mockScheduler) SetRawResults(rawResults map[string]*schedulingtypes.ProfileRunResult) {
	m.rawResults = rawResults
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

	defaultSuccessfulScheduleResults := &schedulingtypes.SchedulingResult{
		ProfileResults: map[string]*schedulingtypes.ProfileRunResult{
			"testProfile": {
				TargetPods: []schedulingtypes.Pod{
					&schedulingtypes.ScoredPod{
						Pod: &schedulingtypes.PodMetrics{
							Pod: &backend.Pod{
								Address:        "192.168.1.100",
								NamespacedName: k8stypes.NamespacedName{Name: "pod1", Namespace: "default"},
							},
						},
					},
				},
			},
		},
		PrimaryProfileName: "testProfile",
	}

	tests := []struct {
		name                   string
		reqBodyMap             map[string]any
		mockSaturationDetector *mockSaturationDetector
		schedulerMockSetup     func(m *mockScheduler)
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
					NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:        "192.168.1.100",
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
					NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:        "192.168.1.100",
				},
				TargetEndpoint: "192.168.1.100:8000",
			},
			wantMutatedBodyModel: model,
		},
		{
			name: "successful chat completions request with multiple messages (critical, saturation ignored)",
			reqBodyMap: map[string]any{
				"model": model,
				"messages": []any{
					map[string]any{
						"role":    "developer",
						"content": "You are a helpful assistant.",
					},
					map[string]any{
						"role":    "user",
						"content": "Hello!",
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
					NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:        "192.168.1.100",
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
					NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:        "192.168.1.100",
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
					NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:        "192.168.1.100",
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
					NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:        "192.168.1.100",
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
			name: "empty messages, expect err",
			reqBodyMap: map[string]any{
				"model":    model,
				"messages": []any{},
			},
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
			wantErrCode: errutil.Internal,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			mockSched := &mockScheduler{}
			if test.schedulerMockSetup != nil {
				test.schedulerMockSetup(mockSched)
			}
			director := NewDirectorWithConfig(ds, mockSched, test.mockSaturationDetector, NewConfig(), nil)

			reqCtx := &handlers.RequestContext{
				Request: &handlers.Request{
					// Create a copy of the map for each test run to avoid mutation issues.
					Body: make(map[string]any),
					Headers: map[string]string{
						requtil.RequestIdHeaderKey: "test-req-id-" + test.name, // Ensure a default request ID
					},
				},
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
				assert.Equal(t, test.wantReqCtx.TargetPod, returnedReqCtx.TargetPod, "reqCtx.TargetPod mismatch")
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

// TestGetCandidatePodsForScheduling is testing getCandidatePodsForScheduling and more specifically the functionality of SubsetFilter.
func TestGetCandidatePodsForScheduling(t *testing.T) {
	var makeFilterMetadata = func(data []any) map[string]any {
		return map[string]any{
			"envoy.lb.subset_hint": map[string]any{
				"x-gateway-destination-endpoint-subset": data,
			},
		}
	}

	testInput := []*corev1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod1",
			},
			Status: corev1.PodStatus{
				PodIP: "10.0.0.1",
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod2",
			},
			Status: corev1.PodStatus{
				PodIP: "10.0.0.2",
			},
		},
	}

	outputPod1 := &backend.Pod{
		NamespacedName: types.NamespacedName{Name: "pod1"},
		Address:        "10.0.0.1",
		Labels:         map[string]string{},
	}

	outputPod2 := &backend.Pod{
		NamespacedName: types.NamespacedName{Name: "pod2"},
		Address:        "10.0.0.2",
		Labels:         map[string]string{},
	}

	tests := []struct {
		name     string
		metadata map[string]any
		output   []schedulingtypes.Pod
	}{
		{
			name:     "SubsetFilter, filter not present — return all pods",
			metadata: map[string]any{},
			output: []schedulingtypes.Pod{
				&schedulingtypes.PodMetrics{
					Pod:          outputPod1,
					MetricsState: backendmetrics.NewMetricsState(),
				},
				&schedulingtypes.PodMetrics{
					Pod:          outputPod2,
					MetricsState: backendmetrics.NewMetricsState(),
				},
			},
		},
		{
			name:     "SubsetFilter, namespace present filter not present — return all pods",
			metadata: map[string]any{"envoy.lb.subset_hint": map[string]any{}},
			output: []schedulingtypes.Pod{
				&schedulingtypes.PodMetrics{
					Pod:          outputPod1,
					MetricsState: backendmetrics.NewMetricsState(),
				},
				&schedulingtypes.PodMetrics{
					Pod:          outputPod2,
					MetricsState: backendmetrics.NewMetricsState(),
				},
			},
		},
		{
			name:     "SubsetFilter, filter present with empty list — return error",
			metadata: makeFilterMetadata([]any{}),
			output:   []schedulingtypes.Pod{},
		},
		{
			name:     "SubsetFilter, subset with one matching pod",
			metadata: makeFilterMetadata([]any{"10.0.0.1"}),
			output: []schedulingtypes.Pod{
				&schedulingtypes.PodMetrics{
					Pod:          outputPod1,
					MetricsState: backendmetrics.NewMetricsState(),
				},
			},
		},
		{
			name:     "SubsetFilter, subset with multiple matching pods",
			metadata: makeFilterMetadata([]any{"10.0.0.1", "10.0.0.2", "10.0.0.3"}),
			output: []schedulingtypes.Pod{
				&schedulingtypes.PodMetrics{
					Pod:          outputPod1,
					MetricsState: backendmetrics.NewMetricsState(),
				},
				&schedulingtypes.PodMetrics{
					Pod:          outputPod2,
					MetricsState: backendmetrics.NewMetricsState(),
				},
			},
		},
		{
			name:     "SubsetFilter, subset with no matching pods",
			metadata: makeFilterMetadata([]any{"10.0.0.3"}),
			output:   []schedulingtypes.Pod{},
		},
	}

	pmf := backendmetrics.NewPodMetricsFactory(&backendmetrics.FakePodMetricsClient{}, time.Second)
	ds := datastore.NewDatastore(t.Context(), pmf)
	for _, testPod := range testInput {
		ds.PodUpdateOrAddIfNotExist(testPod)
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			director := NewDirectorWithConfig(ds, &mockScheduler{}, &mockSaturationDetector{}, NewConfig(), nil)

			got := director.getCandidatePodsForScheduling(context.Background(), test.metadata)

			diff := cmp.Diff(test.output, got, cmpopts.SortSlices(func(a, b schedulingtypes.Pod) bool {
				return a.GetPod().NamespacedName.String() < b.GetPod().NamespacedName.String()
			}))
			if diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}

func TestRandomWeightedDraw(t *testing.T) {
	logger := logutil.NewTestLogger()
	// Note: These tests verify deterministic outcomes for a fixed seed (420).
	// They do not test the statistical properties of the random draw.
	tests := []struct {
		name  string
		model *v1alpha2.InferenceModel
		want  string
	}{
		{
			name: "deterministic draw: 50/50 weights, seed 420",
			model: &v1alpha2.InferenceModel{
				Spec: v1alpha2.InferenceModelSpec{
					TargetModels: []v1alpha2.TargetModel{
						{Name: "canary", Weight: pointer(50)},
						{Name: "v1", Weight: pointer(50)},
					},
				},
			},
			want: "canary",
		},
		{
			name: "deterministic draw: 25/55/50 weights, seed 420",
			model: &v1alpha2.InferenceModel{
				Spec: v1alpha2.InferenceModelSpec{
					TargetModels: []v1alpha2.TargetModel{
						{Name: "canary", Weight: pointer(25)},
						{Name: "v1.1", Weight: pointer(55)},
						{Name: "v1", Weight: pointer(50)},
					},
				},
			},
			want: "v1",
		},
		{
			name: "deterministic draw: 20/20/10 weights, seed 420",
			model: &v1alpha2.InferenceModel{
				Spec: v1alpha2.InferenceModelSpec{
					TargetModels: []v1alpha2.TargetModel{
						{Name: "canary", Weight: pointer(20)},
						{Name: "v1.1", Weight: pointer(20)},
						{Name: "v1", Weight: pointer(10)},
					},
				},
			},
			want: "v1.1",
		},
		{
			name: "deterministic draw: nil weights (uniform), seed 420",
			model: &v1alpha2.InferenceModel{
				Spec: v1alpha2.InferenceModelSpec{
					TargetModels: []v1alpha2.TargetModel{
						{Name: "canary"},
						{Name: "v1.1"},
						{Name: "v1"},
					},
				},
			},
			want: "canary",
		},
	}
	var seedVal int64 = 420
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			model := RandomWeightedDraw(logger, test.model, seedVal)
			assert.Equal(t, test.want, model, "RandomWeightedDraw() with seed %d should produce expected model", seedVal)
		})
	}
}

func TestGetRandomPod(t *testing.T) {
	tests := []struct {
		name      string
		storePods []*corev1.Pod
		expectNil bool
	}{
		{
			name:      "No pods available",
			storePods: []*corev1.Pod{},
			expectNil: true,
		},
		{
			name: "Single pod available",
			storePods: []*corev1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}},
			},
			expectNil: false,
		},
		{
			name: "Multiple pods available",
			storePods: []*corev1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "pod2"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "pod3"}},
			},
			expectNil: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			pmf := backendmetrics.NewPodMetricsFactory(&backendmetrics.FakePodMetricsClient{}, time.Millisecond)
			ds := datastore.NewDatastore(t.Context(), pmf)
			for _, pod := range test.storePods {
				ds.PodUpdateOrAddIfNotExist(pod)
			}
			d := &Director{datastore: ds}
			gotPod := d.GetRandomPod()

			if test.expectNil && gotPod != nil {
				t.Errorf("expected nil pod, got: %v", gotPod)
			}
			if !test.expectNil && gotPod == nil {
				t.Errorf("expected non-nil pod, got nil")
			}
		})
	}
}

func pointer(v int32) *int32 {
	return &v
}

func newTestDirectorWithMockPredictor() (*Director, *mockPredictor) {
	mockPred := &mockPredictor{}
	director := NewDirectorWithConfig(nil, nil, nil, NewConfig(), mockPred)
	return director, mockPred
}

func newTestRequestContext(kvCache float64) *handlers.RequestContext {
	return &handlers.RequestContext{
		Request: &handlers.Request{
			Headers: map[string]string{
				requtil.RequestIdHeaderKey: "test-request-123", // Add request ID for sampler
			},
		},
		Response:  &handlers.Response{Headers: make(map[string]string)},
		Prompt:    "this is a test", // 4 tokens
		TargetPod: &backend.Pod{},
		SchedulingResult: &schedulingtypes.SchedulingResult{
			PrimaryProfileName: "default",
			ProfileResults: map[string]*schedulingtypes.ProfileRunResult{
				"default": {
					TargetPods: []schedulingtypes.Pod{
						&schedulingtypes.ScoredPod{
							Pod: &schedulingtypes.PodMetrics{
								MetricsState: &backendmetrics.MetricsState{KVCacheUsagePercent: kvCache},
							},
						},
					},
				},
			},
		},
		LastSeenMetrics:          map[string]*backendmetrics.MetricsState{"default": {KVCacheUsagePercent: kvCache}},
		RequestReceivedTimestamp: time.Now().Add(-100 * time.Millisecond), // Set received timestamp
	}
}

func TestDirector_HandleResponseHeaders(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	director, mockPred := newTestDirectorWithMockPredictor()

	// Mock TTFT prediction
	mockPred.PredictFunc = func(ctx context.Context, req latencypredictor.PredictionRequest) (*latencypredictor.PredictionResponse, error) {
		return &latencypredictor.PredictionResponse{TTFT: 120.5}, nil
	}

	reqCtx := newTestRequestContext(0.3)

	_, err := director.HandleResponseHeaders(ctx, reqCtx)
	require.NoError(t, err)

	// Header stage should predict TTFT (always predicted for scheduling decisions)
	assert.Equal(t, 120.5, reqCtx.PredictedTTFT, "TTFT should be predicted at header stage")

	// Header stage should not record actual TTFT or add training data
	assert.Equal(t, float64(0), reqCtx.TTFT, "TTFT should not be measured at header stage")
	require.Len(t, mockPred.trainingSamples, 0, "Should not add training samples at header stage")
}

func TestDirector_HandleResponseBodyChunk_FirstToken_WithFirstTPOTPrediction(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	director, mockPred := newTestDirectorWithMockPredictor()

	// Mock TPOT prediction for first token (this should be called)
	predictionCalls := 0
	mockPred.PredictFunc = func(ctx context.Context, req latencypredictor.PredictionRequest) (*latencypredictor.PredictionResponse, error) {
		predictionCalls++
		return &latencypredictor.PredictionResponse{TPOT: 35.5}, nil
	}

	reqCtx := newTestRequestContext(0.4)

	// Simulate first token arriving
	err := director.HandleResponseBodyChunk(ctx, reqCtx)
	require.NoError(t, err)

	// First token should set TTFT
	assert.Greater(t, reqCtx.TTFT, 50.0, "TTFT should be measured and positive")
	assert.Equal(t, 1, reqCtx.GeneratedTokenCount, "Token count should be 1 for first token")
	assert.NotZero(t, reqCtx.LastTokenTimestamp, "LastTokenTimestamp should be set")

	// Should ALWAYS add TTFT training sample
	require.Len(t, mockPred.trainingSamples, 1, "Should add TTFT training sample")
	sample := mockPred.trainingSamples[0]
	assert.Greater(t, sample.ActualTTFT, 50.0, "TTFT training sample should have positive TTFT")
	assert.Equal(t, 0.0, sample.ActualTPOT, "TTFT sample should have zero TPOT")
	assert.Equal(t, 0.4, sample.KVCachePercentage)
	assert.Equal(t, 4, sample.InputTokenLength)

	// Should predict first TPOT in first token block
	assert.Equal(t, 1, predictionCalls, "Should make exactly one TPOT prediction for next token")
	require.Len(t, reqCtx.PredictedTPOTObservations, 1, "Should have first TPOT prediction")
	assert.Equal(t, 35.5, reqCtx.PredictedTPOTObservations[0], "First TPOT prediction should match mocked value")

	// Should not have actual TPOT observations yet (that's for token 2+)
	assert.Len(t, reqCtx.TPOTObservations, 0, "Should not have TPOT observations for first token")

	// Should have initialized the per-request token sampler
	assert.NotNil(t, reqCtx.TokenSampler, "Should have initialized per-request TokenSampler")
}

func TestDirector_HandleResponseBodyChunk_SecondToken_RecordsIfGeneratedTokenCountIs1(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	director, mockPred := newTestDirectorWithMockPredictor()

	// Track prediction calls - should only be called for first token
	predictionCalls := 0
	mockPred.PredictFunc = func(ctx context.Context, req latencypredictor.PredictionRequest) (*latencypredictor.PredictionResponse, error) {
		predictionCalls++
		return &latencypredictor.PredictionResponse{TPOT: 30.0}, nil
	}

	reqCtx := newTestRequestContext(0.5)

	// Simulate first token
	err := director.HandleResponseBodyChunk(ctx, reqCtx)
	require.NoError(t, err)

	// Clear training samples and reset counter after first token
	mockPred.trainingSamples = nil
	predictionCalls = 0

	// Simulate a delay for the second token
	time.Sleep(25 * time.Millisecond)

	// Simulate second token - this is the key test
	err = director.HandleResponseBodyChunk(ctx, reqCtx)
	require.NoError(t, err)

	assert.Equal(t, 2, reqCtx.GeneratedTokenCount, "Token count should be 2")

	// KEY BEHAVIOR: Token 2 should record observation because GeneratedTokenCount was 1 when checked
	// This is due to the implementation logic:
	// if reqCtx.GeneratedTokenCount == 1 || reqCtx.TokenSampler.ShouldPredict(reqCtx.GeneratedTokenCount)
	require.Len(t, reqCtx.TPOTObservations, 1, "Should record TPOT observation for token 2 (GeneratedTokenCount was 1)")
	assert.Greater(t, reqCtx.TPOTObservations[0], 20.0, "TPOT observation should be positive")

	// Should add TPOT training sample for token 2 (always train)
	require.Len(t, mockPred.trainingSamples, 1, "Should add TPOT training sample")
	sample := mockPred.trainingSamples[0]
	assert.Equal(t, 0.0, sample.ActualTTFT, "TPOT sample should have zero TTFT")
	assert.Greater(t, sample.ActualTPOT, 20.0, "TPOT sample should have positive TPOT")

	// Should NOT make new prediction for token 2 (no sampling call should be made)
	assert.Equal(t, 0, predictionCalls, "Should not make new predictions for token 2")

	// Should still have the original first TPOT prediction from token 1
	require.Len(t, reqCtx.PredictedTPOTObservations, 1, "Should still have first TPOT prediction")
}

func TestDirector_HandleResponseBodyChunk_SubsequentTokens_OnlyRecordWhenSampled(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	director, mockPred := newTestDirectorWithMockPredictor()

	// Track prediction calls
	predictionCalls := 0
	mockPred.PredictFunc = func(ctx context.Context, req latencypredictor.PredictionRequest) (*latencypredictor.PredictionResponse, error) {
		predictionCalls++
		return &latencypredictor.PredictionResponse{TPOT: 30.0}, nil
	}

	reqCtx := newTestRequestContext(0.5)

	// Simulate first token (should predict first TPOT)
	err := director.HandleResponseBodyChunk(ctx, reqCtx)
	require.NoError(t, err)

	// Clear training samples from first token to focus on subsequent behavior
	mockPred.trainingSamples = nil
	firstTPOTPredictions := predictionCalls

	// Simulate second token (should record due to GeneratedTokenCount == 1)
	time.Sleep(20 * time.Millisecond)
	err = director.HandleResponseBodyChunk(ctx, reqCtx)
	require.NoError(t, err)

	initialObservations := len(reqCtx.TPOTObservations)

	// Clear training samples to track subsequent tokens
	mockPred.trainingSamples = nil

	// Simulate tokens 3-20 - these should follow normal sampling logic

	num_output_tokens := 50
	for i := 3; i <= num_output_tokens; i++ {
		time.Sleep(15 * time.Millisecond)
		err = director.HandleResponseBodyChunk(ctx, reqCtx)
		require.NoError(t, err)
	}

	// Verify behavior:
	// 1. Training happens for ALL tokens (18 tokens: 3-200)
	assert.Equal(t, num_output_tokens-2, len(mockPred.trainingSamples), "Should train on every token 3-20")

	// 2. Observations only recorded when sampled (subset of tokens 3-20)
	totalObservations := len(reqCtx.TPOTObservations)
	newObservations := totalObservations - initialObservations

	fmt.Printf("Initial observations: %d, New observations: %d, Training samples: %d\n", initialObservations, newObservations, len(mockPred.trainingSamples))

	// Should have fewer observations than training samples for tokens 3-20
	assert.Less(t, newObservations, num_output_tokens, "Should have fewer observations than training samples")
	assert.GreaterOrEqual(t, newObservations, 0, "Should have some observations")

	// Total predictions should be first TPOT + sampled predictions
	totalPredictionCalls := predictionCalls
	sampledPredictions := totalPredictionCalls - firstTPOTPredictions

	// New observations should equal sampled predictions (excluding token 2)
	assert.Equal(t, newObservations, sampledPredictions,
		"New observations should equal sampled predictions")

	assert.Equal(t, num_output_tokens, reqCtx.GeneratedTokenCount, "Should track all generated tokens")
}

const (
	testPostResponseType = "test-post-response"
)

type testPostResponse struct {
	tn                      plugins.TypedName
	lastRespOnResponse      *Response
	lastTargetPodOnResponse string
}

func newTestPostResponse(name string) *testPostResponse {
	return &testPostResponse{
		tn: plugins.TypedName{Type: testPostResponseType, Name: name},
	}
}

func (p *testPostResponse) TypedName() plugins.TypedName {
	return p.tn
}

func (p *testPostResponse) PostResponse(_ context.Context, _ *schedulingtypes.LLMRequest, response *Response, targetPod *backend.Pod) {
	p.lastRespOnResponse = response
	p.lastTargetPodOnResponse = targetPod.NamespacedName.String()
}
