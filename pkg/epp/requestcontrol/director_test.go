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

	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	k8stypes "k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/handlers"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
	testutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/testing"
)

// --- Mocks ---

// mockSaturationDetector provides a minimal mock for testing.
type mockSaturationDetector struct {
	isSaturated bool
}

func (m *mockSaturationDetector) IsSaturated(_ context.Context) bool {
	return m.isSaturated
}

// mockScheduler is a configurable mock for the Scheduler interface.
type mockScheduler struct {
	// Fields for Schedule
	scheduleResult *schedulingtypes.Result
	scheduleErr    error
	scheduleFunc   func(ctx context.Context, b *schedulingtypes.LLMRequest) (*schedulingtypes.Result, error)
	scheduleCalled bool

	// Fields for OnResponse
	onResponseFunc          func(ctx context.Context, resp *schedulingtypes.LLMResponse, targetPodName string)
	onResponseCalled        bool
	lastCtxOnResponse       context.Context
	lastRespOnResponse      *schedulingtypes.LLMResponse
	lastTargetPodOnResponse string
}

func (m *mockScheduler) Schedule(ctx context.Context, b *schedulingtypes.LLMRequest) (*schedulingtypes.Result, error) {
	m.scheduleCalled = true
	if m.scheduleFunc != nil {
		return m.scheduleFunc(ctx, b)
	}
	if m.scheduleErr != nil {
		return nil, m.scheduleErr
	}
	if m.scheduleResult == nil {
		// Provide a default valid pod if not specified.
		return &schedulingtypes.Result{
			TargetPod: &schedulingtypes.ScoredPod{
				Pod: &schedulingtypes.PodMetrics{
					Pod: &backend.Pod{
						Address: "192.168.1.100",
						NamespacedName: k8stypes.NamespacedName{
							Name:      "pod1",
							Namespace: "default",
						},
					},
				},
			},
		}, nil
	}
	return m.scheduleResult, nil
}

func (m *mockScheduler) OnResponse(ctx context.Context, resp *schedulingtypes.LLMResponse, targetPodName string) {
	m.onResponseCalled = true
	m.lastCtxOnResponse = ctx
	m.lastRespOnResponse = resp
	m.lastTargetPodOnResponse = targetPodName
	if m.onResponseFunc != nil {
		m.onResponseFunc(ctx, resp, targetPodName)
	}
}

func (m *mockScheduler) Reset() {
	m.scheduleResult = nil
	m.scheduleErr = nil
	m.scheduleFunc = nil
	m.scheduleCalled = false
	m.onResponseFunc = nil
	m.onResponseCalled = false
	m.lastCtxOnResponse = nil
	m.lastRespOnResponse = nil
	m.lastTargetPodOnResponse = ""
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

	tests := []struct {
		name                   string
		reqBodyMap             map[string]interface{}
		mockSaturationDetector *mockSaturationDetector
		schedulerMockSetup     func(m *mockScheduler)   // Configure the scheduler mock for this test
		wantErrCode            string                   // Expected errutil code string
		wantReqCtx             *handlers.RequestContext // Fields to check in the returned RequestContext
		wantMutatedBodyModel   string                   // Expected model in reqCtx.Request.Body after PostDispatch
	}{
		{
			name: "successful completions request (critical, saturation ignored)",
			reqBodyMap: map[string]interface{}{
				"model":  model,
				"prompt": "critical prompt",
			},
			mockSaturationDetector: &mockSaturationDetector{isSaturated: true},
			wantReqCtx: &handlers.RequestContext{
				Model:               model,
				ResolvedTargetModel: model,
				TargetPod:           "default/pod1",
				TargetEndpoint:      "192.168.1.100:8000",
			},
			wantMutatedBodyModel: model,
		},
		{
			name: "successful chat completions request (critical, saturation ignored)",
			reqBodyMap: map[string]interface{}{
				"model": model,
				"messages": []interface{}{
					map[string]interface{}{
						"role":    "user",
						"content": "critical prompt",
					},
				},
			},
			wantReqCtx: &handlers.RequestContext{
				Model:               model,
				ResolvedTargetModel: model,
				TargetPod:           "default/pod1",
				TargetEndpoint:      "192.168.1.100:8000",
			},
			wantMutatedBodyModel: model,
		},
		{
			name: "successful chat completions request with multiple messages (critical, saturation ignored)",
			reqBodyMap: map[string]interface{}{
				"model": model,
				"messages": []interface{}{
					map[string]interface{}{
						"role":    "developer",
						"content": "You are a helpful assistant.",
					},
					map[string]interface{}{
						"role":    "user",
						"content": "Hello!",
					},
				},
			},
			wantReqCtx: &handlers.RequestContext{
				Model:               model,
				ResolvedTargetModel: model,
				TargetPod:           "default/pod1",
				TargetEndpoint:      "192.168.1.100:8000",
			},
			wantMutatedBodyModel: model,
		},
		{
			name: "successful completions request (sheddable, not saturated)",
			reqBodyMap: map[string]interface{}{
				"model":  modelSheddable,
				"prompt": "sheddable prompt",
			},
			mockSaturationDetector: &mockSaturationDetector{isSaturated: false},
			wantReqCtx: &handlers.RequestContext{
				Model:               modelSheddable,
				ResolvedTargetModel: modelSheddable,
				TargetPod:           "default/pod1",
				TargetEndpoint:      "192.168.1.100:8000",
			},
			wantMutatedBodyModel: modelSheddable,
		},
		{
			name: "successful request with target model resolution",
			reqBodyMap: map[string]interface{}{
				"model":  modelWithResolvedTarget,
				"prompt": "prompt for target resolution",
			},
			mockSaturationDetector: &mockSaturationDetector{isSaturated: false},
			wantReqCtx: &handlers.RequestContext{
				Model:               modelWithResolvedTarget,
				ResolvedTargetModel: "resolved-target-model-A",
				TargetPod:           "default/pod1",
				TargetEndpoint:      "192.168.1.100:8000",
			},
			wantMutatedBodyModel: "resolved-target-model-A",
		},
		{

			name: "request dropped (sheddable, saturated)",
			reqBodyMap: map[string]interface{}{
				"model":  modelSheddable,
				"prompt": "sheddable prompt",
			},
			mockSaturationDetector: &mockSaturationDetector{isSaturated: true},
			wantErrCode:            errutil.InferencePoolResourceExhausted,
		},
		{
			name: "nil saturation detector (proceeds)",
			reqBodyMap: map[string]interface{}{
				"model":  modelSheddable,
				"prompt": "sheddable prompt",
			},
			mockSaturationDetector: nil, // Simulate detector not being configured
			wantReqCtx: &handlers.RequestContext{
				Model:               modelSheddable,
				ResolvedTargetModel: modelSheddable,
				TargetPod:           "default/pod1",
				TargetEndpoint:      "192.168.1.100:8000",
			},
			wantMutatedBodyModel: modelSheddable,
		},
		{
			name:                   "model not found, expect err",
			reqBodyMap:             map[string]interface{}{"prompt": "p"},
			mockSaturationDetector: &mockSaturationDetector{isSaturated: false},
			wantErrCode:            errutil.BadRequest,
		},

		{
			name:        "prompt or messages not found, expect err",
			reqBodyMap:  map[string]interface{}{"model": model},
			wantErrCode: errutil.BadRequest,
		},
		{
			name: "empty messages, expect err",
			reqBodyMap: map[string]interface{}{
				"model":    model,
				"messages": []interface{}{},
			},
			wantErrCode: errutil.BadRequest,
		},
		{
			name: "invalid model defined, expect err",
			reqBodyMap: map[string]interface{}{
				"model":  "non-existent-model",
				"prompt": "test prompt",
			},
			mockSaturationDetector: &mockSaturationDetector{isSaturated: false},
			wantErrCode:            errutil.BadConfiguration,
		},
		{
			name: "invalid target defined, expect err",
			reqBodyMap: map[string]interface{}{
				"model":  "food-review-1",
				"prompt": "test prompt",
			},
			mockSaturationDetector: &mockSaturationDetector{isSaturated: false},
			wantErrCode:            errutil.BadConfiguration,
		},
		{
			name: "scheduler returns error",
			reqBodyMap: map[string]interface{}{
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
			reqBodyMap: map[string]interface{}{
				"model":  model,
				"prompt": "prompt for nil,nil scheduler return",
			},
			schedulerMockSetup: func(m *mockScheduler) {
				// Explicitly set scheduleFunc to return nil, nil
				m.scheduleFunc = func(ctx context.Context, b *schedulingtypes.LLMRequest) (*schedulingtypes.Result, error) {
					return nil, nil
				}
			},
			wantErrCode: errutil.Internal,
		},
		{
			name: "scheduler returns result with nil TargetPod",
			reqBodyMap: map[string]interface{}{
				"model":  model,
				"prompt": "prompt for nil TargetPod in result",
			},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResult = &schedulingtypes.Result{TargetPod: nil}
			},
			wantErrCode: errutil.Internal,
		},
	}

	mockSched := &mockScheduler{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockSched.Reset()
			if tt.schedulerMockSetup != nil {
				tt.schedulerMockSetup(mockSched)
			}

			var sd SaturationDetector
			if tt.mockSaturationDetector != nil {
				sd = tt.mockSaturationDetector
			}
			director := NewDirector(ds, mockSched, sd)

			reqCtx := &handlers.RequestContext{
				Request: &handlers.Request{
					// Create a copy of the map for each test run to avoid mutation
					// issues.
					Body:    make(map[string]interface{}),
					Headers: make(map[string]string), // Initialize headers
				},
			}
			// Deep copy the body map
			for k, v := range tt.reqBodyMap {
				reqCtx.Request.Body[k] = v
			}

			returnedReqCtx, err := director.HandleRequest(ctx, reqCtx)

			if tt.wantErrCode != "" {
				if err == nil {
					t.Fatalf("HandleRequest() should have returned an error with code '%s', but got nil", tt.wantErrCode)
				}
				if e, ok := err.(errutil.Error); ok {
					if e.Code != tt.wantErrCode {
						t.Fatalf("HandleRequest() returned error with code %s, want %s. Full error: %v", e.Code, tt.wantErrCode, err)
					}
				} else {
					t.Fatalf("HandleRequest() returned error of type %T, want errutil.Error. Full error: %v", err, err)
				}
				return
			}

			if err != nil {
				t.Fatalf("HandleRequest() returned unexpected error: %v", err)
			}

			if tt.wantReqCtx != nil {
				if diff := cmp.Diff(tt.wantReqCtx.Model, returnedReqCtx.Model); diff != "" {
					t.Errorf("reqCtx.Model mismatch (-want +got):\n%s", diff)
				}
				if diff := cmp.Diff(tt.wantReqCtx.ResolvedTargetModel, returnedReqCtx.ResolvedTargetModel); diff != "" {
					t.Errorf("reqCtx.ResolvedTargetModel mismatch (-want +got):\n%s", diff)
				}
				if diff := cmp.Diff(tt.wantReqCtx.TargetPod, returnedReqCtx.TargetPod); diff != "" {
					t.Errorf("reqCtx.TargetPod mismatch (-want +got):\n%s", diff)
				}
				if diff := cmp.Diff(tt.wantReqCtx.TargetEndpoint, returnedReqCtx.TargetEndpoint); diff != "" {
					t.Errorf("reqCtx.TargetEndpoint mismatch (-want +got):\n%s", diff)
				}
			}

			if tt.wantMutatedBodyModel != "" {
				if returnedReqCtx.Request.Body == nil {
					t.Errorf("Expected mutated body with model %s, but reqCtx.Request.Body is nil", tt.wantMutatedBodyModel)
				} else {
					if gotModel, ok := returnedReqCtx.Request.Body["model"].(string); !ok || gotModel != tt.wantMutatedBodyModel {
						t.Errorf("Mutated reqCtx.Request.Body model = %q, want %q. Full body: %v", gotModel, tt.wantMutatedBodyModel, returnedReqCtx.Request.Body)
					}
				}
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
						{
							Name:   "canary",
							Weight: pointer(50),
						},
						{
							Name:   "v1",
							Weight: pointer(50),
						},
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
						{
							Name:   "canary",
							Weight: pointer(25),
						},
						{
							Name:   "v1.1",
							Weight: pointer(55),
						},
						{
							Name:   "v1",
							Weight: pointer(50),
						},
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
						{
							Name:   "canary",
							Weight: pointer(20),
						},
						{
							Name:   "v1.1",
							Weight: pointer(20),
						},
						{
							Name:   "v1",
							Weight: pointer(10),
						},
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
						{
							Name: "canary",
						},
						{
							Name: "v1.1",
						},
						{
							Name: "v1",
						},
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
			if model != test.want {
				t.Errorf("RandomWeightedDraw() with seed %d = %q, want %q", seedVal, model, test.want)
			}
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

func TestDirector_HandleResponse(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	ds := datastore.NewDatastore(t.Context(), nil)
	mockSched := &mockScheduler{}
	director := NewDirector(ds, mockSched, nil)

	reqCtx := &handlers.RequestContext{
		Request: &handlers.Request{
			Headers: map[string]string{
				requtil.RequestIdHeaderKey: "test-req-id-for-response",
			},
		},
		Response: &handlers.Response{ // Simulate some response headers
			Headers: map[string]string{"X-Test-Response-Header": "TestValue"},
		},
		TargetPod: "namespace1/test-pod-name",
	}

	_, err := director.HandleResponse(ctx, reqCtx)
	if err != nil {
		t.Fatalf("HandleResponse() returned unexpected error: %v", err)
	}

	if !mockSched.onResponseCalled {
		t.Fatal("Scheduler.OnResponse was not called")
	}
	if diff := cmp.Diff("test-req-id-for-response", mockSched.lastRespOnResponse.RequestId); diff != "" {
		t.Errorf("Scheduler.OnResponse RequestId mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(reqCtx.Response.Headers, mockSched.lastRespOnResponse.Headers); diff != "" {
		t.Errorf("Scheduler.OnResponse Headers mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff("namespace1/test-pod-name", mockSched.lastTargetPodOnResponse); diff != "" {
		t.Errorf("Scheduler.OnResponse TargetPodName mismatch (-want +got):\n%s", diff)
	}
}
