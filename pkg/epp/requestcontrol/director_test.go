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
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/handlers"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	testutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/testing"
)

// --- Mock Implementations ---

// mockSaturationDetector provides a minimal mock for testing.
type mockSaturationDetector struct {
	isSaturated bool
}

func (m *mockSaturationDetector) IsSaturated() bool {
	return m.isSaturated
}

func TestDirector_HandleRequest(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	// --- Setup common objects ---
	model := "food-review"
	modelWithTarget := "food-review-target"

	// InferenceModel definitions
	imFoodReview := testutil.MakeInferenceModel("imFoodReview").
		CreationTimestamp(metav1.Unix(1000, 0)).
		ModelName(model).
		Criticality(v1alpha2.Critical).
		ObjRef()
	imFoodReviewTarget := testutil.MakeInferenceModel("imFoodReviewTarget").
		CreationTimestamp(metav1.Unix(1000, 0)).
		ModelName(modelWithTarget).
		Criticality(v1alpha2.Sheddable).
		ObjRef()

	// Datastore setup
	pmf := backendmetrics.NewPodMetricsFactory(&backendmetrics.FakePodMetricsClient{}, time.Second)
	ds := datastore.NewDatastore(t.Context(), pmf)
	ds.ModelSetIfOlder(imFoodReview)
	ds.ModelSetIfOlder(imFoodReviewTarget)

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
		ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: "default", Labels: map[string]string{"app": "inference"}},
		Status:     corev1.PodStatus{PodIP: "192.168.1.100", Phase: corev1.PodRunning, Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}}},
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
		wantErrCode            string                   // Expected errutil code string
		wantReqCtx             *handlers.RequestContext // Fields to check in the returned RequestContext
		wantMutatedBodyModel   string                   // Expected model in reqCtx.Request.Body after PostDispatch
	}{
		{
			name: "successful critical request (saturation ignored)",
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
			name: "successful sheddable request (not saturated)",
			reqBodyMap: map[string]interface{}{
				"model":  modelWithTarget,
				"prompt": "sheddable prompt",
			},
			mockSaturationDetector: &mockSaturationDetector{isSaturated: false},
			wantReqCtx: &handlers.RequestContext{
				Model:               modelWithTarget,
				ResolvedTargetModel: modelWithTarget,
				TargetPod:           "default/pod1",
				TargetEndpoint:      "192.168.1.100:8000",
			},
			wantMutatedBodyModel: modelWithTarget,
		},
		{
			name: "sheddable request dropped (saturated)",
			reqBodyMap: map[string]interface{}{
				"model":  modelWithTarget,
				"prompt": "sheddable prompt",
			},
			mockSaturationDetector: &mockSaturationDetector{isSaturated: true},
			wantErrCode:            errutil.InferencePoolResourceExhausted,
		},
		{
			name: "nil saturation detector (proceeds)",
			reqBodyMap: map[string]interface{}{
				"model":  modelWithTarget,
				"prompt": "sheddable prompt",
			},
			mockSaturationDetector: nil, // Simulate detector not being configured
			wantReqCtx: &handlers.RequestContext{
				Model:               modelWithTarget,
				ResolvedTargetModel: modelWithTarget,
				TargetPod:           "default/pod1",
				TargetEndpoint:      "192.168.1.100:8000",
			},
			wantMutatedBodyModel: modelWithTarget,
		},
		{
			name:                   "no model defined, expect err",
			reqBodyMap:             map[string]interface{}{"prompt": "p"},
			mockSaturationDetector: &mockSaturationDetector{isSaturated: false},
			wantErrCode:            errutil.BadRequest,
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
			name:        "no prompt in request body",
			reqBodyMap:  map[string]interface{}{"model": model},
			wantErrCode: errutil.BadRequest,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var sd SaturationDetector
			if tt.mockSaturationDetector != nil {
				sd = tt.mockSaturationDetector
			}

			// This should probably be mocked in the future.
			sched := scheduling.NewScheduler(ds)
			director := NewDirector(ds, sched, sd)

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
					t.Fatalf("HandleRequestBody should have returned an error containing '%s', but got nil", tt.wantErrCode)
				}
				if !strings.Contains(err.Error(), tt.wantErrCode) {
					t.Fatalf("HandleRequestBody returned error '%v', which does not contain expected substring '%s'", err, tt.wantErrCode)
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
	tests := []struct {
		name  string
		model *v1alpha2.InferenceModel
		want  string
	}{
		{
			name: "'random' distribution",
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
			name: "'random' distribution",
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
			name: "'random' distribution",
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
			name: "weighted distribution with weight unset",
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
			for range 10000 {
				model := RandomWeightedDraw(logger, test.model, seedVal)
				if model != test.want {
					t.Errorf("Model returned: %v != %v", model, test.want)
					break
				}
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
