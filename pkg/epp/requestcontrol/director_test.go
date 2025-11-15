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
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/handlers"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	requtil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/request"
	testutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/testing"
)

// --- Mocks ---

type mockAdmissionController struct {
	admitErr error
}

func (m *mockAdmissionController) Admit(
	_ context.Context,
	_ *handlers.RequestContext,
	_ []backendmetrics.PodMetrics,
	_ int,
) error {
	return m.admitErr
}

type mockScheduler struct {
	scheduleResults *schedulingtypes.SchedulingResult
	scheduleErr     error
}

func (m *mockScheduler) Schedule(_ context.Context, _ *schedulingtypes.LLMRequest, _ []schedulingtypes.Pod) (*schedulingtypes.SchedulingResult, error) {
	return m.scheduleResults, m.scheduleErr
}

type mockDatastore struct {
	pods     []backendmetrics.PodMetrics
	rewrites []*v1alpha2.InferenceModelRewrite
}

func (ds *mockDatastore) PoolGet() (*v1.InferencePool, error) {
	return nil, nil
}
func (ds *mockDatastore) ObjectiveGet(_ string) *v1alpha2.InferenceObjective {
	return nil
}
func (ds *mockDatastore) PodList(predicate func(backendmetrics.PodMetrics) bool) []backendmetrics.PodMetrics {
	res := []backendmetrics.PodMetrics{}
	for _, pod := range ds.pods {
		if predicate(pod) {
			res = append(res, pod)
		}
	}

	return res
}

func (ds *mockDatastore) RewriteGetAll() []*v1alpha2.InferenceModelRewrite {
	return ds.rewrites
}

func TestDirector_HandleRequest(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	// --- Setup common objects ---
	model := "food-review"
	modelSheddable := "food-review-sheddable"
	modelWithResolvedTarget := "food-review-resolve"
	modelRewritten := "food-review-rewritten"

	objectiveName := "ioFoodReview"
	objectiveNameSheddable := "imFoodReviewSheddable"
	objectiveNameResolve := "imFoodReviewResolve"
	// InferenceObjective definitions
	ioFoodReview := testutil.MakeInferenceObjective("ioFoodReview").
		CreationTimestamp(metav1.Unix(1000, 0)).
		Priority(2).
		ObjRef()
	ioFoodReviewSheddable := testutil.MakeInferenceObjective("imFoodReviewSheddable").
		CreationTimestamp(metav1.Unix(1000, 0)).
		Priority(-1).
		ObjRef()
	ioFoodReviewResolve := testutil.MakeInferenceObjective("imFoodReviewResolve").
		CreationTimestamp(metav1.Unix(1000, 0)).
		Priority(1).
		ObjRef()

	// Datastore setup
	pmf := backendmetrics.NewPodMetricsFactory(&backendmetrics.FakePodMetricsClient{}, time.Second)
	ds := datastore.NewDatastore(t.Context(), pmf, 0)
	ds.ObjectiveSet(ioFoodReview)
	ds.ObjectiveSet(ioFoodReviewResolve)
	ds.ObjectiveSet(ioFoodReviewSheddable)

	rewrite := &v1alpha2.InferenceModelRewrite{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "rewrite-rule",
			CreationTimestamp: metav1.Now(),
		},
		Spec: v1alpha2.InferenceModelRewriteSpec{
			Rules: []v1alpha2.InferenceModelRewriteRule{
				{
					Matches: []v1alpha2.Match{
						{
							Model: &v1alpha2.ModelMatch{
								Value: "food-review",
							},
						},
					},
					Targets: []v1alpha2.TargetModel{
						{
							ModelRewrite: modelRewritten,
							Weight:       100,
						},
					},
				},
			},
		},
	}
	ds.RewriteSet(rewrite)

	pool := &v1.InferencePool{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pool", Namespace: "default"},
		Spec: v1.InferencePoolSpec{
			TargetPorts: []v1.Port{{Number: v1.PortNumber(int32(8000))}},
			Selector: v1.LabelSelector{
				MatchLabels: map[v1.LabelKey]v1.LabelValue{
					"app": "inference",
				},
			},
		},
	}

	scheme := runtime.NewScheme()
	_ = clientgoscheme.AddToScheme(scheme)
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()
	if err := ds.PoolSet(ctx, fakeClient, pool); err != nil {
		t.Fatalf("Error while setting inference pool: %v", err)
	}

	for i := range 5 {
		// Pod setup
		testPod := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("pod%v", i+1),
				Namespace: "default",
				Labels:    map[string]string{"app": "inference"},
			},
			Status: corev1.PodStatus{
				PodIP:      fmt.Sprintf("192.168.%v.100", i+1),
				Phase:      corev1.PodRunning,
				Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}},
			},
		}
		ds.PodUpdateOrAddIfNotExist(testPod)
	}

	defaultSuccessfulScheduleResults := &schedulingtypes.SchedulingResult{
		ProfileResults: map[string]*schedulingtypes.ProfileRunResult{
			"testProfile": {
				TargetPods: []schedulingtypes.Pod{
					&schedulingtypes.ScoredPod{
						Pod: &schedulingtypes.PodMetrics{
							Pod: &backend.Pod{
								Address:        "192.168.1.100",
								Port:           "8000",
								MetricsHost:    "192.168.1.100:8000",
								NamespacedName: types.NamespacedName{Name: "pod1", Namespace: "default"},
							},
						},
					},
					&schedulingtypes.ScoredPod{
						Pod: &schedulingtypes.PodMetrics{
							Pod: &backend.Pod{
								Address:        "192.168.2.100",
								Port:           "8000",
								MetricsHost:    "192.168.2.100:8000",
								NamespacedName: types.NamespacedName{Name: "pod2", Namespace: "default"},
							},
						},
					},
					&schedulingtypes.ScoredPod{
						Pod: &schedulingtypes.PodMetrics{
							Pod: &backend.Pod{
								Address:        "192.168.4.100",
								Port:           "8000",
								MetricsHost:    "192.168.4.100:8000",
								NamespacedName: types.NamespacedName{Name: "pod4", Namespace: "default"},
							},
						},
					},
				},
			},
		},
		PrimaryProfileName: "testProfile",
	}

	tests := []struct {
		name                    string
		reqBodyMap              map[string]any
		mockAdmissionController *mockAdmissionController
		inferenceObjectiveName  string
		schedulerMockSetup      func(m *mockScheduler)
		initialTargetModelName  string                   // Initial target model in the reqCtx.
		wantErrCode             string                   // Expected errutil code string
		wantReqCtx              *handlers.RequestContext // Fields to check in the returned RequestContext
		wantMutatedBodyModel    string                   // Expected model in reqCtx.Request.Body after PostDispatch
	}{
		{
			name: "successful completions request",
			reqBodyMap: map[string]any{
				"model":  model,
				"prompt": "critical prompt",
			},
			mockAdmissionController: &mockAdmissionController{admitErr: nil},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
			},
			initialTargetModelName: model,
			wantReqCtx: &handlers.RequestContext{
				ObjectiveKey:    objectiveName,
				TargetModelName: modelRewritten,
				TargetPod: &backend.Pod{
					NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:        "192.168.1.100",
					Port:           "8000",
					MetricsHost:    "192.168.1.100:8000",
				},
				TargetEndpoint: "192.168.1.100:8000,192.168.2.100:8000,192.168.4.100:8000",
			},
			wantMutatedBodyModel:   modelRewritten,
			inferenceObjectiveName: objectiveName,
		}, {
			name: "successful request with model rewrite",
			reqBodyMap: map[string]any{
				"model":  model,
				"prompt": "some prompt",
			},
			mockAdmissionController: &mockAdmissionController{admitErr: nil},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
			},
			initialTargetModelName: model,
			wantReqCtx: &handlers.RequestContext{
				ObjectiveKey:    model,
				TargetModelName: modelRewritten,
				TargetPod: &backend.Pod{
					NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:        "192.168.1.100",
					Port:           "8000",
					MetricsHost:    "192.168.1.100:8000",
				},
				TargetEndpoint: "192.168.1.100:8000,192.168.2.100:8000,192.168.4.100:8000",
			},
			wantMutatedBodyModel:   modelRewritten,
			inferenceObjectiveName: model,
		}, {
			name: "successful chat completions request",
			reqBodyMap: map[string]any{
				"model": model,
				"messages": []any{
					map[string]any{
						"role":    "user",
						"content": "critical prompt",
					},
				},
			},
			mockAdmissionController: &mockAdmissionController{admitErr: nil},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
			},
			initialTargetModelName: model,
			wantReqCtx: &handlers.RequestContext{
				TargetModelName: modelRewritten,
				TargetPod: &backend.Pod{
					NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:        "192.168.1.100",
					Port:           "8000",
					MetricsHost:    "192.168.1.100:8000",
				},
				TargetEndpoint: "192.168.1.100:8000,192.168.2.100:8000,192.168.4.100:8000",
			},
			wantMutatedBodyModel: modelRewritten,
		}, {
			name: "successful chat completions request with multiple messages",
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
			mockAdmissionController: &mockAdmissionController{admitErr: nil},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
			},
			initialTargetModelName: model,
			wantReqCtx: &handlers.RequestContext{
				ObjectiveKey:    objectiveName,
				TargetModelName: modelRewritten,
				TargetPod: &backend.Pod{
					NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:        "192.168.1.100",
					Port:           "8000",
					MetricsHost:    "192.168.1.100:8000",
				},
				TargetEndpoint: "192.168.1.100:8000,192.168.2.100:8000,192.168.4.100:8000",
			},
			wantMutatedBodyModel:   modelRewritten,
			inferenceObjectiveName: objectiveName,
		}, {
			name: "successful request with target model resolution",
			reqBodyMap: map[string]any{
				"model":  modelWithResolvedTarget,
				"prompt": "prompt for target resolution",
			},
			mockAdmissionController: &mockAdmissionController{admitErr: nil},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
			},
			initialTargetModelName: "resolved-target-model-A",
			wantReqCtx: &handlers.RequestContext{
				ObjectiveKey:    objectiveNameResolve,
				TargetModelName: "resolved-target-model-A",
				TargetPod: &backend.Pod{
					NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:        "192.168.1.100",
					Port:           "8000",
					MetricsHost:    "192.168.1.100:8000",
				},
				TargetEndpoint: "192.168.1.100:8000,192.168.2.100:8000,192.168.4.100:8000",
			},
			wantMutatedBodyModel:   "resolved-target-model-A",
			inferenceObjectiveName: objectiveNameResolve,
		},
		{
			name: "nonexistent target defined, use default inference model",
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
			},
			initialTargetModelName: "food-review-1",
			wantReqCtx: &handlers.RequestContext{
				ObjectiveKey:    "food-review-1",
				TargetModelName: "food-review-1",
				TargetPod: &backend.Pod{
					NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:        "192.168.1.100",
					Port:           "8000",
					MetricsHost:    "192.168.1.100:8000",
				},
				TargetEndpoint: "192.168.1.100:8000,192.168.2.100:8000,192.168.4.100:8000",
			},
			wantMutatedBodyModel: "food-review-1",
			reqBodyMap: map[string]any{
				"model":  "food-review-1",
				"prompt": "test prompt",
			},
			mockAdmissionController: &mockAdmissionController{admitErr: nil},
			inferenceObjectiveName:  "food-review-1",
		},
		{

			name: "request rejected by admission controller",
			reqBodyMap: map[string]any{
				"model":  modelSheddable,
				"prompt": "sheddable prompt",
			},
			inferenceObjectiveName:  objectiveNameSheddable,
			mockAdmissionController: &mockAdmissionController{admitErr: errutil.Error{Code: errutil.InferencePoolResourceExhausted, Msg: "simulated admission rejection"}},
			wantErrCode:             errutil.InferencePoolResourceExhausted,
		},
		{
			name:                    "model not found, expect err",
			reqBodyMap:              map[string]any{"prompt": "p"},
			mockAdmissionController: &mockAdmissionController{admitErr: nil},
			wantErrCode:             errutil.BadRequest,
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
			mockAdmissionController: &mockAdmissionController{admitErr: nil},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleErr = errors.New("simulated scheduler failure")
			},
			wantErrCode:            errutil.InferencePoolResourceExhausted,
			inferenceObjectiveName: objectiveName,
		},
		{
			name: "scheduler returns nil result and nil error",
			reqBodyMap: map[string]any{
				"model":  model,
				"prompt": "prompt for nil,nil scheduler return",
			},
			mockAdmissionController: &mockAdmissionController{admitErr: nil},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = nil
				m.scheduleErr = nil
			},
			wantErrCode:            errutil.Internal,
			inferenceObjectiveName: objectiveName,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			mockSched := &mockScheduler{}
			if test.schedulerMockSetup != nil {
				test.schedulerMockSetup(mockSched)
			}
			director := NewDirectorWithConfig(ds, mockSched, test.mockAdmissionController, NewConfig())

			reqCtx := &handlers.RequestContext{
				Request: &handlers.Request{
					// Create a copy of the map for each test run to avoid mutation issues.
					Body: make(map[string]any),
					Headers: map[string]string{
						requtil.RequestIdHeaderKey: "test-req-id-" + test.name, // Ensure a default request ID
					},
				},
				ObjectiveKey:    test.inferenceObjectiveName,
				TargetModelName: test.initialTargetModelName,
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
				assert.Equal(t, test.wantReqCtx.ObjectiveKey, returnedReqCtx.ObjectiveKey, "reqCtx.Model mismatch")
				assert.Equal(t, test.wantReqCtx.TargetModelName, returnedReqCtx.TargetModelName,
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
			metadata.SubsetFilterNamespace: map[string]any{
				metadata.SubsetFilterKey: data,
			},
		}
	}

	pod1 := &backend.Pod{
		NamespacedName: types.NamespacedName{Name: "pod1"},
		Address:        "10.0.0.1",
		Labels:         map[string]string{},
	}

	pod2 := &backend.Pod{
		NamespacedName: types.NamespacedName{Name: "pod2"},
		Address:        "10.0.0.2",
		Labels:         map[string]string{},
	}

	testInput := []backendmetrics.PodMetrics{
		&backendmetrics.FakePodMetrics{Pod: pod1},
		&backendmetrics.FakePodMetrics{Pod: pod2},
	}

	tests := []struct {
		name     string
		metadata map[string]any
		output   []backendmetrics.PodMetrics
	}{
		{
			name:     "SubsetFilter, filter not present — return all pods",
			metadata: map[string]any{},
			output:   testInput,
		},
		{
			name:     "SubsetFilter, namespace present filter not present — return all pods",
			metadata: map[string]any{metadata.SubsetFilterNamespace: map[string]any{}},
			output:   testInput,
		},
		{
			name:     "SubsetFilter, filter present with empty list — return error",
			metadata: makeFilterMetadata([]any{}),
			output:   []backendmetrics.PodMetrics{},
		},
		{
			name:     "SubsetFilter, subset with one matching pod",
			metadata: makeFilterMetadata([]any{"10.0.0.1"}),
			output: []backendmetrics.PodMetrics{
				&backendmetrics.FakePodMetrics{
					Pod: pod1,
				},
			},
		},
		{
			name:     "SubsetFilter, subset with multiple matching pods",
			metadata: makeFilterMetadata([]any{"10.0.0.1", "10.0.0.2", "10.0.0.3"}),
			output:   testInput,
		},
		{
			name:     "SubsetFilter, subset with no matching pods",
			metadata: makeFilterMetadata([]any{"10.0.0.3"}),
			output:   []backendmetrics.PodMetrics{},
		},
	}

	ds := &mockDatastore{pods: testInput}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			director := NewDirectorWithConfig(ds, &mockScheduler{}, &mockAdmissionController{}, NewConfig())

			got := director.getCandidatePodsForScheduling(context.Background(), test.metadata)

			diff := cmp.Diff(test.output, got, cmpopts.SortSlices(func(a, b backendmetrics.PodMetrics) bool {
				return a.GetPod().NamespacedName.String() < b.GetPod().NamespacedName.String()
			}))
			if diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
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

	scheme := runtime.NewScheme()
	_ = clientgoscheme.AddToScheme(scheme)
	_ = v1alpha2.Install(scheme)
	_ = v1.Install(scheme)
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		Build()
	pool := &v1.InferencePool{
		Spec: v1.InferencePoolSpec{
			TargetPorts: []v1.Port{
				{Number: 8000},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			pmf := backendmetrics.NewPodMetricsFactory(&backendmetrics.FakePodMetricsClient{}, time.Millisecond)
			ds := datastore.NewDatastore(t.Context(), pmf, 0)
			err := ds.PoolSet(t.Context(), fakeClient, pool)
			if err != nil {
				t.Errorf("unexpected error setting pool: %s", err)
			}
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

func TestDirector_ApplyWeightedModelRewrite(t *testing.T) {
	_ = logutil.NewTestLoggerIntoContext(context.Background())

	// Mock InferenceModelRewrite objects
	rewriteOld := &v1alpha2.InferenceModelRewrite{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "rewrite-old",
			CreationTimestamp: metav1.Unix(1000, 0),
		},
		Spec: v1alpha2.InferenceModelRewriteSpec{
			Rules: []v1alpha2.InferenceModelRewriteRule{
				{
					Matches: []v1alpha2.Match{
						{
							Model: &v1alpha2.ModelMatch{
								Value: "model-a",
							},
						},
					},
					Targets: []v1alpha2.TargetModel{
						{
							ModelRewrite: "model-a-old-tuned",
							Weight:       100,
						},
					},
				},
			},
		},
	}

	rewriteNew := &v1alpha2.InferenceModelRewrite{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "rewrite-new",
			CreationTimestamp: metav1.Unix(2000, 0),
		},
		Spec: v1alpha2.InferenceModelRewriteSpec{
			Rules: []v1alpha2.InferenceModelRewriteRule{
				{
					Matches: []v1alpha2.Match{
						{
							Model: &v1alpha2.ModelMatch{
								Value: "model-a",
							},
						},
					},
					Targets: []v1alpha2.TargetModel{
						{
							ModelRewrite: "model-a-new-tuned",
							Weight:       100,
						},
					},
				},
			},
		},
	}

	rewriteB := &v1alpha2.InferenceModelRewrite{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "rewrite-b",
			CreationTimestamp: metav1.Unix(1500, 0),
		},
		Spec: v1alpha2.InferenceModelRewriteSpec{
			Rules: []v1alpha2.InferenceModelRewriteRule{
				{
					Matches: []v1alpha2.Match{
						{
							Model: &v1alpha2.ModelMatch{
								Value: "model-b",
							},
						},
					},
					Targets: []v1alpha2.TargetModel{
						{
							ModelRewrite: "model-b-tuned",
							Weight:       100,
						},
					},
				},
			},
		},
	}

	rewriteWeighted := &v1alpha2.InferenceModelRewrite{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "rewrite-weighted",
			CreationTimestamp: metav1.Unix(1200, 0),
		},
		Spec: v1alpha2.InferenceModelRewriteSpec{
			Rules: []v1alpha2.InferenceModelRewriteRule{
				{
					Matches: []v1alpha2.Match{
						{
							Model: &v1alpha2.ModelMatch{
								Value: "model-c",
							},
						},
					},
					Targets: []v1alpha2.TargetModel{
						{
							ModelRewrite: "model-c-v1",
							Weight:       70,
						},
						{
							ModelRewrite: "model-c-v2",
							Weight:       30,
						},
					},
				},
			},
		},
	}

	tests := []struct {
		name           string
		rewrites       []*v1alpha2.InferenceModelRewrite
		incomingModel  string
		expectedTarget []string
		initialTarget  string // Initial value of reqCtx.TargetModelName
	}{
		{
			name:           "no rewrites",
			rewrites:       []*v1alpha2.InferenceModelRewrite{},
			incomingModel:  "model-x",
			expectedTarget: []string{"model-x"},
			initialTarget:  "model-x",
		},
		{
			name:           "single matching rewrite",
			rewrites:       []*v1alpha2.InferenceModelRewrite{rewriteB},
			incomingModel:  "model-b",
			expectedTarget: []string{"model-b-tuned"},
			initialTarget:  "model-b",
		},
		{
			name:           "no matching rewrite",
			rewrites:       []*v1alpha2.InferenceModelRewrite{rewriteB},
			incomingModel:  "model-x",
			expectedTarget: []string{"model-x"},
			initialTarget:  "model-x",
		},
		{
			name:           "oldest rewrite wins for duplicate model",
			rewrites:       []*v1alpha2.InferenceModelRewrite{rewriteNew, rewriteOld}, // New is first, but Old has older timestamp
			incomingModel:  "model-a",
			expectedTarget: []string{"model-a-old-tuned"},
			initialTarget:  "model-a",
		},
		{
			name:           "weighted rewrite applied (probabilistic check)",
			rewrites:       []*v1alpha2.InferenceModelRewrite{rewriteWeighted},
			incomingModel:  "model-c",
			initialTarget:  "model-c",
			expectedTarget: []string{"model-c-v1", "model-c-v2"},
		},
		{
			name:           "initial TargetModelName is respected if no rewrite matches",
			rewrites:       []*v1alpha2.InferenceModelRewrite{rewriteB},
			incomingModel:  "model-x",
			initialTarget:  "pre-existing-target",
			expectedTarget: []string{"pre-existing-target"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			mockDs := &mockDatastore{rewrites: test.rewrites}
			director := NewDirectorWithConfig(mockDs, &mockScheduler{}, &mockAdmissionController{}, NewConfig())

			reqCtx := &handlers.RequestContext{
				IncomingModelName: test.incomingModel,
				TargetModelName:   test.initialTarget,
			}

			director.applyWeightedModelRewrite(reqCtx)
			assert.Contains(t, test.expectedTarget, reqCtx.TargetModelName, "TargetModelName mismatch")
		})
	}
}

func TestDirector_SelectWeightedModel(t *testing.T) {
	tests := []struct {
		name           string
		targets        []v1alpha2.TargetModel
		possibleModels map[string]bool // For probabilistic cases
	}{
		{
			name: "single target",
			targets: []v1alpha2.TargetModel{
				{ModelRewrite: "model-a", Weight: 100},
			},
			possibleModels: map[string]bool{"model-a": true},
		},
		{
			name: "multiple targets, equal weight",
			targets: []v1alpha2.TargetModel{
				{ModelRewrite: "model-a", Weight: 50},
				{ModelRewrite: "model-b", Weight: 50},
			},
			possibleModels: map[string]bool{"model-a": true, "model-b": true},
		},
		{
			name: "multiple targets, different weights",
			targets: []v1alpha2.TargetModel{
				{ModelRewrite: "model-x", Weight: 70},
				{ModelRewrite: "model-y", Weight: 30},
			},
			possibleModels: map[string]bool{"model-x": true, "model-y": true},
		},
		{
			name: "zero total weight, distribute evenly",
			targets: []v1alpha2.TargetModel{
				{ModelRewrite: "model-z1", Weight: 0},
				{ModelRewrite: "model-z2", Weight: 0},
			},
			possibleModels: map[string]bool{"model-z1": true, "model-z2": true},
		},
	}

	director := &Director{}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Run multiple times to check distribution
			counter := make(map[string]int)
			numRuns := 1000
			for i := 0; i < numRuns; i++ {
				selected := director.selectWeightedModel(test.targets)
				counter[selected]++
			}

			// Assert that all selected models are within the possible models
			for model := range counter {
				if _, ok := test.possibleModels[model]; !ok {
					t.Errorf("Selected model %s is not in possible models %v", model, test.possibleModels)
				}
			}

			// Basic check for distribution (e.g., if 70/30, expect roughly 700/300)
			if len(test.targets) > 1 {
				totalWeight := int32(0)
				for _, target := range test.targets {
					totalWeight += target.Weight
				}

				if totalWeight == 0 { // Special case for zero total weight
					for _, target := range test.targets {
						expectedCount := numRuns / len(test.targets)
						assert.InDelta(t, expectedCount, counter[target.ModelRewrite], float64(numRuns)/float64(len(test.targets))*0.2, "Distribution for %s is off", target.ModelRewrite)
					}
				} else {
					for _, target := range test.targets {
						expectedCount := float64(numRuns) * (float64(target.Weight) / float64(totalWeight))
						assert.InDelta(t, expectedCount, float64(counter[target.ModelRewrite]), expectedCount*0.2, "Distribution for %s is off", target.ModelRewrite)
					}
				}
			}
		})
	}
}

func TestDirector_HandleResponseReceived(t *testing.T) {
	pr1 := newTestResponseReceived("pr1")

	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	ds := datastore.NewDatastore(t.Context(), nil, 0)
	mockSched := &mockScheduler{}
	director := NewDirectorWithConfig(ds, mockSched, &mockAdmissionController{}, NewConfig().WithResponseReceivedPlugins(pr1))

	reqCtx := &handlers.RequestContext{
		Request: &handlers.Request{
			Headers: map[string]string{
				requtil.RequestIdHeaderKey: "test-req-id-for-response",
			},
		},
		Response: &handlers.Response{ // Simulate some response headers
			Headers: map[string]string{"X-Test-Response-Header": "TestValue"},
		},

		TargetPod: &backend.Pod{NamespacedName: types.NamespacedName{Namespace: "namespace1", Name: "test-pod-name"}},
	}

	_, err := director.HandleResponseReceived(ctx, reqCtx)
	if err != nil {
		t.Fatalf("HandleResponse() returned unexpected error: %v", err)
	}

	if diff := cmp.Diff("test-req-id-for-response", pr1.lastRespOnResponse.RequestId); diff != "" {
		t.Errorf("Scheduler.OnResponse RequestId mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(reqCtx.Response.Headers, pr1.lastRespOnResponse.Headers); diff != "" {
		t.Errorf("Scheduler.OnResponse Headers mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff("namespace1/test-pod-name", pr1.lastTargetPodOnResponse); diff != "" {
		t.Errorf("Scheduler.OnResponse TargetPodName mismatch (-want +got):\n%s", diff)
	}
}

func TestDirector_HandleResponseStreaming(t *testing.T) {
	ps1 := newTestResponseStreaming("ps1")

	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	ds := datastore.NewDatastore(t.Context(), nil, 0)
	mockSched := &mockScheduler{}
	director := NewDirectorWithConfig(ds, mockSched, nil, NewConfig().WithResponseStreamingPlugins(ps1))

	reqCtx := &handlers.RequestContext{
		Request: &handlers.Request{
			Headers: map[string]string{
				requtil.RequestIdHeaderKey: "test-req-id-for-streaming",
			},
		},
		Response: &handlers.Response{
			Headers: map[string]string{"X-Test-Streaming-Header": "StreamValue"},
		},
		TargetPod: &backend.Pod{NamespacedName: types.NamespacedName{Namespace: "namespace1", Name: "test-pod-name"}},
	}

	_, err := director.HandleResponseBodyStreaming(ctx, reqCtx)
	if err != nil {
		t.Fatalf("HandleResponseBodyStreaming() returned unexpected error: %v", err)
	}

	if diff := cmp.Diff("test-req-id-for-streaming", ps1.lastRespOnStreaming.RequestId); diff != "" {
		t.Errorf("Scheduler.OnStreaming RequestId mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(reqCtx.Response.Headers, ps1.lastRespOnStreaming.Headers); diff != "" {
		t.Errorf("Scheduler.OnStreaming Headers mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff("namespace1/test-pod-name", ps1.lastTargetPodOnStreaming); diff != "" {
		t.Errorf("Scheduler.OnStreaming TargetPodName mismatch (-want +got):\n%s", diff)
	}
}

func TestDirector_HandleResponseComplete(t *testing.T) {
	pc1 := newTestResponseComplete("pc1")

	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	ds := datastore.NewDatastore(t.Context(), nil, 0)
	mockSched := &mockScheduler{}
	director := NewDirectorWithConfig(ds, mockSched, nil, NewConfig().WithResponseCompletePlugins(pc1))

	reqCtx := &handlers.RequestContext{
		Request: &handlers.Request{
			Headers: map[string]string{
				requtil.RequestIdHeaderKey: "test-req-id-for-complete",
			},
		},
		Response: &handlers.Response{
			Headers: map[string]string{"X-Test-Complete-Header": "CompleteValue"},
		},
		TargetPod: &backend.Pod{NamespacedName: types.NamespacedName{Namespace: "namespace1", Name: "test-pod-name"}},
	}

	_, err := director.HandleResponseBodyComplete(ctx, reqCtx)
	if err != nil {
		t.Fatalf("HandleResponseBodyComplete() returned unexpected error: %v", err)
	}

	if diff := cmp.Diff("test-req-id-for-complete", pc1.lastRespOnComplete.RequestId); diff != "" {
		t.Errorf("Scheduler.OnComplete RequestId mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(reqCtx.Response.Headers, pc1.lastRespOnComplete.Headers); diff != "" {
		t.Errorf("Scheduler.OnComplete Headers mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff("namespace1/test-pod-name", pc1.lastTargetPodOnComplete); diff != "" {
		t.Errorf("Scheduler.OnComplete TargetPodName mismatch (-want +got):\n%s", diff)
	}
}

const (
	testResponseReceivedType = "test-response-received"
	testPostStreamingType    = "test-response-streaming"
	testPostCompleteType     = "test-response-complete"
)

type testResponseReceived struct {
	tn                      plugins.TypedName
	lastRespOnResponse      *Response
	lastTargetPodOnResponse string
}

type testResponseStreaming struct {
	tn                       plugins.TypedName
	lastRespOnStreaming      *Response
	lastTargetPodOnStreaming string
}

type testResponseComplete struct {
	tn                      plugins.TypedName
	lastRespOnComplete      *Response
	lastTargetPodOnComplete string
}

func newTestResponseReceived(name string) *testResponseReceived {
	return &testResponseReceived{
		tn: plugins.TypedName{Type: testResponseReceivedType, Name: name},
	}
}

func newTestResponseStreaming(name string) *testResponseStreaming {
	return &testResponseStreaming{
		tn: plugins.TypedName{Type: testPostStreamingType, Name: name},
	}
}

func newTestResponseComplete(name string) *testResponseComplete {
	return &testResponseComplete{
		tn: plugins.TypedName{Type: testPostCompleteType, Name: name},
	}
}

func (p *testResponseReceived) TypedName() plugins.TypedName {
	return p.tn
}

func (p *testResponseStreaming) TypedName() plugins.TypedName {
	return p.tn
}

func (p *testResponseComplete) TypedName() plugins.TypedName {
	return p.tn
}

func (p *testResponseReceived) ResponseReceived(_ context.Context, _ *schedulingtypes.LLMRequest, response *Response, targetPod *backend.Pod) {
	p.lastRespOnResponse = response
	p.lastTargetPodOnResponse = targetPod.NamespacedName.String()
}

func (p *testResponseStreaming) ResponseStreaming(_ context.Context, _ *schedulingtypes.LLMRequest, response *Response, targetPod *backend.Pod) {
	p.lastRespOnStreaming = response
	p.lastTargetPodOnStreaming = targetPod.NamespacedName.String()
}

func (p *testResponseComplete) ResponseComplete(_ context.Context, _ *schedulingtypes.LLMRequest, response *Response, targetPod *backend.Pod) {
	p.lastRespOnComplete = response
	p.lastTargetPodOnComplete = targetPod.NamespacedName.String()
}
