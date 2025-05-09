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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/handlers"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	testutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/testing"
)

func TestHandleRequest(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	// Setup datastore
	tsModel := "food-review"
	modelWithTarget := "food-review-0"
	model1 := testutil.MakeInferenceModel("model1").
		CreationTimestamp(metav1.Unix(1000, 0)).
		ModelName(tsModel).ObjRef()
	model2 := testutil.MakeInferenceModel("model2").
		CreationTimestamp(metav1.Unix(1000, 0)).
		ModelName(modelWithTarget).ObjRef()
	pmf := backendmetrics.NewPodMetricsFactory(&backendmetrics.FakePodMetricsClient{}, time.Second)
	ds := datastore.NewDatastore(t.Context(), pmf)
	ds.ModelSetIfOlder(model1)
	ds.ModelSetIfOlder(model2)

	pool := &v1alpha2.InferencePool{
		Spec: v1alpha2.InferencePoolSpec{
			TargetPortNumber: int32(8000),
			Selector: map[v1alpha2.LabelKey]v1alpha2.LabelValue{
				"some-key": "some-val",
			},
		},
	}
	pod := &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}, Status: corev1.PodStatus{PodIP: "address-1"}}
	scheme := runtime.NewScheme()
	_ = clientgoscheme.AddToScheme(scheme)
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		Build()
	if err := ds.PoolSet(ctx, fakeClient, pool); err != nil {
		t.Error(err, "Error while setting inference pool")
	}
	ds.PodUpdateOrAddIfNotExist(pod)

	tests := []struct {
		name         string
		reqBodyMap   map[string]interface{}
		wantErrCode  string
		wantReqCtx   *handlers.RequestContext
		wantRespBody map[string]interface{}
	}{
		{
			name: "successful request",
			reqBodyMap: map[string]interface{}{
				"model":  tsModel,
				"prompt": "test prompt",
			},
			wantReqCtx: &handlers.RequestContext{
				Model:               tsModel,
				ResolvedTargetModel: tsModel,
				TargetPod:           "/pod1",
				TargetEndpoint:      "address-1:8000",
			},
			wantRespBody: map[string]interface{}{
				"model":  tsModel,
				"prompt": "test prompt",
			},
		},
		{
			name: "successful request with target model",
			reqBodyMap: map[string]interface{}{
				"model":  modelWithTarget,
				"prompt": "test prompt",
			},
			wantReqCtx: &handlers.RequestContext{
				Model:               modelWithTarget,
				ResolvedTargetModel: modelWithTarget,
				TargetPod:           "/pod1",
				TargetEndpoint:      "address-1:8000",
			},
			wantRespBody: map[string]interface{}{
				"model":  modelWithTarget,
				"prompt": "test prompt",
			},
		},
		{
			name:        "no model defined, expect err",
			wantErrCode: errutil.BadRequest,
		},
		{
			name: "invalid model defined, expect err",
			reqBodyMap: map[string]interface{}{
				"model":  "non-existent-model",
				"prompt": "test prompt",
			},
			wantErrCode: errutil.BadConfiguration,
		},
		{
			name: "invalid target defined, expect err",
			reqBodyMap: map[string]interface{}{
				"model":  "food-review-1",
				"prompt": "test prompt",
			},
			wantErrCode: errutil.BadConfiguration,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := NewDirector(ds, scheduling.NewScheduler(ds))
			reqCtx := &handlers.RequestContext{
				Request: &handlers.Request{
					Body: test.reqBodyMap,
				},
			}
			reqCtx, err := server.HandleRequest(ctx, reqCtx)

			if test.wantErrCode != "" {
				if err == nil {
					t.Fatalf("HandleRequestBody should have returned an error containing '%s', but got nil", test.wantErrCode)
				}
				if !strings.Contains(err.Error(), test.wantErrCode) {
					t.Fatalf("HandleRequestBody returned error '%v', which does not contain expected substring '%s'", err, test.wantErrCode)
				}
				return
			}

			if err != nil {
				t.Fatalf("HandleRequestBody returned unexpected error: %v", err)
			}

			if test.wantReqCtx != nil {
				if diff := cmp.Diff(test.wantReqCtx.Model, reqCtx.Model); diff != "" {
					t.Errorf("HandleRequestBody returned unexpected reqCtx.Model, diff(-want, +got): %v", diff)
				}
				if diff := cmp.Diff(test.wantReqCtx.ResolvedTargetModel, reqCtx.ResolvedTargetModel); diff != "" {
					t.Errorf("HandleRequestBody returned unexpected reqCtx.ResolvedTargetModel, diff(-want, +got): %v", diff)
				}
				if diff := cmp.Diff(test.wantReqCtx.TargetPod, reqCtx.TargetPod); diff != "" {
					t.Errorf("HandleRequestBody returned unexpected reqCtx.TargetPod, diff(-want, +got): %v", diff)
				}
				if diff := cmp.Diff(test.wantReqCtx.TargetEndpoint, reqCtx.TargetEndpoint); diff != "" {
					t.Errorf("HandleRequestBody returned unexpected reqCtx.TargetEndpoint, diff(-want, +got): %v", diff)
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
			pmf := metrics.NewPodMetricsFactory(&metrics.FakePodMetricsClient{}, time.Millisecond)
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
