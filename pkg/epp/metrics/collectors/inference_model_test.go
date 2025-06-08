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

package collectors

import (
	"context"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/component-base/metrics/testutil"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
)

func TestNoInferenceModelMetricsCollected(t *testing.T) {
	pmf := backendmetrics.NewPodMetricsFactory(&backendmetrics.FakePodMetricsClient{}, 0)
	ds := datastore.NewDatastore(context.Background(), pmf)

	collector := &inferenceModelMetricsCollector{
		ds: ds,
	}

	if err := testutil.CollectAndCompare(collector, strings.NewReader(""), ""); err != nil {
		t.Fatal(err)
	}
}

func TestInferenceModelMetricsCollected(t *testing.T) {
	pmf := backendmetrics.NewPodMetricsFactory(&backendmetrics.FakePodMetricsClient{}, 0)
	ds := datastore.NewDatastore(context.Background(), pmf)

	scheme := runtime.NewScheme()
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		Build()

	pool := &v1alpha2.InferencePool{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pool",
		},
		Spec: v1alpha2.InferencePoolSpec{
			TargetPortNumber: 8000,
		},
	}
	_ = ds.PoolSet(context.Background(), fakeClient, pool)

	// Add multiple models
	model1 := &v1alpha2.InferenceModel{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-model-1",
		},
		Spec: v1alpha2.InferenceModelSpec{
			ModelName: "llama-3-8b",
			PoolRef: v1alpha2.PoolObjectReference{
			},
		},
	}
	ds.ModelSetIfOlder(model1)

	model2 := &v1alpha2.InferenceModel{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-model-2",
		},
		Spec: v1alpha2.InferenceModelSpec{
			ModelName: "llama-3-70b",
			PoolRef: v1alpha2.PoolObjectReference{
				Name: "test-pool",
			},
		},
	}
	ds.ModelSetIfOlder(model2)

	collector := &inferenceModelMetricsCollector{
		ds: ds,
	}

	err := testutil.CollectAndCompare(collector, strings.NewReader(`
		# HELP inference_model_ready [ALPHA] Indicates which InferenceModels are ready to serve by the epp. Value 1 indicates the model is tracked and ready, 0 indicates not ready.
		# TYPE inference_model_ready gauge
		inference_model_ready{model_name="llama-3-70b",pool_name="test-pool"} 1
		inference_model_ready{model_name="llama-3-8b",pool_name="test-pool"} 1
`), "inference_model_ready")
	if err != nil {
		t.Fatal(err)
	}
}
