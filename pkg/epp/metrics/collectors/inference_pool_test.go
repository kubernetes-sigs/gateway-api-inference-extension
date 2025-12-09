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
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/component-base/metrics/testutil"

	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	poolutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/pool"
)

var (
	pod1 = &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
		},
	}
	pod1NamespacedName = types.NamespacedName{Name: pod1.Name + "-rank-0", Namespace: pod1.Namespace}
	pod1Metrics        = &backendmetrics.MetricsState{
		WaitingQueueSize:    100,
		KVCacheUsagePercent: 0.2,
		MaxActiveModels:     2,
	}
)

func TestNoMetricsCollected(t *testing.T) {
	period := time.Second
	factories := []datalayer.EndpointFactory{
		backendmetrics.NewPodMetricsFactory(&backendmetrics.FakePodMetricsClient{}, period),
		datalayer.NewEndpointFactory([]datalayer.DataSource{&datalayer.FakeDataSource{}}, period),
	}
	for _, epf := range factories {
		ds := datastore.NewDatastore(context.Background(), epf, 0)

		collector := &inferencePoolMetricsCollector{
			ds: ds,
		}

		if err := testutil.CollectAndCompare(collector, strings.NewReader(""), ""); err != nil {
			t.Fatal(err)
		}
	}
}

func TestMetricsCollected(t *testing.T) {
	metrics := map[types.NamespacedName]*backendmetrics.MetricsState{
		pod1NamespacedName: pod1Metrics,
	}
	period := time.Millisecond
	factories := []datalayer.EndpointFactory{
		backendmetrics.NewPodMetricsFactory(&backendmetrics.FakePodMetricsClient{Res: metrics}, period),
		datalayer.NewEndpointFactory([]datalayer.DataSource{&datalayer.FakeDataSource{Metrics: metrics}}, period),
	}
	for _, epf := range factories {
		inferencePool := &v1.InferencePool{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-pool",
			},
			Spec: v1.InferencePoolSpec{
				TargetPorts: []v1.Port{{Number: v1.PortNumber(int32(8000))}},
			},
		}
		ds := datastore.NewDatastore(context.Background(), epf, 0)

		scheme := runtime.NewScheme()
		fakeClient := fake.NewClientBuilder().
			WithScheme(scheme).
			Build()

		_ = ds.PoolSet(context.Background(), fakeClient, poolutil.InferencePoolToEndpointPool(inferencePool))
		_ = ds.PodUpdateOrAddIfNotExist(pod1)

		time.Sleep(1 * time.Second)

		collector := &inferencePoolMetricsCollector{
			ds: ds,
		}
		err := testutil.CollectAndCompare(collector, strings.NewReader(`
		# HELP inference_pool_per_pod_queue_size [ALPHA] The total number of requests pending in the model server queue for each underlying pod.
		# TYPE inference_pool_per_pod_queue_size gauge
		inference_pool_per_pod_queue_size{model_server_pod="pod1-rank-0",name="test-pool"} 100
`), "inference_pool_per_pod_queue_size")
		if err != nil {
			t.Fatal(err)
		}
	}
}
