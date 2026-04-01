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

package metrics

import (
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	eppmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	poolutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/pool"
)

func TestRefreshPrometheusMetricsAvgValues(t *testing.T) {
	eppmetrics.Register()
	eppmetrics.Reset()

	logger := logr.Discard()

	// Pod1: RunningRequests=0, Queue=1
	// Pod2: RunningRequests=1, Queue=2
	// Correct avg: RunningRequests=0.5, Queue=1.5
	// Integer truncation would give: RunningRequests=0, Queue=1
	ds := &fakeOddMetricsDataStore{}

	refreshPrometheusMetrics(logger, ds, 100*time.Millisecond)

	families, err := ctrlmetrics.Registry.Gather()
	assert.NoError(t, err)

	findGauge := func(name string) float64 {
		for _, f := range families {
			if f.GetName() == name {
				for _, m := range f.GetMetric() {
					return m.GetGauge().GetValue()
				}
			}
		}
		t.Fatalf("metric %s not found", name)
		return 0
	}

	avgRunning := findGauge("inference_pool_average_running_requests")
	avgQueue := findGauge("inference_pool_average_queue_size")

	assert.InDelta(t, 0.5, avgRunning, 0.001, "average running requests should be 0.5, not truncated to 0")
	assert.InDelta(t, 1.5, avgQueue, 0.001, "average queue size should be 1.5, not truncated to 1")
}

type fakeOddMetricsDataStore struct{}

func (f *fakeOddMetricsDataStore) PoolGet() (*datalayer.EndpointPool, error) {
	pool := &v1.InferencePool{Spec: v1.InferencePoolSpec{TargetPorts: []v1.Port{{Number: 8000}}}}
	return poolutil.InferencePoolToEndpointPool(pool), nil
}

func (f *fakeOddMetricsDataStore) PodList(predicate func(fwkdl.Endpoint) bool) []fwkdl.Endpoint {
	pod1 := &fwkdl.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "pod1", Namespace: "default"},
		Address:        "1.2.3.4:5678",
	}
	pod2 := &fwkdl.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "pod2", Namespace: "default"},
		Address:        "1.2.3.4:5679",
	}
	m1 := &fwkdl.Metrics{
		RunningRequestsSize: 0,
		WaitingQueueSize:    1,
		KVCacheUsagePercent: 10.0,
		UpdateTime:          time.Now(),
	}
	m2 := &fwkdl.Metrics{
		RunningRequestsSize: 1,
		WaitingQueueSize:    2,
		KVCacheUsagePercent: 20.0,
		UpdateTime:          time.Now(),
	}
	ep1 := fwkdl.NewEndpoint(pod1, m1)
	ep2 := fwkdl.NewEndpoint(pod2, m2)
	pods := []fwkdl.Endpoint{ep1, ep2}
	res := []fwkdl.Endpoint{}
	for _, pod := range pods {
		if predicate(pod) {
			res = append(res, pod)
		}
	}
	return res
}
