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
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/source/mocks"
)

var (
	pod1Info = &fwkdl.EndpointMetadata{
		NamespacedName: types.NamespacedName{
			Name:      "pod1-rank-0",
			Namespace: "default",
		},
		PodName: "pod1",
	}
	initial = &MetricsState{
		WaitingQueueSize:    0,
		KVCacheUsagePercent: 0.2,
		MaxActiveModels:     2,
		ActiveModels: map[string]int{
			"foo": 1,
			"bar": 1,
		},
		WaitingModels: map[string]int{},
	}
	updated = &MetricsState{
		WaitingQueueSize:    9999,
		KVCacheUsagePercent: 0.99,
		MaxActiveModels:     99,
		ActiveModels: map[string]int{
			"foo": 1,
			"bar": 1,
		},
		WaitingModels: map[string]int{},
	}
)

func TestMetricsRefresh(t *testing.T) {
	ctx := context.Background()

	// Test both legacy (PodMetricsFactory) and new (EndpointLifecycle) approaches
	testCases := []struct {
		name    string
		factory datalayer.EndpointFactory
		setupFn func() (datalayer.EndpointFactory, func(map[types.NamespacedName]*MetricsState))
	}{
		{
			name: "Legacy PodMetricsFactory",
			setupFn: func() (datalayer.EndpointFactory, func(map[types.NamespacedName]*MetricsState)) {
				pmc := &FakePodMetricsClient{}
				pmf := NewPodMetricsFactory(pmc, time.Millisecond)
				setMetrics := func(metrics map[types.NamespacedName]*MetricsState) {
					pmc.SetRes(metrics)
				}
				return pmf, setMetrics
			},
		},
		{
			name: "New EndpointLifecycle with mock DataSource",
			setupFn: func() (datalayer.EndpointFactory, func(map[types.NamespacedName]*MetricsState)) {
				mockDS := &mocks.MetricsDataSource{
					Metrics: make(map[types.NamespacedName]*fwkdl.Metrics),
				}
				epf := datalayer.NewEndpointFactory([]fwkdl.DataSource{mockDS}, time.Millisecond)
				setMetrics := func(metrics map[types.NamespacedName]*MetricsState) {
					// Convert MetricsState to fwkdl.Metrics
					converted := make(map[types.NamespacedName]*fwkdl.Metrics)
					for nn, ms := range metrics {
						converted[nn] = &fwkdl.Metrics{
							WaitingQueueSize:    ms.WaitingQueueSize,
							KVCacheUsagePercent: ms.KVCacheUsagePercent,
							MaxActiveModels:     ms.MaxActiveModels,
							ActiveModels:        ms.ActiveModels,
							WaitingModels:       ms.WaitingModels,
						}
					}
					mockDS.SetMetrics(converted)
				}
				return epf, setMetrics
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			factory, setMetrics := tc.setupFn()

			// The refresher is initialized with empty metrics.
			pm := factory.NewEndpoint(ctx, pod1Info, &FakeRefresherDataStore{})

			// Use setMetrics to simulate an update of metrics from the pod.
			// Verify that the metrics are updated.
			setMetrics(map[types.NamespacedName]*MetricsState{pod1Info.NamespacedName: initial})
			condition := func(collect *assert.CollectT) {
				gotMetrics := pm.GetMetrics()
				assert.Equal(collect, initial.WaitingQueueSize, gotMetrics.WaitingQueueSize)
				assert.Equal(collect, initial.KVCacheUsagePercent, gotMetrics.KVCacheUsagePercent)
				assert.Equal(collect, initial.MaxActiveModels, gotMetrics.MaxActiveModels)
			}
			assert.EventuallyWithT(t, condition, time.Second, time.Millisecond)

			// Stop the loop, and simulate metric update again, this time the endpoint won't get the
			// new update.
			factory.ReleaseEndpoint(pm)
			time.Sleep(time.Millisecond * 10 /* small buffer for robustness */)
			setMetrics(map[types.NamespacedName]*MetricsState{pod1Info.NamespacedName: updated})
			// Still expect the same condition (no metrics update).
			assert.EventuallyWithT(t, condition, time.Second, time.Millisecond)
		})
	}
}

type FakeRefresherDataStore struct{}

func (f *FakeRefresherDataStore) PoolGet() (*datalayer.EndpointPool, error) {
	return &datalayer.EndpointPool{}, nil
}

func (f *FakeRefresherDataStore) PodList(func(PodMetrics) bool) []PodMetrics {
	// Not implemented.
	return nil
}
