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

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
)

var (
	pod1 = &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod1",
			Namespace: "default",
		},
	}
	initial = &Metrics{
		WaitingQueueSize:    0,
		KVCacheUsagePercent: 0.2,
		MaxActiveModels:     2,
		ActiveModels: map[string]int{
			"foo": 1,
			"bar": 1,
		},
		WaitingModels: map[string]int{},
	}
	updated = &Metrics{
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
	pmc := &FakePodMetricsClient{}
	pmf := NewPodMetricsFactory(pmc, time.Millisecond)

	// The refresher is initialized with empty metrics.
	pm := pmf.NewPodMetrics(ctx, pod1, &fakeDataStore{})

	namespacedName := types.NamespacedName{Name: pod1.Name, Namespace: pod1.Namespace}
	// Use SetRes to simulate an update of metrics from the pod.
	// Verify that the metrics are updated.
	pmc.SetRes(map[types.NamespacedName]*Metrics{namespacedName: initial})
	condition := func(collect *assert.CollectT) {
		assert.True(collect, cmp.Equal(pm.GetMetrics(), initial, cmpopts.IgnoreFields(Metrics{}, "UpdateTime")))
	}
	assert.EventuallyWithT(t, condition, time.Second, time.Millisecond)

	// Stop the loop, and simulate metric update again, this time the PodMetrics won't get the
	// new update.
	pm.StopRefreshLoop()
	pmc.SetRes(map[types.NamespacedName]*Metrics{namespacedName: updated})
	// Still expect the same condition (no metrics update).
	assert.EventuallyWithT(t, condition, time.Second, time.Millisecond)
}

type fakeDataStore struct{}

func (f *fakeDataStore) PoolGet() (*v1alpha2.InferencePool, error) {
	return &v1alpha2.InferencePool{Spec: v1alpha2.InferencePoolSpec{TargetPortNumber: 8000}}, nil
}
func (f *fakeDataStore) PodGetAll() []PodMetrics {
	// Not implemented.
	return nil
}
func (f *fakeDataStore) PodList(func(PodMetrics) bool) []PodMetrics {
	// Not implemented.
	return nil
}
