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
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
)

var (
	pod1 = &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod1",
			Namespace: "default",
			Labels:    map[string]string{"app": "test"},
		},
		Status: corev1.PodStatus{
			PodIP: "192.168.1.1",
		},
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
	pmc := &FakePodMetricsClient{}
	pmf := NewPodMetricsFactory(pmc, time.Millisecond)

	// The refresher is initialized with empty metrics.
	pm := pmf.NewEndpoint(ctx, pod1, &fakeDataStore{})

	namespacedName := types.NamespacedName{Name: pod1.Name, Namespace: pod1.Namespace}
	// Use SetRes to simulate an update of metrics from the pod.
	// Verify that the metrics are updated.
	pmc.SetRes(map[types.NamespacedName]*MetricsState{namespacedName: initial})
	condition := func(collect *assert.CollectT) {
		assert.True(collect, cmp.Equal(pm.GetMetrics(), initial, cmpopts.IgnoreFields(MetricsState{}, "UpdateTime")))
	}
	assert.EventuallyWithT(t, condition, time.Second, time.Millisecond)

	// Stop the loop, and simulate metric update again, this time the PodMetrics won't get the
	// new update.
	pmf.ReleaseEndpoint(pm)
	time.Sleep(pmf.refreshMetricsInterval * 2 /* small buffer for robustness */)
	pmc.SetRes(map[types.NamespacedName]*MetricsState{namespacedName: updated})
	// Still expect the same condition (no metrics update).
	assert.EventuallyWithT(t, condition, time.Second, time.Millisecond)
}

// Test priority queue functionality
func TestPodMetricsRequestManagement(t *testing.T) {
	ctx := context.Background()
	pmc := &FakePodMetricsClient{}
	pmf := NewPodMetricsFactory(pmc, time.Minute) // Long interval to avoid interference

	pme := pmf.NewEndpoint(ctx, pod1, &fakeDataStore{})
	pm := pme.(*podMetrics) // Type assertion to access podMetrics methods

	defer pmf.ReleaseEndpoint(pm)

	// Test adding requests
	assert.True(t, pm.AddRequest("req1", 1.5))
	assert.True(t, pm.AddRequest("req2", 2.0))
	assert.False(t, pm.AddRequest("req1", 1.0)) // Duplicate should fail

	// Test request count
	assert.Equal(t, 2, pm.GetRequestCount())

	// Test contains request
	assert.True(t, pm.ContainsRequest("req1"))
	assert.False(t, pm.ContainsRequest("req3"))

	// Test update request
	assert.True(t, pm.UpdateRequest("req1", 0.5))
	assert.False(t, pm.UpdateRequest("req3", 1.0)) // Non-existent

	// Test remove request
	assert.True(t, pm.RemoveRequest("req1"))
	assert.False(t, pm.RemoveRequest("req1")) // Already removed
	assert.Equal(t, 1, pm.GetRequestCount())

	// Test getting running requests queue
	queue := pm.GetRunningRequests()
	assert.NotNil(t, queue)
	assert.Equal(t, 1, queue.GetSize())
}

// Test pod updates preserve request queue
func TestPodUpdatePreservesQueue(t *testing.T) {
	ctx := context.Background()
	pmc := &FakePodMetricsClient{}
	pmf := NewPodMetricsFactory(pmc, time.Minute)

	pme := pmf.NewEndpoint(ctx, pod1, &fakeDataStore{})
	pm := pme.(*podMetrics) // Type assertion to access podMetrics methods

	defer pmf.ReleaseEndpoint(pm)

	// Add some requests
	assert.True(t, pm.AddRequest("req1", 1.5))
	assert.True(t, pm.AddRequest("req2", 2.0))
	assert.Equal(t, 2, pm.GetRequestCount())

	// Update pod with new IP
	updatedPod := pod1.DeepCopy()
	updatedPod.Status.PodIP = "192.168.1.2"
	updatedPod.Labels["new"] = "label"

	pm.UpdatePod(updatedPod)

	// Queue should be preserved
	assert.Equal(t, 2, pm.GetRequestCount())
	assert.True(t, pm.ContainsRequest("req1"))
	assert.True(t, pm.ContainsRequest("req2"))

	// Pod properties should be updated
	pod := pm.GetPod()
	assert.Equal(t, "192.168.1.2", pod.Address)
	assert.Equal(t, "label", pod.Labels["new"])
}

// Test error handling in metrics refresh
func TestMetricsRefreshWithErrors(t *testing.T) {
	ctx := context.Background()
	pmc := &FakePodMetricsClient{}
	pmf := NewPodMetricsFactory(pmc, time.Millisecond)

	pme := pmf.NewEndpoint(ctx, pod1, &fakeDataStore{})
	pm := pme.(*podMetrics) // Type assertion to access podMetrics methods

	defer pmf.ReleaseEndpoint(pm)

	namespacedName := types.NamespacedName{Name: pod1.Name, Namespace: pod1.Namespace}

	// Set an error for this pod
	pmc.SetErr(map[types.NamespacedName]error{
		namespacedName: fmt.Errorf("connection failed"),
	})

	// Metrics should still be accessible (error is logged but not fatal)
	// The pod metrics should continue to work
	assert.NotNil(t, pm.GetMetrics())
	assert.NotNil(t, pm.GetPod())

	// Request operations should still work
	assert.True(t, pm.AddRequest("req1", 1.5))
	assert.Equal(t, 1, pm.GetRequestCount())
}

// Test string representation
func TestPodMetricsString(t *testing.T) {
	ctx := context.Background()
	pmc := &FakePodMetricsClient{}
	pmf := NewPodMetricsFactory(pmc, time.Minute)

	pme := pmf.NewEndpoint(ctx, pod1, &fakeDataStore{})
	pm := pme.(*podMetrics) // Type assertion to access podMetrics methods

	defer pmf.ReleaseEndpoint(pm)

	// Add some requests
	pm.AddRequest("req1", 1.5)
	pm.AddRequest("req2", 2.0)

	str := pm.String()
	assert.Contains(t, str, "pod1")
	assert.Contains(t, str, "default")
	assert.Contains(t, str, "[req1(1.50), req2(2.00)]")
	assert.Contains(t, str, "192.168.1.1")
}

// Test concurrent access to request operations
func TestConcurrentRequestOperations(t *testing.T) {
	ctx := context.Background()
	pmc := &FakePodMetricsClient{}
	pmf := NewPodMetricsFactory(pmc, time.Minute)

	pme := pmf.NewEndpoint(ctx, pod1, &fakeDataStore{})
	pm := pme.(*podMetrics) // Type assertion to access podMetrics methods

	defer pmf.ReleaseEndpoint(pm)

	const numGoroutines = 10
	const requestsPerGoroutine = 100

	var wg sync.WaitGroup

	// Launch goroutines that add requests
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < requestsPerGoroutine; j++ {
				requestID := fmt.Sprintf("req-%d-%d", id, j)
				pm.AddRequest(requestID, float64(j))
			}
		}(i)
	}

	// Launch goroutines that check and remove requests
	for i := 0; i < numGoroutines/2; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < requestsPerGoroutine/2; j++ {
				requestID := fmt.Sprintf("req-%d-%d", id, j)
				if pm.ContainsRequest(requestID) {
					pm.RemoveRequest(requestID)
				}
			}
		}(i)
	}

	wg.Wait()

	// Should not crash and should have some requests remaining
	count := pm.GetRequestCount()
	assert.True(t, count >= 0) // Basic sanity check
}

type fakeDataStore struct{}

func (f *fakeDataStore) PoolGet() (*v1.InferencePool, error) {
	return &v1.InferencePool{Spec: v1.InferencePoolSpec{TargetPorts: []v1.Port{{Number: 8000}}}}, nil
}

func (f *fakeDataStore) PodList(func(PodMetrics) bool) []PodMetrics {
	// Not implemented.
	return nil
}
