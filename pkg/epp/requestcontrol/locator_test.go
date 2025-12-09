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
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"
)

// --- DatastorePodLocator Tests ---

func TestDatastorePodLocator_Locate(t *testing.T) {
	t.Parallel()

	podA := makeMockPodMetrics("pod-a", "10.0.0.1")
	podB := makeMockPodMetrics("pod-b", "10.0.0.2")
	podC := makeMockPodMetrics("pod-c", "10.0.0.3")

	allPods := []backendmetrics.PodMetrics{podA, podB, podC}
	mockDS := &mockDatastore{pods: allPods}
	locator := NewDatastorePodLocator(mockDS)

	tests := []struct {
		name           string
		metadata       map[string]any
		expectedPodIPs []string
	}{
		{
			name:           "Nil metadata returns all pods",
			metadata:       nil,
			expectedPodIPs: []string{"10.0.0.1", "10.0.0.2", "10.0.0.3"},
		},
		{
			name:           "Empty metadata returns all pods",
			metadata:       map[string]any{},
			expectedPodIPs: []string{"10.0.0.1", "10.0.0.2", "10.0.0.3"},
		},
		{
			name: "Metadata without subset namespace returns all pods",
			metadata: map[string]any{
				"other-filter": "value",
			},
			expectedPodIPs: []string{"10.0.0.1", "10.0.0.2", "10.0.0.3"},
		},
		{
			name: "Subset filter with single match",
			metadata: makeMetadataWithSubset([]any{
				"10.0.0.1:8080",
			}),
			expectedPodIPs: []string{"10.0.0.1"},
		},
		{
			name: "Subset filter with multiple matches",
			metadata: makeMetadataWithSubset([]any{
				"10.0.0.1:8080",
				"10.0.0.3:9090",
			}),
			expectedPodIPs: []string{"10.0.0.1", "10.0.0.3"},
		},
		{
			name: "Subset filter with no matches (Scale-from-Zero scenario)",
			metadata: makeMetadataWithSubset([]any{
				"192.168.1.1:8080", // Does not exist in mockDS
			}),
			expectedPodIPs: []string{},
		},
		{
			name: "Subset filter is present but list is empty",
			metadata: map[string]any{
				metadata.SubsetFilterNamespace: map[string]any{
					metadata.SubsetFilterKey: []any{},
				},
			},
			expectedPodIPs: []string{},
		},
		{
			name: "Subset filter contains malformed data (non-string)",
			metadata: makeMetadataWithSubset([]any{
				"10.0.0.1:8080",
				12345, // Should be ignored
			}),
			expectedPodIPs: []string{"10.0.0.1"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			result := locator.Locate(context.Background(), tc.metadata)

			var gotIPs []string
			for _, pm := range result {
				gotIPs = append(gotIPs, pm.GetMetadata().GetIPAddress())
			}
			assert.ElementsMatch(t, tc.expectedPodIPs, gotIPs, "Locate returned unexpected set of pods")
		})
	}
}

// --- CachedPodLocator Tests ---

func TestCachedPodLocator_CachingBehavior(t *testing.T) {
	t.Parallel()

	mockDelegate := &mockPodLocator{
		result: []backendmetrics.PodMetrics{makeMockPodMetrics("p1", "1.1.1.1")},
	}

	// Use a short TTL for testing.
	ttl := 20 * time.Millisecond
	cached := NewCachedPodLocator(context.Background(), mockDelegate, ttl)
	meta := makeMetadataWithSubset([]any{"1.1.1.1:80"})

	// 1. First Call: Should hit delegate
	res1 := cached.Locate(context.Background(), meta)
	require.Len(t, res1, 1)
	assert.Equal(t, 1, mockDelegate.callCount(), "Expected delegate to be called on first access")

	// 2. Second Call (Immediate): Should hit cache
	res2 := cached.Locate(context.Background(), meta)
	require.Len(t, res2, 1)
	assert.Equal(t, 1, mockDelegate.callCount(), "Expected delegate NOT to be called again (cache hit)")

	// 3. Wait for Expiry
	time.Sleep(ttl * 2)

	// 4. Third Call (Expired): Should hit delegate again
	res3 := cached.Locate(context.Background(), meta)
	require.Len(t, res3, 1)
	assert.Equal(t, 2, mockDelegate.callCount(), "Expected delegate to be called after TTL expiry")
}

func TestCachedPodLocator_CacheKeyDeterminism(t *testing.T) {
	t.Parallel()

	mockDelegate := &mockPodLocator{}
	cached := NewCachedPodLocator(context.Background(), mockDelegate, time.Minute)

	// Scenario: subset [A, B] should generate same cache key as [B, A].
	metaOrder1 := makeMetadataWithSubset([]any{"10.0.0.1:80", "10.0.0.2:80"})
	metaOrder2 := makeMetadataWithSubset([]any{"10.0.0.2:80", "10.0.0.1:80"})

	cached.Locate(context.Background(), metaOrder1)
	assert.Equal(t, 1, mockDelegate.callCount(), "Initial call")

	cached.Locate(context.Background(), metaOrder2)
	assert.Equal(t, 1, mockDelegate.callCount(), "Different order of subset endpoints should hit the same cache entry")
}

func TestCachedPodLocator_Concurrency_ThunderingHerd(t *testing.T) {
	t.Parallel()

	// Simulate a slow delegate to exacerbate race conditions.
	mockDelegate := &mockPodLocator{
		delay: 10 * time.Millisecond,
		result: []backendmetrics.PodMetrics{
			makeMockPodMetrics("p1", "1.1.1.1"),
		},
	}

	cached := NewCachedPodLocator(context.Background(), mockDelegate, 100*time.Millisecond)
	meta := makeMetadataWithSubset([]any{"1.1.1.1:80"})

	concurrency := 50
	var wg sync.WaitGroup
	wg.Add(concurrency)

	start := make(chan struct{})

	// Spawn N routines trying to Locate simultaneously.
	for range concurrency {
		go func() {
			defer wg.Done()
			<-start // Synchronize start.
			res := cached.Locate(context.Background(), meta)
			assert.Len(t, res, 1)
		}()
	}

	close(start) // Release the hounds.
	wg.Wait()

	// Ideally, the delegate should be called exactly once.
	// However, due to double-checked locking, strict 'once' is guaranteed.
	assert.Equal(t, 1, mockDelegate.callCount(), "Delegate should be called exactly once despite concurrent access")
}

func TestCachedPodLocator_DifferentSubsetsAreIsolated(t *testing.T) {
	t.Parallel()

	mockDelegate := &mockPodLocator{}
	cached := NewCachedPodLocator(context.Background(), mockDelegate, time.Minute)

	metaA := makeMetadataWithSubset([]any{"10.0.0.1:80"})
	metaB := makeMetadataWithSubset([]any{"10.0.0.2:80"})

	cached.Locate(context.Background(), metaA)
	assert.Equal(t, 1, mockDelegate.callCount())

	cached.Locate(context.Background(), metaB)
	assert.Equal(t, 2, mockDelegate.callCount(), "Different subsets must trigger distinct delegate calls")
}

func TestCachedPodLocator_CacheIsolation_EmptyVsDefault(t *testing.T) {
	t.Parallel()

	mockDelegate := &mockPodLocator{
		result: []backendmetrics.PodMetrics{makeMockPodMetrics("p1", "1.1.1.1")},
	}
	cached := NewCachedPodLocator(context.Background(), mockDelegate, time.Minute)

	// 1. Request All Pods (No Metadata)
	res1 := cached.Locate(context.Background(), nil)
	assert.NotEmpty(t, res1)
	assert.Equal(t, 1, mockDelegate.callCount())

	// 2. Request Empty Subset (Should be distinct key)
	// We expect the delegate to be called AGAIN because "__explicit_empty__" is not "__default__".
	metaEmpty := makeMetadataWithSubset([]any{})
	_ = cached.Locate(context.Background(), metaEmpty)
	assert.Equal(t, 2, mockDelegate.callCount(), "Empty subset should not hit the default cache key")
}

// --- Helpers & Mocks ---

// mockPodLocator implements contracts.PodLocator.
type mockPodLocator struct {
	mu     sync.Mutex
	calls  int
	delay  time.Duration
	result []backendmetrics.PodMetrics
}

func (m *mockPodLocator) Locate(ctx context.Context, _ map[string]any) []backendmetrics.PodMetrics {
	m.mu.Lock()
	m.calls++
	delay := m.delay
	result := m.result
	m.mu.Unlock()

	if delay > 0 {
		time.Sleep(delay)
	}
	return result
}

func (m *mockPodLocator) callCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.calls
}

func makeMockPodMetrics(name, ip string) backendmetrics.PodMetrics {
	return &backendmetrics.FakePodMetrics{
		Pod: &backend.Pod{
			NamespacedName: types.NamespacedName{Namespace: "default", Name: name},
			Address:        ip,
		},
	}
}

func makeMetadataWithSubset(endpoints []any) map[string]any {
	return map[string]any{
		metadata.SubsetFilterNamespace: map[string]any{
			metadata.SubsetFilterKey: endpoints,
		},
	}
}
