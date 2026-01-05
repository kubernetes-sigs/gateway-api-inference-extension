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

package concurrencydetector

import (
	"context"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

// TestNewPlugin_Configuration validates that the plugin correctly applies defaults and respects explicit configuration
// values.
func TestNewPlugin_Configuration(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name                   string
		config                 Config
		effectiveMax           int64
		effectiveHeadroomBurst int64
	}{
		{
			name:                   "defaults_applied_on_zero_values",
			config:                 Config{}, // Zero values
			effectiveMax:           100,      // DefaultMaxConcurrency
			effectiveHeadroomBurst: 100,      // Default Headroom is 0.0, so limit == max
		},
		{
			name: "explicit_values_respected",
			config: Config{
				MaxConcurrency: 50,
				Headroom:       0.2, // 20% burst
			},
			effectiveMax:           50,
			effectiveHeadroomBurst: 60, // 50 * 1.2 = 60
		},
		{
			name: "negative_max_resets_to_default",
			config: Config{
				MaxConcurrency: -10,
			},
			effectiveMax:           100,
			effectiveHeadroomBurst: 100,
		},
		{
			name: "negative_headroom_resets_to_default",
			config: Config{
				MaxConcurrency: 10,
				Headroom:       -0.5,
			},
			effectiveMax:           10,
			effectiveHeadroomBurst: 10, // Headroom resets to 0.0
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			ctx := context.Background()
			detector := NewDetector(tc.config)
			podName := "test-pod"

			// 1. Verify MaxConcurrency via IsSaturated
			// Drive load up to effectiveMax - 1
			driveLoad(ctx, detector, podName, int(tc.effectiveMax-1))
			require.False(t, detector.IsSaturated(ctx, []backendmetrics.PodMetrics{newFakePodMetric(podName)}),
				"expected NOT saturated at limit-1")

			// Increment to effectiveMax
			driveLoad(ctx, detector, podName, 1)
			require.True(t, detector.IsSaturated(ctx, []backendmetrics.PodMetrics{newFakePodMetric(podName)}),
				"expected saturated at limit")

			// 2. Verify Headroom via Filter
			// Reset state first via DeletePod.
			detector.DeletePod(fullPodName(podName))

			// Drive load to burst limit
			driveLoad(ctx, detector, podName, int(tc.effectiveHeadroomBurst))

			// Filter should KEEP the pod at the burst limit
			kept := detector.Filter(ctx, nil, nil, []schedulingtypes.Pod{newStubSchedulingPod(podName)})
			require.Len(t, kept, 1, "expected pod to be kept at burst limit")

			// Exceed burst limit
			driveLoad(ctx, detector, podName, 1)
			kept = detector.Filter(ctx, nil, nil, []schedulingtypes.Pod{newStubSchedulingPod(podName)})
			require.Len(t, kept, 0, "expected pod to be filtered above burst limit")
		})
	}
}

// TestDetector_IsSaturated verifies the global circuit breaker logic.
// It ensures saturation is reported ONLY when ALL candidate pods are full.
func TestDetector_IsSaturated(t *testing.T) {
	t.Parallel()

	const maxConcurrency = 5
	config := Config{MaxConcurrency: maxConcurrency}

	tests := []struct {
		name           string
		podLoadSetup   map[string]int // Map of PodName -> Request Count
		candidatePods  []string       // Pods passed to IsSaturated
		wantSaturation bool
	}{
		{
			name:           "empty_candidate_list_fail_closed",
			podLoadSetup:   nil,
			candidatePods:  []string{},
			wantSaturation: true,
		},
		{
			name:           "single_pod_with_capacity",
			podLoadSetup:   map[string]int{"pod-a": 4},
			candidatePods:  []string{"pod-a"},
			wantSaturation: false,
		},
		{
			name:           "single_pod_full",
			podLoadSetup:   map[string]int{"pod-a": 5},
			candidatePods:  []string{"pod-a"},
			wantSaturation: true,
		},
		{
			name:           "multi_pod_one_available",
			podLoadSetup:   map[string]int{"pod-a": 5, "pod-b": 4},
			candidatePods:  []string{"pod-a", "pod-b"},
			wantSaturation: false,
		},
		{
			name:           "multi_pod_all_full",
			podLoadSetup:   map[string]int{"pod-a": 5, "pod-b": 6},
			candidatePods:  []string{"pod-a", "pod-b"},
			wantSaturation: true,
		},
		{
			name:           "unknown_pod_assumed_empty",
			podLoadSetup:   nil,
			candidatePods:  []string{"pod-unknown"},
			wantSaturation: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			ctx := context.Background()
			detector := NewDetector(config)

			// Setup load.
			for podName, load := range tc.podLoadSetup {
				driveLoad(ctx, detector, podName, load)
			}

			// Build candidates.
			candidates := make([]backendmetrics.PodMetrics, 0, len(tc.candidatePods))
			for _, name := range tc.candidatePods {
				candidates = append(candidates, newFakePodMetric(name))
			}

			got := detector.IsSaturated(ctx, candidates)
			require.Equal(t, tc.wantSaturation, got, "IsSaturated result mismatch")
		})
	}
}

// TestDetector_Lifecycle verifies the full state transition cycle:
// New -> PreRequest (Inc) -> ResponseComplete (Dec) -> DeletePod (Reset).
func TestDetector_Lifecycle(t *testing.T) {
	t.Parallel()

	// MaxConcurrency 1 makes state changes immediate.
	detector := NewDetector(Config{MaxConcurrency: 1})
	ctx := context.Background()
	podName := "lifecycle-pod"
	candidates := []backendmetrics.PodMetrics{newFakePodMetric(podName)}

	// 1. Initially Empty
	require.False(t, detector.IsSaturated(ctx, candidates), "expected initially empty")

	// 2. Increment (Saturated)
	detector.PreRequest(ctx, nil, makeSchedulingResult(podName))
	require.True(t, detector.IsSaturated(ctx, candidates), "expected saturated after 1 request")

	// 3. Decrement (Available)
	targetPod := newStubSchedulingPod(podName)
	detector.ResponseComplete(ctx, nil, nil, targetPod.pod)
	require.False(t, detector.IsSaturated(ctx, candidates), "expected available after completion")

	// 4. Increment again -> Delete -> Verify Reset
	detector.PreRequest(ctx, nil, makeSchedulingResult(podName))
	require.True(t, detector.IsSaturated(ctx, candidates), "re-saturation failed")

	detector.DeletePod(fullPodName(podName))

	// After deletion, the pod is "unknown" to the tracker, effectively count=0.
	require.False(t, detector.IsSaturated(ctx, candidates), "expected clean state after DeletePod")
}

// TestDetector_ConcurrencyStress performs a targeted race condition check.
// It verifies that atomic counters remain accurate under heavy contention.
func TestDetector_ConcurrencyStress(t *testing.T) {
	t.Parallel()

	// Config doesn't matter much here as we check internal state, but keep it consistent.
	detector := NewDetector(Config{MaxConcurrency: 10000})
	ctx := context.Background()
	podName := "stress-pod"
	fullID := fullPodName(podName)

	// 1. Pre-warm the tracker.
	// We must ensure the atomic counter exists in the map before starting the race.
	// Otherwise, early 'dec' calls might be ignored (safety feature) if they beat the first 'inc' calls, causing a
	// positive drift.
	warmUpRes := makeSchedulingResult(podName)
	warmUpPod := newStubSchedulingPod(podName)
	detector.PreRequest(ctx, nil, warmUpRes)                // Creates entry, count=1
	detector.ResponseComplete(ctx, nil, nil, warmUpPod.pod) // Decrements, count=0

	const (
		numGoroutines = 50
		opsPerRoutine = 1000
	)

	var wg sync.WaitGroup
	wg.Add(numGoroutines * 2)

	// Launch increments.
	for range numGoroutines {
		go func() {
			defer wg.Done()
			res := makeSchedulingResult(podName)
			for range opsPerRoutine {
				detector.PreRequest(ctx, nil, res)
			}
		}()
	}

	// Launch decrements.
	for range numGoroutines {
		go func() {
			defer wg.Done()
			targetPod := newStubSchedulingPod(podName)
			for range opsPerRoutine {
				detector.ResponseComplete(ctx, nil, nil, targetPod.pod)
			}
		}()
	}

	wg.Wait()

	// Strict white-box check: Counter MUST be exactly 0.
	finalCount := detector.tracker.get(fullID)
	require.Equal(t, int64(0), finalCount, "atomic counter drift detected; expected 0")
}

// --- Test Helpers & Mocks ---

func driveLoad(ctx context.Context, detector *Detector, podName string, count int) {
	res := makeSchedulingResult(podName)
	for i := 0; i < count; i++ {
		detector.PreRequest(ctx, nil, res)
	}
}

func fullPodName(name string) string {
	return types.NamespacedName{Name: name, Namespace: "default"}.String()
}

// makeSchedulingResult creates a minimal result for PreRequest
func makeSchedulingResult(podName string) *schedulingtypes.SchedulingResult {
	return &schedulingtypes.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*schedulingtypes.ProfileRunResult{
			"default": {
				TargetPods: []schedulingtypes.Pod{newStubSchedulingPod(podName)},
			},
		},
	}
}

func newFakePodMetric(name string) *backendmetrics.FakePodMetrics {
	return &backendmetrics.FakePodMetrics{
		Pod: &backend.Pod{NamespacedName: types.NamespacedName{Name: name, Namespace: "default"}},
	}
}

// stubSchedulingPod mocks schedulingtypes.Pod for Filter.
// It embeds the interface to satisfy the compiler but only implements GetPod.
type stubSchedulingPod struct {
	schedulingtypes.Pod
	pod *backend.Pod
}

func newStubSchedulingPod(name string) *stubSchedulingPod {
	return &stubSchedulingPod{
		pod: &backend.Pod{NamespacedName: types.NamespacedName{Name: name, Namespace: "default"}},
	}
}

func (f *stubSchedulingPod) GetPod() *backend.Pod { return f.pod }
