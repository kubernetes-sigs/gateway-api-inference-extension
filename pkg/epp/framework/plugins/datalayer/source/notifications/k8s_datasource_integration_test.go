/*
Copyright 2026 The Kubernetes Authors.

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

package notifications

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
)

func TestIntegrationBindNotificationSource(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode (requires envtest/kubebuilder)")
	}

	testEnv := &envtest.Environment{}
	cfg, err := testEnv.Start()
	if err != nil {
		t.Skipf("Skipping test: envtest not available (%v)", err)
	}
	defer func() {
		if err := testEnv.Stop(); err != nil {
			t.Logf("Failed to stop test environment: %v", err)
		}
	}()

	mgr, err := ctrl.NewManager(cfg, ctrl.Options{
		Metrics: metricsserver.Options{
			BindAddress: "0",
		},
		HealthProbeBindAddress: "0",
	})
	require.NoError(t, err)

	k8sClient, err := client.New(cfg, client.Options{Scheme: mgr.GetScheme()})
	require.NoError(t, err)

	gvk := schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"}
	src := NewK8sNotificationSource(NotificationSourceType, "pod-watcher", gvk)
	extractor := NewMockNotificationExtractor("test-extractor")
	require.NoError(t, src.AddExtractor(extractor))

	err = datalayer.BindNotificationSource(src, mgr)
	require.NoError(t, err)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Start the manager in a goroutine
	mgrStopped := make(chan struct{})
	go func() {
		defer close(mgrStopped)
		if err := mgr.Start(ctx); err != nil {
			t.Logf("Manager failed: %v", err)
		}
	}()

	// Wait for manager to be ready
	require.Eventually(t, func() bool {
		return mgr.GetCache().WaitForCacheSync(ctx)
	}, 20*time.Second, 50*time.Millisecond, "Manager cache failed to sync")

	testPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "integration-test-pod",
			Namespace: "default",
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  "test-container",
					Image: "nginx:latest",
				},
			},
		},
	}

	t.Run("Capture Creation", func(t *testing.T) {
		initialCount := len(extractor.GetEvents())
		t.Logf("Initial event count before create: %d", initialCount)

		err = k8sClient.Create(ctx, testPod)
		require.NoError(t, err)

		// Wait for and validate the create event
		found := waitForEvent(t, extractor, initialCount, fwkdl.EventAddOrUpdate,
			testPod.Name, 15*time.Second)
		require.True(t, found, "should have received create event for test pod")
	})

	t.Run("Capture Update", func(t *testing.T) {
		initialCount := len(extractor.GetEvents())
		t.Logf("Initial event count before update: %d", initialCount)

		// Fetch latest version and modify
		current := &corev1.Pod{}
		err = k8sClient.Get(ctx, client.ObjectKey{Name: testPod.Name, Namespace: testPod.Namespace}, current)
		require.NoError(t, err)

		// Update labels (allowed for Pods)
		if current.Labels == nil {
			current.Labels = make(map[string]string)
		}
		current.Labels["test-update"] = "true"

		err = k8sClient.Update(ctx, current)
		require.NoError(t, err)

		// Wait for and validate the update event
		found := waitForEvent(t, extractor, initialCount, fwkdl.EventAddOrUpdate,
			testPod.Name, 15*time.Second)
		require.True(t, found, "should have received update event")
	})

	t.Run("Capture Deletion", func(t *testing.T) {
		initialCount := len(extractor.GetEvents())
		t.Logf("Initial event count before delete: %d", initialCount)

		err = k8sClient.Delete(ctx, testPod)
		require.NoError(t, err)

		// Wait for and validate the delete event
		found := waitForEvent(t, extractor, initialCount, fwkdl.EventDelete,
			testPod.Name, 15*time.Second)
		require.True(t, found, "should have received delete event")
	})

	cancel()
	select {
	case <-mgrStopped:
	case <-time.After(5 * time.Second):
		t.Log("Manager did not stop gracefully")
	}
}

// waitForEvent is a helper function that waits for a specific event to be received by the extractor.
// It returns true if the event is found within the timeout, false otherwise.
func waitForEvent(t *testing.T, extractor *MockNotificationExtractor, initialCount int,
	eventType fwkdl.EventType, objectName string, timeout time.Duration) bool {
	t.Helper()

	var found bool
	require.Eventually(t, func() bool {
		events := extractor.GetEvents()
		if len(events) <= initialCount {
			return false
		}

		// Find the event matching the criteria
		for i := initialCount; i < len(events); i++ {
			if events[i].Type == eventType && events[i].Object.GetName() == objectName {
				found = true
				return true
			}
		}
		return false
	}, timeout, 200*time.Millisecond, "Timeout waiting for %v event for %s", eventType, objectName)

	return found
}
