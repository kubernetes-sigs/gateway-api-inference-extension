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

package epp

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	crconfig "sigs.k8s.io/controller-runtime/pkg/config"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer/mocks"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/source/notifications"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
)

const (
	eventWaitTimeout   = 15 * time.Second       // max time to wait for an event
	cacheSyncTimeout   = 10 * time.Second       // max time to wait for cache sync
	cacheSyncInterval  = 50 * time.Millisecond  // polling interval for cache sync
	testContextTimeout = 30 * time.Second       // overall timeout for each test
	eventPollInterval  = 200 * time.Millisecond // polling interval for events
)

// TestIntegrationBindNotificationSource tests the end to end integration of
// k8s notification datasource with the controller-runtime registration and
// event mutation notification.
func TestIntegrationBindNotificationSource(t *testing.T) {
	setup := setupIntegrationTest(t, false)
	pod := newTestPod("integration-test-pod", setup.namespace)

	t.Run("capture creation", func(t *testing.T) {
		before := len(setup.extractor.GetEvents())
		require.NoError(t, setup.mgrClient.Create(setup.ctx, pod))
		assertEventAndReconcile(t, setup.extractor, setup.reconciler, fwkdl.EventAddOrUpdate, pod.Name, before, 0)
	})

	t.Run("capture update", func(t *testing.T) {
		before := len(setup.extractor.GetEvents())
		require.NoError(t, updatePod(setup.ctx, setup.mgrClient, pod))
		assertEventAndReconcile(t, setup.extractor, setup.reconciler, fwkdl.EventAddOrUpdate, pod.Name, before, 0)
	})

	t.Run("capture deletion", func(t *testing.T) {
		before := len(setup.extractor.GetEvents())
		require.NoError(t, setup.mgrClient.Delete(setup.ctx, pod))
		assertEventAndReconcile(t, setup.extractor, setup.reconciler, fwkdl.EventDelete, pod.Name, before, 0)
	})
}

// TestIntegrationBindNotificationSourceWithReconciler validates that
// k8s notification source and (directly registered) reconcilers can coexist.
func TestIntegrationBindNotificationSourceWithReconciler(t *testing.T) {
	setup := setupIntegrationTest(t, true)
	pod := newTestPod("test-pod-with-reconciler", setup.namespace)

	t.Run("capture creation", func(t *testing.T) {
		extractorBefore := len(setup.extractor.GetEvents())
		reconcilerBefore := setup.reconciler.getCallCount(pod.Name)
		require.NoError(t, setup.mgrClient.Create(setup.ctx, pod))
		assertEventAndReconcile(t, setup.extractor, setup.reconciler, fwkdl.EventAddOrUpdate, pod.Name, extractorBefore, reconcilerBefore)
	})

	t.Run("capture update", func(t *testing.T) {
		extractorBefore := len(setup.extractor.GetEvents())
		reconcilerBefore := setup.reconciler.getCallCount(pod.Name)
		require.NoError(t, updatePod(setup.ctx, setup.mgrClient, pod))
		assertEventAndReconcile(t, setup.extractor, setup.reconciler, fwkdl.EventAddOrUpdate, pod.Name, extractorBefore, reconcilerBefore)
	})

	t.Run("capture deletion", func(t *testing.T) {
		extractorBefore := len(setup.extractor.GetEvents())
		reconcilerBefore := setup.reconciler.getCallCount(pod.Name)
		require.NoError(t, setup.mgrClient.Delete(setup.ctx, pod))
		assertEventAndReconcile(t, setup.extractor, setup.reconciler, fwkdl.EventDelete, pod.Name, extractorBefore, reconcilerBefore)
	})
}

// testSetup holds common test resources.
type testSetup struct {
	namespace  string
	ctx        context.Context
	cancel     context.CancelFunc
	mgr        ctrl.Manager
	mgrClient  client.Client
	extractor  *mocks.NotificationExtractor
	reconciler *testPodReconciler
}

// setupIntegrationTest creates test environment with manager and notification source.
func setupIntegrationTest(t *testing.T, withReconciler bool) *testSetup {
	t.Helper()

	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	uid := uuid.New().String()[:8] // create unique namespace
	nsName := "epp-test-" + uid
	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: nsName}}
	require.NoError(t, k8sClient.Create(context.Background(), ns))

	t.Cleanup(func() {
		_ = k8sClient.Delete(context.Background(), ns)
		metrics.Reset()
	})

	mgr, mgrClient := setupTestManager(t, testEnv.Config, nsName)

	var reconciler *testPodReconciler
	if withReconciler {
		reconciler = newTestPodReconciler(mgr.GetCache())
		require.NoError(t, reconciler.SetupWithManager(mgr))
	}

	// set up notification source for Pod events
	gvk := schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"}
	src := notifications.NewK8sNotificationSource(notifications.NotificationSourceType, "pod-watcher", gvk)
	extractor := mocks.NewNotificationExtractor("pod-extractor")
	require.NoError(t, src.AddExtractor(extractor))
	require.NoError(t, datalayer.BindNotificationSource(src, mgr))

	ctx, cancel := context.WithTimeout(context.Background(), testContextTimeout)
	t.Cleanup(cancel)

	startManagerAndWaitForSync(t, mgr, ctx)

	return &testSetup{
		namespace:  nsName,
		ctx:        ctx,
		cancel:     cancel,
		mgr:        mgr,
		mgrClient:  mgrClient,
		extractor:  extractor,
		reconciler: reconciler,
	}
}

// setupTestManager creates a controller-runtime manager for testing.
func setupTestManager(t *testing.T, cfg *rest.Config, nsName string) (ctrl.Manager, client.Client) {
	t.Helper()

	skipValidation := true
	mgr, err := ctrl.NewManager(cfg, ctrl.Options{
		Scheme: testScheme,
		Cache: cache.Options{
			DefaultNamespaces: map[string]cache.Config{
				nsName: {},
			},
		},
		Controller: crconfig.Controller{
			SkipNameValidation: &skipValidation,
		},
		Metrics: metricsserver.Options{
			BindAddress: "0",
		},
		HealthProbeBindAddress: "0",
		LeaderElection:         false,
	})
	require.NoError(t, err)

	mgrClient, err := client.New(cfg, client.Options{Scheme: mgr.GetScheme()})
	require.NoError(t, err)

	return mgr, mgrClient
}

// startManagerAndWaitForSync starts the manager and waits for cache sync.
func startManagerAndWaitForSync(t *testing.T, mgr ctrl.Manager, ctx context.Context) {
	t.Helper()

	errChan := make(chan error, 1)
	go func() {
		if err := mgr.Start(ctx); err != nil {
			if ctx.Err() == nil {
				t.Errorf("Manager failed to start: %v", err)
				errChan <- err
			} else {
				logger.Info("Manager stopped due to context cancellation", "error", err)
			}
		}
		close(errChan)
	}()

	require.Eventually(t, func() bool {
		select {
		case err := <-errChan:
			if err != nil {
				return false
			}
		default:
		}
		return mgr.GetCache().WaitForCacheSync(ctx)
	}, cacheSyncTimeout, cacheSyncInterval,
		"Manager cache failed to sync within %v", cacheSyncTimeout)
}

// newTestPod creates a test pod.
func newTestPod(name, namespace string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
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
}

// updatePod updates a pod by adding a label.
func updatePod(ctx context.Context, c client.Client, pod *corev1.Pod) error {
	current := &corev1.Pod{}
	if err := c.Get(ctx, client.ObjectKey{Name: pod.Name, Namespace: pod.Namespace}, current); err != nil {
		return err
	}
	if current.Labels == nil {
		current.Labels = make(map[string]string)
	}
	current.Labels["updated"] = "true"
	return c.Update(ctx, current)
}

// assertEventAndReconcile verifies extractor received event and reconciler was called.
func assertEventAndReconcile(t *testing.T, extractor *mocks.NotificationExtractor,
	reconciler *testPodReconciler, eventType fwkdl.EventType, podName string,
	extractorCountBefore, reconcilerCountBefore int) {
	t.Helper()

	found := waitForEvent(t, extractor, extractorCountBefore, eventType, podName)
	require.True(t, found,
		"Extractor should have received %v event for pod %q. Events: %d (expected > %d)",
		eventType, podName, len(extractor.GetEvents()), extractorCountBefore)

	if reconciler != nil {
		require.Eventually(t, func() bool {
			return reconciler.getCallCount(podName) > reconcilerCountBefore
		}, eventWaitTimeout, eventPollInterval,
			"Reconciler should have been called for pod %q after %v event",
			podName, eventType)
	}
}

// waitForEvent waits for a specific event from the extractor.
func waitForEvent(t *testing.T, extractor *mocks.NotificationExtractor, initialCount int,
	eventType fwkdl.EventType, objectName string) bool {
	t.Helper()

	var found bool
	require.Eventually(t, func() bool {
		events := extractor.GetEvents()
		if len(events) <= initialCount {
			return false
		}

		for i := initialCount; i < len(events); i++ {
			if events[i].Type == eventType && events[i].Object.GetName() == objectName {
				found = true
				return true
			}
		}
		return false
	}, eventWaitTimeout, eventPollInterval,
		"Timeout waiting for %v event for %q. Events: %d (started with %d)",
		eventType, objectName, len(extractor.GetEvents()), initialCount)

	return found
}

// testPodReconciler tracks reconciliation calls for testing.
type testPodReconciler struct {
	client.Reader
	calls map[string]int
	mu    sync.Mutex
}

func newTestPodReconciler(reader client.Reader) *testPodReconciler {
	return &testPodReconciler{
		Reader: reader,
		calls:  make(map[string]int),
	}
}

func (r *testPodReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.calls[req.Name]++
	return ctrl.Result{}, nil
}

func (r *testPodReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&corev1.Pod{}).
		Complete(r)
}

// getCallCount returns reconcile call count for a pod.
func (r *testPodReconciler) getCallCount(podName string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.calls[podName]
}
