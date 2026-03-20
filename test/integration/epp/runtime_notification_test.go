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
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/extractor/mocks"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/source/notifications"
)

var (
	// errTest is a test error used to verify error handling in extractors.
	errTest = errors.New("test error")
)

// setupRuntimeWithExtractor creates a runtime configured with a notification source and extractor.
// This helper reduces code duplication across test cases.
func setupRuntimeWithExtractor(r *datalayer.Runtime, extractorName string) (*mocks.NotificationExtractor, error) {
	gvk := schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"}
	src := notifications.NewK8sNotificationSource(notifications.NotificationSourceType, "pod-watcher", gvk)
	ext := mocks.NewNotificationExtractor(extractorName)
	cfg := &datalayer.Config{
		Sources: []datalayer.DataSourceConfig{
			{Plugin: src, Extractors: []fwkdl.Extractor{ext}},
		},
	}
	return ext, r.Configure(cfg, false, "", logger)
}

func TestRuntimeNotificationDispatch(t *testing.T) {
	tests := []struct {
		name    string
		setup   func(*datalayer.Runtime) (*mocks.NotificationExtractor, error)
		trigger func(*testSetup) error
		verify  func(*mocks.NotificationExtractor, int) bool
	}{
		{
			name: "runtime dispatches add event to extractor",
			setup: func(r *datalayer.Runtime) (*mocks.NotificationExtractor, error) {
				return setupRuntimeWithExtractor(r, "test-extractor-add")
			},
			trigger: func(s *testSetup) error {
				pod := newTestPod("test-pod", s.namespace)
				return s.mgrClient.Create(s.ctx, pod)
			},
			verify: func(ext *mocks.NotificationExtractor, initialCount int) bool {
				events := ext.GetEvents()
				if len(events) <= initialCount {
					return false
				}
				for i := initialCount; i < len(events); i++ {
					if events[i].Type == fwkdl.EventAddOrUpdate {
						return true
					}
				}
				return false
			},
		},
		{
			name: "runtime dispatches update event to extractor",
			setup: func(r *datalayer.Runtime) (*mocks.NotificationExtractor, error) {
				return setupRuntimeWithExtractor(r, "test-extractor-update")
			},
			trigger: func(s *testSetup) error {
				pod := newTestPod("test-pod-update", s.namespace)
				if err := s.mgrClient.Create(s.ctx, pod); err != nil {
					return err
				}
				return updatePod(s.ctx, s.mgrClient, pod)
			},
			verify: func(ext *mocks.NotificationExtractor, initialCount int) bool {
				events := ext.GetEvents()
				if len(events) <= initialCount {
					return false
				}
				for i := initialCount; i < len(events); i++ {
					if events[i].Type == fwkdl.EventAddOrUpdate {
						return true
					}
				}
				return false
			},
		},
		{
			name: "runtime dispatches delete event to extractor",
			setup: func(r *datalayer.Runtime) (*mocks.NotificationExtractor, error) {
				return setupRuntimeWithExtractor(r, "test-extractor-delete")
			},
			trigger: func(s *testSetup) error {
				pod := newTestPod("test-pod-delete", s.namespace)
				if err := s.mgrClient.Create(s.ctx, pod); err != nil {
					return err
				}
				// Wait a bit to ensure the pod is synced before deletion
				time.Sleep(100 * time.Millisecond)
				return s.mgrClient.Delete(s.ctx, pod)
			},
			verify: func(ext *mocks.NotificationExtractor, initialCount int) bool {
				events := ext.GetEvents()
				if len(events) <= initialCount {
					return false
				}
				for i := initialCount; i < len(events); i++ {
					if events[i].Type == fwkdl.EventDelete {
						return true
					}
				}
				return false
			},
		},
		{
			name: "multiple extractors receive events",
			setup: func(r *datalayer.Runtime) (*mocks.NotificationExtractor, error) {
				gvk := schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"}
				src := notifications.NewK8sNotificationSource(notifications.NotificationSourceType, "pod-watcher", gvk)
				ext1 := mocks.NewNotificationExtractor("extractor-1")
				ext2 := mocks.NewNotificationExtractor("extractor-2")
				cfg := &datalayer.Config{
					Sources: []datalayer.DataSourceConfig{
						{Plugin: src, Extractors: []fwkdl.Extractor{ext1, ext2}},
					},
				}
				return ext1, r.Configure(cfg, false, "", logger)
			},
			trigger: func(s *testSetup) error {
				pod := newTestPod("test-pod-multi", s.namespace)
				return s.mgrClient.Create(s.ctx, pod)
			},
			verify: func(ext *mocks.NotificationExtractor, initialCount int) bool {
				events := ext.GetEvents()
				return len(events) > initialCount
			},
		},
		{
			name: "extractor error doesn't stop other extractors",
			setup: func(r *datalayer.Runtime) (*mocks.NotificationExtractor, error) {
				gvk := schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"}
				src := notifications.NewK8sNotificationSource(notifications.NotificationSourceType, "pod-watcher", gvk)
				errExtractor := mocks.NewNotificationExtractor("error-extractor").WithExtractError(errTest)
				workingExtractor := mocks.NewNotificationExtractor("working-extractor")
				cfg := &datalayer.Config{
					Sources: []datalayer.DataSourceConfig{
						{Plugin: src, Extractors: []fwkdl.Extractor{errExtractor, workingExtractor}},
					},
				}
				return workingExtractor, r.Configure(cfg, false, "", logger)
			},
			trigger: func(s *testSetup) error {
				pod := newTestPod("test-pod-error", s.namespace)
				return s.mgrClient.Create(s.ctx, pod)
			},
			verify: func(ext *mocks.NotificationExtractor, initialCount int) bool {
				events := ext.GetEvents()
				return len(events) > initialCount
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			setup := setupIntegrationTest(t, false)

			r := datalayer.NewRuntime(time.Second)
			extractor, err := tc.setup(r)
			require.NoError(t, err)

			err = r.Start(setup.ctx, setup.mgr)
			require.NoError(t, err)

			initialCount := len(extractor.GetEvents())

			err = tc.trigger(setup)
			require.NoError(t, err)

			require.Eventually(t, func() bool {
				return tc.verify(extractor, initialCount)
			}, eventWaitTimeout, eventPollInterval,
				"Timeout waiting for extractor to receive event")
		})
	}
}

func TestRuntimeNotificationWithRuntime(t *testing.T) {
	setup := setupIntegrationTest(t, false)
	pod := newTestPod("runtime-test-pod", setup.namespace)

	r := datalayer.NewRuntime(time.Second)

	gvk := schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"}
	src := notifications.NewK8sNotificationSource(notifications.NotificationSourceType, "pod-watcher", gvk)
	extractor := mocks.NewNotificationExtractor("pod-extractor")

	cfg := &datalayer.Config{
		Sources: []datalayer.DataSourceConfig{
			{Plugin: src, Extractors: []fwkdl.Extractor{extractor}},
		},
	}
	require.NoError(t, r.Configure(cfg, false, "", logger))

	require.NoError(t, r.Start(setup.ctx, setup.mgr))

	initialCount := len(extractor.GetEvents())
	require.NoError(t, setup.mgrClient.Create(setup.ctx, pod))

	assertEventAndReconcile(t, extractor, setup.reconciler, fwkdl.EventAddOrUpdate, pod.Name, initialCount, 0)
}

func TestRuntimeNotificationDifferentGVKs(t *testing.T) {
	setup := setupIntegrationTest(t, false)
	r := datalayer.NewRuntime(time.Second)

	podGvk := schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"}
	podSrc := notifications.NewK8sNotificationSource(notifications.NotificationSourceType, "pod-watcher", podGvk)
	podExtractor := mocks.NewNotificationExtractor("pod-extractor")

	svcGvk := schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Service"}
	svcSrc := notifications.NewK8sNotificationSource(notifications.NotificationSourceType, "svc-watcher", svcGvk)
	svcExtractor := mocks.NewNotificationExtractor("svc-extractor").WithGVK(svcGvk)

	cfg := &datalayer.Config{
		Sources: []datalayer.DataSourceConfig{
			{Plugin: podSrc, Extractors: []fwkdl.Extractor{podExtractor}},
			{Plugin: svcSrc, Extractors: []fwkdl.Extractor{svcExtractor}},
		},
	}

	require.NoError(t, r.Configure(cfg, false, "", logger))
	require.NoError(t, r.Start(setup.ctx, setup.mgr))

	pod := newTestPod("test-pod-gvk", setup.namespace)
	require.NoError(t, setup.mgrClient.Create(setup.ctx, pod))

	require.Eventually(t, func() bool {
		return len(podExtractor.GetEvents()) > 0
	}, eventWaitTimeout, eventPollInterval)

	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-svc",
			Namespace: setup.namespace,
		},
		Spec: corev1.ServiceSpec{
			Ports: []corev1.ServicePort{
				{
					Port: 80,
				},
			},
		},
	}
	require.NoError(t, setup.mgrClient.Create(setup.ctx, svc))

	require.Eventually(t, func() bool {
		return len(svcExtractor.GetEvents()) > 0
	}, eventWaitTimeout, eventPollInterval)
}
