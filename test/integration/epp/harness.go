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

package epp

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap/zapcore"
	"google.golang.org/grpc"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	metricsutils "k8s.io/component-base/metrics/testutil"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	crmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"

	eppRunner "sigs.k8s.io/gateway-api-inference-extension/cmd/epp/runner"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	eppServer "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/server"
	epptestutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/testing"
	"sigs.k8s.io/gateway-api-inference-extension/test/integration"
)

// Global State (Initialized in TestMain)
var (
	k8sClient     client.Client
	testEnv       *envtest.Environment
	testScheme    = runtime.NewScheme()
	logger        = zap.New(zap.UseDevMode(true), zap.Level(-1*zapcore.Level(logutil.DEFAULT)))
	baseResources []*unstructured.Unstructured
)

const (
	testPoolName = "vllm-llama3-8b-instruct-pool"
	testConfig   = `
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: queue-scorer
  - type: kv-cache-utilization-scorer
  - type: prefix-cache-scorer
  - type: lora-affinity-scorer
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: queue-scorer
      - pluginRef: kv-cache-utilization-scorer
      - pluginRef: prefix-cache-scorer
      - pluginRef: lora-affinity-scorer
`
)

// HarnessConfig holds configuration options for the TestHarness.
type HarnessConfig struct {
	// StandaloneMode indicates if the EPP should run without watching Gateway API CRDs.
	StandaloneMode bool
}

// HarnessOption is a functional option for configuring the TestHarness.
type HarnessOption func(*HarnessConfig)

// WithStandaloneMode configures the harness to run in Standalone mode.
// In this mode, CRD watchers are disabled and a static EndpointPool is injected.
func WithStandaloneMode() HarnessOption {
	return func(c *HarnessConfig) {
		c.StandaloneMode = true
	}
}

// TestHarness encapsulates the environment for a single isolated EPP test run.
// It manages the lifecycle of the controller manager, the EPP server, and the K8s namespace.
type TestHarness struct {
	t         *testing.T
	ctx       context.Context
	Namespace string

	// --- Config State ---
	StandaloneMode bool

	Client    extProcPb.ExternalProcessor_ProcessClient
	Datastore datastore.Datastore

	// Internal handles for cleanup
	grpcConn *grpc.ClientConn

	fakePmc *backendmetrics.FakePodMetricsClient
}

// NewTestHarness boots up a fully isolated test environment.
// It creates a unique Namespace, scopes the Manager to that Namespace, and starts the components.
// Note: EPP tests must run serially because they rely on the global Prometheus registry.
func NewTestHarness(t *testing.T, ctx context.Context, opts ...HarnessOption) *TestHarness {
	t.Helper()

	config := &HarnessConfig{}
	for _, opt := range opts {
		opt(config)
	}

	// Create dedicated namespace for the whole test
	uid := uuid.New().String()[:8]
	testNamespaceName := "epp-test-" + uid
	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: testNamespaceName}}
	require.NoError(t, k8sClient.Create(ctx, ns), "failed to create test namespace")

	eppOptions := defaultEppServerOptions(t, testNamespaceName)
	if config.StandaloneMode {
		// Only standalone EPP need to set the EndpointSelector.
		eppOptions.EndpointSelector = "app=" + testPoolName
	}

	fakePmc := &backendmetrics.FakePodMetricsClient{}
	mgr, dataStore, err := eppRunner.NewTestRunnerSetup(ctx, testEnv.Config, eppOptions, fakePmc)
	require.NoError(t, err, "failed to create manager")
	mgrCtx, mgrCancel := context.WithCancel(ctx)

	// Start Manager.
	go func() {
		if err := mgr.Start(mgrCtx); err != nil {
			// Context cancellation is expected during teardown.
			if !strings.Contains(err.Error(), "context canceled") {
				logger.Error(err, "manager stopped unexpectedly")
			}
		}
	}()

	client, conn := integration.ExtProcServerClient(
		t,
		mgrCtx,
		eppOptions.GRPCPort,
		logger,
	)

	h := &TestHarness{
		t:              t,
		ctx:            mgrCtx,
		Namespace:      eppOptions.PoolNamespace,
		StandaloneMode: config.StandaloneMode,
		Client:         client,
		Datastore:      dataStore,
		grpcConn:       conn,
		fakePmc:        fakePmc,
	}

	t.Cleanup(func() {
		mgrCancel()
		_ = h.grpcConn.Close()
		// Deleting the Namespace cascades to all contained resources.
		_ = k8sClient.Delete(context.Background(), &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: eppOptions.PoolNamespace}})
		// Crucial: Reset global metrics registry to prevent pollution between serial tests.
		metrics.Reset()
	})

	return h
}

func defaultEppServerOptions(t *testing.T, namespace string) *eppServer.Options {
	t.Helper()

	eppOptions := eppServer.NewOptions()
	eppOptions.PoolName = testPoolName
	eppOptions.PoolNamespace = namespace
	eppOptions.ConfigText = testConfig

	metricsPort, err := integration.GetFreePort()
	require.NoError(t, err)
	eppOptions.MetricsPort = metricsPort

	grpcPort, err := integration.GetFreePort()
	require.NoError(t, err)
	eppOptions.GRPCPort = grpcPort

	healthPort, err := integration.GetFreePort()
	require.NoError(t, err)
	eppOptions.GRPCHealthPort = healthPort
	eppOptions.EndpointTargetPorts = []int{8000}
	eppOptions.SecureServing = false
	return eppOptions
}

// --- Fluent Builder API ---

// WithBaseResources injects the standard pool and objective definitions into the test namespace.
// These resources are pre-parsed in TestMain to avoid I/O overhead in the loop.
func (h *TestHarness) WithBaseResources() *TestHarness {
	h.t.Helper()
	for _, obj := range baseResources {
		copy := obj.DeepCopy()
		copy.SetNamespace(h.Namespace)
		require.NoError(h.t, k8sClient.Create(h.ctx, copy), "failed to create base resource: %s", obj.GetKind())
	}
	return h
}

// WithPods creates pod objects in the API server and configures the fake metrics client.
func (h *TestHarness) WithPods(pods []podState) *TestHarness {
	h.t.Helper()
	metricsMap := make(map[types.NamespacedName]*fwkdl.Metrics)

	// Pre-calculate metrics and register them with the fake client.
	for _, p := range pods {
		metricsKeyName := fmt.Sprintf("pod-%d-rank-0", p.index)
		activeModelsMap := make(map[string]int)
		for _, m := range p.activeModels {
			activeModelsMap[m] = 1
		}

		metricsMap[types.NamespacedName{Namespace: h.Namespace, Name: metricsKeyName}] = &fwkdl.Metrics{
			WaitingQueueSize:    p.queueSize,
			KVCacheUsagePercent: p.kvCacheUsage,
			ActiveModels:        activeModelsMap,
			WaitingModels:       make(map[string]int),
		}
	}
	h.fakePmc.SetRes(metricsMap)

	// Create K8s Objects.
	for _, p := range pods {
		name := fmt.Sprintf("pod-%d", p.index)

		// Create K8s object.
		pod := epptestutil.MakePod(name).
			Namespace(h.Namespace).
			ReadyCondition(). // Sets Status.Conditions.
			Labels(map[string]string{"app": testPoolName}).
			IP(fmt.Sprintf("192.168.1.%d", p.index+1)).
			Complete().
			ObjRef()

		// Snapshot the status (Create wipes it).
		intendedStatus := pod.Status

		// Create the resource.
		require.NoError(h.t, k8sClient.Create(h.ctx, pod), "failed to create pod %s", name)

		// Restore Status on the created K8s object which now has the correct ResourceVersion/UID.
		pod.Status = intendedStatus

		// Update Status subresource.
		require.NoError(h.t, k8sClient.Status().Update(h.ctx, pod), "failed to update status for pod %s", name)
	}

	return h
}

// WaitForReadyPodsMetric blocks until the prometheus metric 'inference_pool_ready_pods' matches the expected count.
// This ensures the background metric collector has fully synced.
func (h *TestHarness) WaitForReadyPodsMetric(expectedCount int) {
	h.t.Helper()

	expected := cleanMetric(metricReadyPods(expectedCount))
	require.Eventually(h.t, func() bool {
		err := metricsutils.GatherAndCompare(crmetrics.Registry, strings.NewReader(expected),
			"inference_pool_ready_pods")
		return err == nil
	}, 10*time.Second, 50*time.Millisecond, "Timed out waiting for inference_pool_ready_pods metric to settle")
}

// WaitForSync blocks until the EPP Datastore has synced the expected number of pods.
// In Standard mode, it also waits for the InferencePool CRD to sync.
func (h *TestHarness) WaitForSync(expectedPods int, checkModelObjective string) *TestHarness {
	h.t.Helper()
	require.Eventually(h.t, func() bool {
		// If we are NOT in standalone mode, we must wait for the Pool CRD to sync.
		// In Standalone mode, there is no CRD controller, so this check is skipped.
		if !h.StandaloneMode && !h.Datastore.PoolHasSynced() {
			return false
		}

		if len(h.Datastore.PodList(datastore.AllPodsPredicate)) != expectedPods {
			return false
		}
		// In Standalone mode, Objectives are not CRDs, so we skip checking the Objective store unless we add logic to mock
		// that too.
		// For now, we skip objective verification in Standalone.
		if !h.StandaloneMode && checkModelObjective != "" && h.Datastore.ObjectiveGet(checkModelObjective) == nil {
			return false
		}
		return true
	}, 10*time.Second, 50*time.Millisecond,
		"Datastore sync timed out.\n- Mode: Standalone=%v\n- PoolSynced: %v\n- Pods Found: %d (Expected: %d)",
		h.StandaloneMode,
		h.Datastore.PoolHasSynced(),
		len(h.Datastore.PodList(datastore.AllPodsPredicate)),
		expectedPods,
	)
	return h
}

// ExpectMetrics asserts that specific metrics match the expected Prometheus output.
// It uses Eventually to allow for slight delays in metric recording (e.g. async token counting).
func (h *TestHarness) ExpectMetrics(expected map[string]string) {
	h.t.Helper()
	for name, value := range expected {
		var err error
		assert.Eventually(h.t, func() bool {
			err = metricsutils.GatherAndCompare(crmetrics.Registry, strings.NewReader(value), name)
			return err == nil
		}, 2*time.Second, 50*time.Millisecond, "Timed out waiting for metric %s to match: %v", name)
		if err != nil {
			h.t.Errorf("Metric mismatch for %s: %v", name, err)
		}
	}
}
