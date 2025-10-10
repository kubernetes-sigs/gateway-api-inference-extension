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

package tests

import (
	"fmt"
	"math"
	"net/http"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/require"
	"golang.org/x/sync/errgroup"
	"k8s.io/apimachinery/pkg/types"
	gwhttp "sigs.k8s.io/gateway-api/conformance/utils/http"
	"sigs.k8s.io/gateway-api/conformance/utils/suite"
	"sigs.k8s.io/gateway-api/pkg/features"

	"sigs.k8s.io/gateway-api-inference-extension/conformance/resources"
	k8sutils "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/kubernetes"
	"sigs.k8s.io/gateway-api-inference-extension/conformance/utils/traffic"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/test"
)

func init() {
	ConformanceTests = append(ConformanceTests, GatewayWeightedAcrossTwoInferencePools)
}

// GatewayWeightedAcrossTwoInferencePools verifies that Gateway splits traffic across two
// InferencePools according to backendRef weights, and that each request is routed to an
// endpoint of the selected InferencePool.
var GatewayWeightedAcrossTwoInferencePools = suite.ConformanceTest{
	ShortName:   "GatewayWeightedAcrossTwoInferencePools",
	Description: "Gateway should split traffic across two InferencePools based on backendRef weights and route only to endpoints of the selected InferencePool",
	Manifests:   []string{"tests/gateway_weighted_two_pools.yaml"},
	Features: []features.FeatureName{
		features.SupportGateway,
		features.FeatureName("SupportInferencePool"),
		features.SupportGateway,
	},
	Test: func(t *testing.T, s *suite.ConformanceTestSuite) {
		const (
			hostname = "primary.example.com"
			path     = "/weighted-two-pools-test"

			// Sample size so the weight signal dominates random noise.
			totalRequests      = 200
			concurrentRequests = 5

			// These route weights must match the test manifest.
			primaryWeight   = 70
			secondaryWeight = 30
		)

		// Objects under test.
		httpRouteNN := types.NamespacedName{Name: "httproute-weighted-two-pools", Namespace: resources.AppBackendNamespace}
		gatewayNN := resources.PrimaryGatewayNN
		primaryPoolNN := resources.PrimaryInferencePoolNN
		secondaryPoolNN := types.NamespacedName{Name: "secondary-inference-pool", Namespace: resources.AppBackendNamespace}

		// Labels for the two deployments defined in base.yaml.
		primaryLabels := map[string]string{"app": "primary-inference-model-server"}
		secondaryLabels := map[string]string{"app": "secondary-inference-model-server"}

		t.Log("Verifying HTTPRoute and both InferencePools are accepted and the Gateway has an address.")
		k8sutils.HTTPRouteMustBeAcceptedAndResolved(t, s.Client, s.TimeoutConfig, httpRouteNN, gatewayNN)
		k8sutils.InferencePoolMustBeAcceptedByParent(t, s.Client, primaryPoolNN, gatewayNN)
		k8sutils.InferencePoolMustBeAcceptedByParent(t, s.Client, secondaryPoolNN, gatewayNN)
		gwAddr := k8sutils.GetGatewayEndpoint(t, s.Client, s.TimeoutConfig, gatewayNN)

		// Discover pods for each pool and build quick lookup sets.
		t.Logf("Fetching primary backend pods with labels: %v", primaryLabels)
		primaryPods, err := k8sutils.GetPodsWithLabel(t, s.Client, resources.AppBackendNamespace, primaryLabels, s.TimeoutConfig)
		require.NoError(t, err)
		require.Len(t, primaryPods, 3) // base.yaml uses 3 replicas

		t.Logf("Fetching secondary backend pods with labels: %v", secondaryLabels)
		secondaryPods, err := k8sutils.GetPodsWithLabel(t, s.Client, resources.AppBackendNamespace, secondaryLabels, s.TimeoutConfig)
		require.NoError(t, err)
		require.Len(t, secondaryPods, 3) // base.yaml uses 3 replicas

		primaryPodNames := make([]string, 0, len(primaryPods))
		primaryPodIPs := make([]string, 0, len(primaryPods))
		for _, p := range primaryPods {
			primaryPodNames = append(primaryPodNames, p.Name)
			primaryPodIPs = append(primaryPodIPs, p.Status.PodIP)
		}
		secondaryPodNames := make([]string, 0, len(secondaryPods))
		secondaryPodIPs := make([]string, 0, len(secondaryPods))
		for _, p := range secondaryPods {
			secondaryPodNames = append(secondaryPodNames, p.Name)
			secondaryPodIPs = append(secondaryPodIPs, p.Status.PodIP)
		}

		// Provide a union list of eligible endpoints for the test. Each pool's EPP
		// should filter to endpoints that actually belong to its pool.
		allIPs := append(append([]string{}, primaryPodIPs...), secondaryPodIPs...)
		eppHeaderValue := strings.Join(allIPs, ",")

		requestBody := `{
			"model": "conformance-fake-model",
			"prompt": "Write as if you were a critic: San Francisco"
		}`

		// Send requests with the union header and verify the split roughly matches the
		// weight distribution of the test manifest.
		var (
			roundTripper  = s.RoundTripper
			g             errgroup.Group
			primaryHits   int64
			secondaryHits int64
			headers       = map[string]string{
				test.HeaderTestEppEndPointSelectionKey: eppHeaderValue,
			}
			expected = gwhttp.ExpectedResponse{
				Request: gwhttp.Request{
					Host:    hostname,
					Path:    path,
					Method:  http.MethodPost,
					Headers: headers,
				},
				Response: gwhttp.Response{
					StatusCode: http.StatusOK,
				},
				// Leave backend empty to avoid enforcing a specific pod prefix in CompareRequest.
				Namespace: resources.AppBackendNamespace,
			}
		)

		primarySet := func() map[string]struct{} {
			m := make(map[string]struct{}, len(primaryPodNames))
			for _, n := range primaryPodNames {
				m[n] = struct{}{}
			}
			return m
		}()
		secondarySet := func() map[string]struct{} {
			m := make(map[string]struct{}, len(secondaryPodNames))
			for _, n := range secondaryPodNames {
				m[n] = struct{}{}
			}
			return m
		}()

		req := gwhttp.MakeRequest(t, &expected, gwAddr, "HTTP", "http")
		g.SetLimit(concurrentRequests)
		for range totalRequests {
			g.Go(func() error {
				cReq, cRes, err := traffic.MakeCallRoundTripper(t, roundTripper, &traffic.RequestWithBody{
					Request: req,
					Body:    strings.NewReader(requestBody),
				})
				if err != nil {
					return fmt.Errorf("failed to roundtrip request: %w", err)
				}
				if err := gwhttp.CompareRequest(t, &req, cReq, cRes, expected); err != nil {
					return fmt.Errorf("response expectation failed: %w", err)
				}

				// Attribute response to pool by the backend pod name.
				if _, ok := primarySet[cReq.Pod]; ok {
					atomic.AddInt64(&primaryHits, 1)
				} else if _, ok := secondarySet[cReq.Pod]; ok {
					atomic.AddInt64(&secondaryHits, 1)
				} else {
					return fmt.Errorf("request was handled by unexpected pod %q (not in either pool)", cReq.Pod)
				}

				return nil
			})
		}
		require.NoError(t, g.Wait(), "requests failed")

		ph := float64(atomic.LoadInt64(&primaryHits))
		sh := float64(atomic.LoadInt64(&secondaryHits))
		total := ph + sh
		require.Greater(t, total, 0.0)

		observedPrimary := ph / total
		expectedPrimary := float64(primaryWeight) / float64(primaryWeight+secondaryWeight)

		// Allow either a 10 percentage-point absolute error, or a 3-sigma
		// binomial confidence interval (whichever is larger). This keeps
		// flakiness low while still detecting obvious mis-weighting.
		sigma := math.Sqrt(expectedPrimary * (1.0 - expectedPrimary) / total)
		absTolerance := math.Max(0.10, 3.0*sigma)

		diff := math.Abs(observedPrimary - expectedPrimary)
		if diff > absTolerance {
			t.Fatalf("weighted split out of bounds: observed primary=%.3f (hits=%d/%d), expected=%.3f, tolerance=±%.3f",
				observedPrimary, int64(ph), int64(total), expectedPrimary, absTolerance)
		}

		t.Logf("Weighted split OK: primary=%.3f (hits=%d/%d), expected=%.3f, tolerance=±%.3f; secondary hits=%d",
			observedPrimary, int64(ph), int64(total), expectedPrimary, absTolerance, int64(sh))

		// Sanity: ensure responses only came from pods we enumerated.
		require.True(t, slices.Contains([]int{int(ph + sh)}, int(total)))
	},
}
