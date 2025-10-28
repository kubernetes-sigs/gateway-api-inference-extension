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
			require.NotEmpty(t, p.Status.PodIP, "primary pod %s has no IP yet", p.Name)
			primaryPodNames = append(primaryPodNames, p.Name)
			primaryPodIPs = append(primaryPodIPs, p.Status.PodIP)
		}

		secondaryPodNames := make([]string, 0, len(secondaryPods))
		secondaryPodIPs := make([]string, 0, len(secondaryPods))
		for _, p := range secondaryPods {
			require.NotEmpty(t, p.Status.PodIP, "secondary pod %s has no IP yet", p.Name)
			secondaryPodNames = append(secondaryPodNames, p.Name)
			secondaryPodIPs = append(secondaryPodIPs, p.Status.PodIP)
		}

		// Send one targeted request per backend Pod to ensure EPP readiness.
		allIPs := append(append([]string{}, primaryPodIPs...), secondaryPodIPs...)
		allNames := append(append([]string{}, primaryPodNames...), secondaryPodNames...)
		for i := 0; i < len(allIPs); i++ {
			gwhttp.MakeRequestAndExpectEventuallyConsistentResponse(
				t,
				s.RoundTripper,
				s.TimeoutConfig,
				gwAddr,
				gwhttp.ExpectedResponse{
					Request: gwhttp.Request{
						Host:   hostname,
						Path:   path,
						Method: http.MethodPost,
						Body:   `{"model":"conformance-fake-model","prompt":"Warmup"}`,
						Headers: map[string]string{
							test.HeaderTestEppEndPointSelectionKey: allIPs[i],
						},
					},
					Response: gwhttp.Response{
						StatusCodes: []int{http.StatusOK},
					},
					Backend:   allNames[i],
					Namespace: resources.AppBackendNamespace,
				},
			)
		}

		// Provide a union list of eligible endpoints for the test. Each pool's EPP
		// should filter to endpoints that actually belong to its pool.
		eppHeaderValue := strings.Join(allIPs, ",")

		requestBody := `{
			"model": "conformance-fake-model",
			"prompt": "Write as if you were a critic: San Francisco"
		}`

		// Build quick lookup sets for attributing each hit to a pool by backend pod name.
		primarySet := make(map[string]struct{}, len(primaryPodNames))
		for _, n := range primaryPodNames {
			primarySet[n] = struct{}{}
		}
		secondarySet := make(map[string]struct{}, len(secondaryPodNames))
		for _, n := range secondaryPodNames {
			secondarySet[n] = struct{}{}
		}

		headers := map[string]string{
			test.HeaderTestEppEndPointSelectionKey: eppHeaderValue,
		}
		expected := gwhttp.ExpectedResponse{
			Request: gwhttp.Request{
				Host:    hostname,
				Path:    path,
				Method:  http.MethodPost,
				Headers: headers,
				Body:    requestBody,
			},
			Response: gwhttp.Response{
				StatusCode: http.StatusOK,
			},
			Namespace: resources.AppBackendNamespace,
		}
		req := gwhttp.MakeRequest(t, &expected, gwAddr, "HTTP", "http")

		var primaryHits, secondaryHits atomic.Int64
		var g errgroup.Group
		g.SetLimit(concurrentRequests)

		for i := 0; i < totalRequests; i++ {
			g.Go(func() error {
				cReq, cRes, err := s.RoundTripper.CaptureRoundTrip(req)
				if err != nil {
					return fmt.Errorf("failed to roundtrip request: %w", err)
				}
				if err := gwhttp.CompareRoundTrip(t, &req, cReq, cRes, expected); err != nil {
					return fmt.Errorf("response expectation failed: %w", err)
				}

				// Attribute response to pool by backend pod name.
				if _, ok := primarySet[cReq.Pod]; ok {
					primaryHits.Add(1)
				} else if _, ok := secondarySet[cReq.Pod]; ok {
					secondaryHits.Add(1)
				} else {
					return fmt.Errorf("request was handled by unexpected pod %q (not in either pool)", cReq.Pod)
				}
				return nil
			})
		}
		require.NoError(t, g.Wait(), "requests failed")

		ph := float64(primaryHits.Load())
		sh := float64(secondaryHits.Load())
		total := ph + sh
		require.Equal(t, int64(totalRequests), int64(total), "sum of hits must equal number of attempts")
		require.Greater(t, total, 0.0)

		observedPrimary := ph / total
		expectedPrimary := float64(primaryWeight) / float64(primaryWeight+secondaryWeight)

		// Allow either a 10 percentage-point absolute error, or a 3-sigma binomial CI.
		sigma := math.Sqrt(expectedPrimary * (1.0 - expectedPrimary) / total)
		absTolerance := math.Max(0.10, 3.0*sigma)

		diff := math.Abs(observedPrimary - expectedPrimary)
		require.LessOrEqualf(t, diff, absTolerance,
			"weighted split out of bounds: observed primary=%.3f (hits=%d/%d), expected=%.3f, tolerance=±%.3f",
			observedPrimary, int64(ph), int64(total), expectedPrimary, absTolerance)
		t.Logf("Weighted split OK: primary=%.3f (hits=%d/%d), expected=%.3f, tolerance=±%.3f; secondary hits=%d",
			observedPrimary, int64(ph), int64(total), expectedPrimary, absTolerance, int64(sh))
	},
}
