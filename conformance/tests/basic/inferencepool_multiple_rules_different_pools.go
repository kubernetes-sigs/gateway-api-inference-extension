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

package basic

import (
	"context"
	"fmt"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/gateway-api/conformance/utils/http"
	"sigs.k8s.io/gateway-api/conformance/utils/suite"
	"sigs.k8s.io/gateway-api/pkg/features"

	"sigs.k8s.io/gateway-api-inference-extension/conformance/tests"
	k8sutils "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/kubernetes"
)

func init() {
	tests.ConformanceTests = append(tests.ConformanceTests, HTTPRouteMultipleRulesDifferentPools)
}

var HTTPRouteMultipleRulesDifferentPools = suite.ConformanceTest{
	ShortName:   "HTTPRouteMultipleRulesDifferentPools",
	Description: "An HTTPRoute with two rules routing to two different InferencePools",
	Manifests:   []string{"tests/basic/inferencepool_multiple_rules_different_pools.yaml"},
	Features: []features.FeatureName{
		features.SupportGateway,
		features.SupportHTTPRoute,
		features.FeatureName("SupportInferencePool"),
	},
	Test: func(t *testing.T, s *suite.ConformanceTestSuite) {
		const (
			appBackendNamespace = "gateway-conformance-app-backend"
			infraNamespace      = "gateway-conformance-infra"

			poolPrimaryName   = "pool-primary"
			poolSecondaryName = "pool-secondary"
			routeName         = "httproute-multiple-rules-different-pools"
			gatewayName       = "conformance-gateway"

			backendPrimaryLabelValue   = "inference-model-1"
			backendSecondaryLabelValue = "inference-model-2"
			backendAppLabelKey         = "app"
			backendPort                = 3000

			primaryPath        = "/primary"
			secondaryPath      = "/secondary"
			eppSelectionHeader = "test-epp-endpoint-selection"
		)

		primaryPoolNN := types.NamespacedName{Name: poolPrimaryName, Namespace: appBackendNamespace}
		secondaryPoolNN := types.NamespacedName{Name: poolSecondaryName, Namespace: appBackendNamespace}
		routeNN := types.NamespacedName{Name: routeName, Namespace: appBackendNamespace}
		gatewayNN := types.NamespacedName{Name: gatewayName, Namespace: infraNamespace}

		t.Run("Wait for resources to be accepted", func(t *testing.T) {
			k8sutils.HTTPRouteAndInferencePoolMustBeAcceptedAndRouteAccepted(t, s.Client, routeNN, gatewayNN, primaryPoolNN)
			k8sutils.HTTPRouteAndInferencePoolMustBeAcceptedAndRouteAccepted(t, s.Client, routeNN, gatewayNN, secondaryPoolNN)
		})

		t.Run("Traffic should be routed to the correct pool based on path", func(t *testing.T) {
			primarySelector := labels.SelectorFromSet(labels.Set{backendAppLabelKey: backendPrimaryLabelValue})
			secondarySelector := labels.SelectorFromSet(labels.Set{backendAppLabelKey: backendSecondaryLabelValue})

			getPod := func(selector labels.Selector) *corev1.Pod {
				var pods corev1.PodList
				ctx, cancel := context.WithTimeout(context.Background(), s.TimeoutConfig.RequestTimeout)
				defer cancel()

				wait.PollUntilContextTimeout(ctx, 1*time.Second, s.TimeoutConfig.RequestTimeout, true, func(ctx context.Context) (bool, error) {
					if err := s.Client.List(ctx, &pods, &client.ListOptions{
						LabelSelector: selector,
						Namespace:     appBackendNamespace,
					}); err != nil {
						return false, fmt.Errorf("failed to list pods: %w", err)
					}

					if len(pods.Items) > 0 && pods.Items[0].Status.PodIP != "" {
						return true, nil
					}
					return false, nil
				})

				if len(pods.Items) == 0 {
					t.Fatalf("timed out waiting for pods with selector %s", selector.String())
				}
				return &pods.Items[0]
			}

			primaryPod := getPod(primarySelector)
			secondaryPod := getPod(secondarySelector)

			gwAddr := k8sutils.GetGatewayEndpoint(t, s.Client, s.TimeoutConfig, gatewayNN)

			testCases := []struct {
				name          string
				path          string
				targetPodIP   string
				targetPodName string
			}{
				{
					name:          "request to primary pool",
					path:          primaryPath,
					targetPodIP:   primaryPod.Status.PodIP,
					targetPodName: primaryPod.Name,
				},
				{
					name:          "request to secondary pool",
					path:          secondaryPath,
					targetPodIP:   secondaryPod.Status.PodIP,
					targetPodName: secondaryPod.Name,
				},
			}

			for _, tc := range testCases {
				t.Run(tc.name, func(t *testing.T) {
					expectedResponse := http.ExpectedResponse{
						Request: http.Request{
							Path: tc.path,
							Headers: map[string]string{
								eppSelectionHeader: fmt.Sprintf("%s:%d", tc.targetPodIP, backendPort),
							},
						},
						Backend:   tc.targetPodName,
						Namespace: appBackendNamespace,
					}
					http.MakeRequestAndExpectEventuallyConsistentResponse(t, s.RoundTripper, s.TimeoutConfig, gwAddr, expectedResponse)
				})
			}
		})
	},
}
