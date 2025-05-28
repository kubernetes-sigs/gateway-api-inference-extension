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
	"testing"

	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api-inference-extension/conformance/tests"
	k8sutils "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/kubernetes"
	trafficutils "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/traffic"
	"sigs.k8s.io/gateway-api/conformance/utils/suite"
)

func init() {
	tests.ConformanceTests = append(tests.ConformanceTests, HTTPRouteMultipleRulesDifferentPools)
}

// HTTPRouteMultipleRulesDifferentPools defines the test case for validating
// that an HTTPRoute can successfully route to multiple distinct InferencePools
// based on different rules, and that traffic is routed accordingly.
var HTTPRouteMultipleRulesDifferentPools = suite.ConformanceTest{
	ShortName:   "HTTPRouteMultipleRulesDifferentPools",
	Description: "Validates that a single HTTPRoute can route to multiple different InferencePools based on distinct rules, and verifies traffic flow.",
	Manifests:   []string{"tests/basic/inferencepool_multiple_rules_different_pools.yaml"},
	Test: func(t *testing.T, s *suite.ConformanceTestSuite) {
		const (
			appBackendNamespace            = "gateway-conformance-app-backend"
			infraNamespace                 = "gateway-conformance-infra"
			httpRouteName                  = "httproute-multi-pool-rules"
			poolPrimaryName                = "pool-primary"
			poolSecondaryName              = "pool-secondary"
			gatewayName                    = "conformance-gateway"
			backendPrimaryDeploymentName   = "multi-pool-backend-primary-deployment"
			backendSecondaryDeploymentName = "multi-pool-backend-secondary-deployment"
			requestHostname                = "multi-rules.example.com"
		)

		routeNN := types.NamespacedName{Name: httpRouteName, Namespace: appBackendNamespace}
		poolPrimaryNN := types.NamespacedName{Name: poolPrimaryName, Namespace: appBackendNamespace}
		poolSecondaryNN := types.NamespacedName{Name: poolSecondaryName, Namespace: appBackendNamespace}
		gatewayNN := types.NamespacedName{Name: gatewayName, Namespace: infraNamespace}

		t.Run("HTTPRoute should be Accepted, ResolvedRefs and Primary InferencePool should be RouteAccepted", func(t *testing.T) {
			k8sutils.HTTPRouteAndInferencePoolMustBeAcceptedAndRouteAccepted(t, s.Client, routeNN, gatewayNN, poolPrimaryNN)
		})

		t.Run("Secondary InferencePool should be RouteAccepted", func(t *testing.T) {
			k8sutils.InferencePoolMustBeRouteAccepted(t, s.Client, poolSecondaryNN)
		})

		t.Run("Verify traffic routing to both pools", func(t *testing.T) {
			timeoutConfig := s.TimeoutConfig

			gwAddr := k8sutils.GetGatewayEndpoint(t, s.Client, timeoutConfig, gatewayNN)
			t.Logf("Gateway endpoint for %s/%s is: %s", gatewayNN.Namespace, gatewayNN.Name, gwAddr)

			t.Logf("Making request to /app-primary, expecting backend %s", backendPrimaryDeploymentName)
			trafficutils.MakeRequestAndExpectSuccess(
				t,
				s.RoundTripper,
				timeoutConfig,
				gwAddr,
				requestHostname,
				"/app-primary",
				backendPrimaryDeploymentName,
				appBackendNamespace,
			)
			t.Logf("Successfully routed traffic to /app-primary via %s to a pod in %s", gwAddr, backendPrimaryDeploymentName)

			t.Logf("Making request to /app-secondary, expecting backend %s", backendSecondaryDeploymentName)
			trafficutils.MakeRequestAndExpectSuccess(
				t,
				s.RoundTripper,
				timeoutConfig,
				gwAddr,
				requestHostname,
				"/app-secondary",
				backendSecondaryDeploymentName,
				appBackendNamespace,
			)
			t.Logf("Successfully routed traffic to /app-secondary via %s to a pod in %s", gwAddr, backendSecondaryDeploymentName)
		})
	},
}
