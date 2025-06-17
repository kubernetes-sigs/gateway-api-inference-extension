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

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api/conformance/utils/suite"
	"sigs.k8s.io/gateway-api/pkg/features"

	"sigs.k8s.io/gateway-api-inference-extension/conformance/tests"
	k8sutils "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/kubernetes"
	trafficutils "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/traffic"
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

			primaryPath   = "/primary"
			secondaryPath = "/secondary"
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

			primaryPod := k8sutils.GetPod(t, s.Client, appBackendNamespace, primarySelector, s.TimeoutConfig.RequestTimeout)
			secondaryPod := k8sutils.GetPod(t, s.Client, appBackendNamespace, secondarySelector, s.TimeoutConfig.RequestTimeout)

			gwAddr := k8sutils.GetGatewayEndpoint(t, s.Client, s.TimeoutConfig, gatewayNN)

			t.Run("request to primary pool", func(t *testing.T) {
				trafficutils.MakeRequestAndExpectResponseFromPod(t, s.RoundTripper, s.TimeoutConfig, gwAddr, primaryPath, primaryPod)
			})

			t.Run("request to secondary pool", func(t *testing.T) {
				trafficutils.MakeRequestAndExpectResponseFromPod(t, s.RoundTripper, s.TimeoutConfig, gwAddr, secondaryPath, secondaryPod)
			})
		})
	},
}
