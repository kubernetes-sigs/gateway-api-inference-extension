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
	"sigs.k8s.io/gateway-api/conformance/utils/suite"
	"sigs.k8s.io/gateway-api/pkg/features"

	// Local project imports
	"sigs.k8s.io/gateway-api-inference-extension/conformance/tests"
	k8sutils "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/kubernetes"
	trafficutils "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/traffic"
)

func init() {
	tests.ConformanceTests = append(tests.ConformanceTests, InferencePoolHTTPRoutePortValidation)
}

var InferencePoolHTTPRoutePortValidation = suite.ConformanceTest{
	ShortName:   "InferencePoolHTTPRoutePortValidation",
	Description: "Validates HTTPRoute backendRef port configurations (unspecified, matching, non-matching) when referencing an InferencePool, and checks resulting status conditions.",
	Manifests:   []string{"tests/basic/inferencepool_httproute_port_validation.yaml"},
	Features: []features.FeatureName{
		features.FeatureName("SupportInferencePool"),
		features.SupportGateway,
	},
	Test: func(t *testing.T, s *suite.ConformanceTestSuite) {
		const (
			appBackendNamespace   = "gateway-conformance-app-backend"
			infraNamespace        = "gateway-conformance-infra"
			gatewayName           = "conformance-primary-gateway"
			poolName              = "target-pool-port-validation"
			backendDeploymentName = "infra-backend-deployment-port-test"
		)

		gatewayNN := types.NamespacedName{Name: gatewayName, Namespace: infraNamespace}
		poolNN := types.NamespacedName{Name: poolName, Namespace: appBackendNamespace}

		gatewayAddr := k8sutils.GetGatewayEndpoint(t, s.Client, s.TimeoutConfig, gatewayNN)

		t.Run("Scenario 1: HTTPRoute backendRef to InferencePool with Port Unspecified", func(t *testing.T) {
			routeNN := types.NamespacedName{Name: "httproute-pool-port-unspecified", Namespace: appBackendNamespace}
			hostname := "port-unspecified.example.com"
			path := "/test-port-unspecified"

			k8sutils.HTTPRouteMustBeAcceptedAndResolved(t, s.Client, s.TimeoutConfig, routeNN, gatewayNN)
			k8sutils.InferencePoolMustBeAcceptedByParent(t, s.Client, poolNN)

			trafficutils.MakeRequestAndExpectSuccess(
				t,
				s.RoundTripper,
				s.TimeoutConfig,
				gatewayAddr,
				hostname,
				path,
				backendDeploymentName,
				appBackendNamespace,
			)
		})

		t.Run("Scenario 2: HTTPRoute backendRef to InferencePool with Port Specified and Matching", func(t *testing.T) {
			routeNN := types.NamespacedName{Name: "httproute-pool-port-matching", Namespace: appBackendNamespace}
			hostname := "port-matching.example.com"
			path := "/test-port-matching"

			k8sutils.HTTPRouteMustBeAcceptedAndResolved(t, s.Client, s.TimeoutConfig, routeNN, gatewayNN)
			k8sutils.InferencePoolMustBeAcceptedByParent(t, s.Client, poolNN)

			trafficutils.MakeRequestAndExpectSuccess(
				t,
				s.RoundTripper,
				s.TimeoutConfig,
				gatewayAddr,
				hostname,
				path,
				backendDeploymentName,
				appBackendNamespace,
			)
		})

		// TODO: Add a warning check after the required change is made per discussion in github.com/kubernetes-sigs/gateway-api-inference-extension/discussions/918
		t.Run("Scenario 3: HTTPRoute backendRef to InferencePool with Port Specified and Non-Matching. Request still passing because HTTP Port is ignored when inferencePool is backendRef", func(t *testing.T) {
			routeNN := types.NamespacedName{Name: "httproute-pool-port-non-matching", Namespace: appBackendNamespace}
			hostname := "port-non-matching.example.com"
			path := "/test-port-non-matching"

			k8sutils.HTTPRouteMustBeAcceptedAndResolved(t, s.Client, s.TimeoutConfig, routeNN, gatewayNN)
			k8sutils.InferencePoolMustBeAcceptedByParent(t, s.Client, poolNN)

			trafficutils.MakeRequestAndExpectSuccess(
				t,
				s.RoundTripper,
				s.TimeoutConfig,
				gatewayAddr,
				hostname,
				path,
				backendDeploymentName,
				appBackendNamespace,
			)
		})
	},
}
