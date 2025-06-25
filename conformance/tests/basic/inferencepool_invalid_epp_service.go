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
	"sigs.k8s.io/gateway-api/conformance/utils/kubernetes"
	"sigs.k8s.io/gateway-api/conformance/utils/suite"
	"sigs.k8s.io/gateway-api/pkg/features"

	"sigs.k8s.io/gateway-api-inference-extension/conformance/tests"
	trafficutils "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/traffic"
)

func init() {
	tests.ConformanceTests = append(tests.ConformanceTests, InferencePoolInvalidEPPService)
}

var InferencePoolInvalidEPPService = suite.ConformanceTest{
	ShortName:   "InferencePoolInvalidEPPService",
	Description: "An HTTPRoute that references an InferencePool with a non-existent EPP service should have a ResolvedRefs condition with a status of False and a reason of BackendNotFound.",
	Manifests:   []string{"tests/basic/inferencepool_invalid_epp_service.yaml"},
	Features: []features.FeatureName{
		features.SupportGateway,
		features.SupportHTTPRoute,
		features.FeatureName("SupportInferencePool"),
	},
	Test: func(t *testing.T, s *suite.ConformanceTestSuite) {
		const (
			routePath      = "/invalid-epp-test"
			infraNamespace = "gateway-conformance-infra"
			appNamespace   = "gateway-conformance-app-backend"
		)

		routeNN := types.NamespacedName{Name: "httproute-for-invalid-epp-pool", Namespace: appNamespace}
		gwNN := types.NamespacedName{Name: "conformance-primary-gateway", Namespace: infraNamespace}

		gwAddr := kubernetes.GatewayAndHTTPRoutesMustBeAccepted(t, s.Client, s.TimeoutConfig, s.ControllerName, kubernetes.NewGatewayRef(gwNN), routeNN)

		t.Run("Request to a route with an invalid backend reference receives a 500 response", func(t *testing.T) {
			trafficutils.MakeRequestAndExpectEventuallyConsistentResponse(t, s.RoundTripper, s.TimeoutConfig, gwAddr, trafficutils.Request{
				Path:               routePath,
				ExpectedStatusCode: 5, // Expecting response status code 5XX.
			})
		})
	},
}
