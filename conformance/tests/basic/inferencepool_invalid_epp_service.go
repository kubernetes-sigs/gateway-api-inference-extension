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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api/conformance/utils/kubernetes"
	"sigs.k8s.io/gateway-api/conformance/utils/suite"
	"sigs.k8s.io/gateway-api/pkg/features"

	"sigs.k8s.io/gateway-api-inference-extension/conformance/tests"
	gatewayv1 "sigs.k8s.io/gateway-api/apis/v1"
)

func init() {
	tests.ConformanceTests = append(tests.ConformanceTests, InferencePoolInvalidEPPService)
}

var InferencePoolInvalidEPPService = suite.ConformanceTest{
	ShortName:   "InferencePoolInvalidEPPService",
	Description: "Validate that a Gateway reports a failure status when an HTTPRoute references an InferencePool whose EPP service reference does not exist.",
	Manifests:   []string{"tests/basic/inferencepool_invalid_epp_service.yaml"},
	Features: []features.FeatureName{
		features.SupportGateway,
		features.SupportHTTPRoute,
		features.FeatureName("SupportInferencePool"),
	},
	Test: func(t *testing.T, s *suite.ConformanceTestSuite) {
		const (
			infraNamespace = "gateway-conformance-infra"
			gatewayName    = "conformance-gateway"
		)

		gatewayNN := types.NamespacedName{Name: gatewayName, Namespace: infraNamespace}

		t.Run("Verify Gateway reports Programmed:Invalid backend reference", func(t *testing.T) {
			expectedCondition := metav1.Condition{
				Type:   string(gatewayv1.GatewayConditionProgrammed),
				Status: metav1.ConditionFalse,
				Reason: string(gatewayv1.GatewayReasonInvalid),
			}

			kubernetes.GatewayMustHaveCondition(t, s.Client, s.TimeoutConfig, gatewayNN, expectedCondition)
		})
	},
}
