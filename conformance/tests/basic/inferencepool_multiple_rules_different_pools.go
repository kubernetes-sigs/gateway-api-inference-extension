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
	gatewayv1 "sigs.k8s.io/gateway-api/apis/v1" // For standard condition types
	"sigs.k8s.io/gateway-api/conformance/utils/suite"
	"sigs.k8s.io/gateway-api/pkg/features" // For standard feature names

	// Import the tests package to append to ConformanceTests
	"sigs.k8s.io/gateway-api-inference-extension/conformance/tests"
	infrakubernetes "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/kubernetes"
)

func init() {
	// Register the InferencePoolAccepted test case with the conformance suite.
	// This ensures it will be discovered and run by the test runner.
	tests.ConformanceTests = append(tests.ConformanceTests, InferencePoolAccepted)
}

// InferencePoolAccepted defines the test case for verifying basic InferencePool acceptance.
var InferencePoolAccepted = suite.ConformanceTest{
	ShortName:   "InferencePoolAccepted",
	Description: "A minimal InferencePool resource should be accepted by the controller and report an Accepted condition",
	Manifests:   []string{"tests/basic/inferencepool_accepted.yaml"},
	Features:    []features.FeatureName{},
	Test: func(t *testing.T, s *suite.ConformanceTestSuite) {
		// created by the associated manifest file.
		poolNN := types.NamespacedName{Name: "inferencepool-basic-accepted", Namespace: "gateway-conformance-app-backend"}

		t.Run("InferencePool should have Accepted condition set to True", func(t *testing.T) {
			// Define the expected status condition. We use the standard "Accepted"
			// condition type from the Gateway API for consistency.
			acceptedCondition := metav1.Condition{
				Type:   string(gatewayv1.GatewayConditionAccepted), // Standard condition type
				Status: metav1.ConditionTrue,
				Reason: "", // "" means we don't strictly check the Reason for this basic test.
			}
			infrakubernetes.InferencePoolMustHaveCondition(t, s.Client, poolNN, acceptedCondition)
		})
	},
}
