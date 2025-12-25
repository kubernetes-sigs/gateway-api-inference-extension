package tests

import (
	"testing"

	"sigs.k8s.io/gateway-api/conformance/utils/suite"
	"sigs.k8s.io/gateway-api/pkg/features"

	"sigs.k8s.io/gateway-api-inference-extension/conformance/resources"
	k8sutils "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/kubernetes"
)

func init() {
	ConformanceTests = append(ConformanceTests, InferencePoolControllerName)
}

var InferencePoolControllerName = suite.ConformanceTest{
	ShortName:   "InferencePoolControllerName",
	Description: "InferencePool status parents should include controllerName",
	Manifests:   []string{"tests/inferencepool_accepted.yaml"},
	Features: []features.FeatureName{
		features.FeatureName("SupportInferencePool"),
		features.SupportGateway,
	},
	Test: func(t *testing.T, s *suite.ConformanceTestSuite) {
		poolNN := resources.PrimaryInferencePoolNN

		t.Run("InferencePool parent status reports controllerName", func(t *testing.T) {
			k8sutils.InferencePoolMustHaveControllerName(t, s.Client, poolNN)
		})
	},
}
