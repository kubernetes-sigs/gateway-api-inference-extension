package basic

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	gatewayv1 "sigs.k8s.io/gateway-api/apis/v1"
	gwhttp "sigs.k8s.io/gateway-api/conformance/utils/http"
	"sigs.k8s.io/gateway-api/conformance/utils/suite"
	"sigs.k8s.io/gateway-api/pkg/features"

	"sigs.k8s.io/gateway-api-inference-extension/conformance/tests"
	k8sutils "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/kubernetes"
)

func init() {
	tests.ConformanceTests = append(tests.ConformanceTests, InferencePoolParentStatus)
}

var InferencePoolParentStatus = suite.ConformanceTest{
	ShortName:   "InferencePoolResolvedRefsCondition",
	Description: "Verify that an InferencePool correctly updates its parent-specific status (e.g., Accepted condition) when referenced by HTTPRoutes attached to shared Gateways, and clears parent statuses when no longer referenced.",
	Manifests:   []string{"tests/basic/inferencepool_resolvedrefs_condition.yaml"},
	Features: []features.FeatureName{
		features.FeatureName("SupportInferencePool"),
		features.SupportGateway,
	},
	Test: func(t *testing.T, s *suite.ConformanceTestSuite) {
		const (
			appBackendNamespace   = "gateway-conformance-app-backend"
			infraNamespace        = "gateway-conformance-infra"
			poolName              = "multi-gateway-pool"
			sharedGateway1Name    = "conformance-gateway"
			sharedGateway2Name    = "conformance-secondary-gateway"
			httpRoute1Name        = "httproute-for-gw1"
			httpRoute2Name        = "httproute-for-gw2"
			hostnameGw1           = "gw1.example.com"
			pathGw1               = "/conformance-gateway-test"
			hostnameGw2           = "secondary.example.com"
			pathGw2               = "/gateway-2-test"
			backendServicePodName = "infra-backend-deployment"
		)

		poolNN := types.NamespacedName{Name: poolName, Namespace: appBackendNamespace}
		httpRoute1NN := types.NamespacedName{Name: httpRoute1Name, Namespace: appBackendNamespace}
		httpRoute2NN := types.NamespacedName{Name: httpRoute2Name, Namespace: appBackendNamespace}
		gateway1NN := types.NamespacedName{Name: sharedGateway1Name, Namespace: infraNamespace}
		gateway2NN := types.NamespacedName{Name: sharedGateway2Name, Namespace: infraNamespace}

		k8sutils.HTTPRouteMustBeAcceptedAndResolved(t, s.Client, s.TimeoutConfig, httpRoute1NN, gateway1NN)
		k8sutils.HTTPRouteMustBeAcceptedAndResolved(t, s.Client, s.TimeoutConfig, httpRoute2NN, gateway2NN)

		gw1Addr := k8sutils.GetGatewayEndpoint(t, s.Client, s.TimeoutConfig, gateway1NN)
		gw2Addr := k8sutils.GetGatewayEndpoint(t, s.Client, s.TimeoutConfig, gateway2NN)

		t.Run("InferencePool should show Accepted:True by parents and be routable via multiple HTTPRoutes", func(t *testing.T) {
			k8sutils.InferencePoolMustBeAcceptedByParent(t, s.Client, poolNN)
			t.Logf("InferencePool %s has parent status Accepted:True as expected with two references.", poolNN.String())

			expectedResponseGw1 := gwhttp.ExpectedResponse{
				Request: gwhttp.Request{
					Host:   hostnameGw1,
					Path:   pathGw1,
					Method: "GET",
				},
				ExpectedRequest: &gwhttp.ExpectedRequest{
					Request: gwhttp.Request{
						Host:   hostnameGw1,
						Path:   pathGw1,
						Method: "GET",
					},
				},
				Response: gwhttp.Response{
					StatusCode: http.StatusOK,
				},
				Backend:   backendServicePodName,
				Namespace: appBackendNamespace,
			}
			gwhttp.MakeRequestAndExpectEventuallyConsistentResponse(t, s.RoundTripper, s.TimeoutConfig, gw1Addr, expectedResponseGw1)

			expectedResponseGw2 := gwhttp.ExpectedResponse{
				Request: gwhttp.Request{
					Host:   hostnameGw2,
					Path:   pathGw2,
					Method: "GET",
				},
				ExpectedRequest: &gwhttp.ExpectedRequest{
					Request: gwhttp.Request{
						Host:   hostnameGw2,
						Path:   pathGw2,
						Method: "GET",
					},
				},
				Response: gwhttp.Response{
					StatusCode: http.StatusOK,
				},
				Backend:   backendServicePodName,
				Namespace: appBackendNamespace,
			}
			gwhttp.MakeRequestAndExpectEventuallyConsistentResponse(t, s.RoundTripper, s.TimeoutConfig, gw2Addr, expectedResponseGw2)
		})

		t.Run("Delete httproute-for-gw1 and verify InferencePool status and routing via gw2", func(t *testing.T) {
			httproute1 := &gatewayv1.HTTPRoute{
				ObjectMeta: metav1.ObjectMeta{Name: httpRoute1NN.Name, Namespace: httpRoute1NN.Namespace},
			}
			t.Logf("Deleting HTTPRoute %s", httpRoute1NN.String())
			require.NoError(t, s.Client.Delete(context.TODO(), httproute1), "failed to delete httproute-for-gw1")

			t.Logf("Waiting for %v for Gateway conditions to update after deleting HTTPRoute %s", s.TimeoutConfig.GatewayMustHaveCondition, httpRoute1NN.String())
			time.Sleep(s.TimeoutConfig.GatewayMustHaveCondition)

			k8sutils.InferencePoolMustBeAcceptedByParent(t, s.Client, poolNN)
			t.Logf("InferencePool %s still has parent status Accepted:True as expected with one reference remaining.", poolNN.String())

			expectedResponseGw2StillOk := gwhttp.ExpectedResponse{
				Request: gwhttp.Request{
					Host:   hostnameGw2,
					Path:   pathGw2,
					Method: "GET",
				},
				ExpectedRequest: &gwhttp.ExpectedRequest{
					Request: gwhttp.Request{
						Host:   hostnameGw2,
						Path:   pathGw2,
						Method: "GET",
					},
				},
				Response: gwhttp.Response{
					StatusCode: http.StatusOK,
				},
				Backend:   backendServicePodName,
				Namespace: appBackendNamespace,
			}
			gwhttp.MakeRequestAndExpectEventuallyConsistentResponse(t, s.RoundTripper, s.TimeoutConfig, gw2Addr, expectedResponseGw2StillOk)

			expectedResponseGw1NotFound := gwhttp.ExpectedResponse{
				Request: gwhttp.Request{
					Host:   hostnameGw1,
					Path:   pathGw1,
					Method: "GET",
				},
				Response: gwhttp.Response{
					StatusCode: http.StatusNotFound,
				},
			}
			gwhttp.MakeRequestAndExpectEventuallyConsistentResponse(t, s.RoundTripper, s.TimeoutConfig, gw1Addr, expectedResponseGw1NotFound)
		})

		t.Run("Delete httproute-for-gw2 and verify InferencePool has no parent statuses and is not routable", func(t *testing.T) {
			httproute2 := &gatewayv1.HTTPRoute{
				ObjectMeta: metav1.ObjectMeta{Name: httpRoute2NN.Name, Namespace: httpRoute2NN.Namespace},
			}
			t.Logf("Deleting HTTPRoute %s", httpRoute2NN.String())
			require.NoError(t, s.Client.Delete(context.TODO(), httproute2), "failed to delete httproute-for-gw2")

			k8sutils.InferencePoolMustHaveNoParents(t, s.Client, poolNN)
			t.Logf("InferencePool %s correctly shows no parent statuses, indicating it's no longer referenced.", poolNN.String())

			expectedResponseGw2NotFound := gwhttp.ExpectedResponse{
				Request: gwhttp.Request{
					Host:   hostnameGw2,
					Path:   pathGw2,
					Method: "GET",
				},
				Response: gwhttp.Response{
					StatusCode: http.StatusNotFound,
				},
			}
			gwhttp.MakeRequestAndExpectEventuallyConsistentResponse(t, s.RoundTripper, s.TimeoutConfig, gw2Addr, expectedResponseGw2NotFound)
		})

		t.Logf("InferencePoolResolvedRefsCondition test completed.")
	},
}
