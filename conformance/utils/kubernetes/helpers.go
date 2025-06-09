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

// Package kubernetes contains helper functions for interacting with
// Kubernetes objects within the conformance test suite.
package kubernetes

import (
	"context"
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"sigs.k8s.io/controller-runtime/pkg/client"

	inferenceapi "sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/conformance/utils/config"
	gatewayv1 "sigs.k8s.io/gateway-api/apis/v1"
	gatewayapiconfig "sigs.k8s.io/gateway-api/conformance/utils/config"
	gatewayk8sutils "sigs.k8s.io/gateway-api/conformance/utils/kubernetes"
)

// checkCondition is a helper function similar to findConditionInList or CheckCondition
// from the Gateway API conformance utilities.
// It checks if the expectedCondition is present in the conditions list.
// If expectedCondition.Reason is an empty string, it matches any reason.
func checkCondition(t *testing.T, conditions []metav1.Condition, expectedCondition metav1.Condition) bool {
	t.Helper()
	for _, cond := range conditions {
		if cond.Type == expectedCondition.Type {
			if cond.Status == expectedCondition.Status {
				if expectedCondition.Reason == "" || cond.Reason == expectedCondition.Reason {
					return true
				}
				t.Logf("Condition %s found with Status %s, but Reason %s did not match expected %s",
					expectedCondition.Type, cond.Status, cond.Reason, expectedCondition.Reason)
			} else {
				t.Logf("Condition %s found, but Status %s did not match expected %s",
					expectedCondition.Type, cond.Status, expectedCondition.Status)
			}
		}
	}
	t.Logf("Condition %s with Status %s (and Reason %s if specified) not found in conditions list: %+v",
		expectedCondition.Type, expectedCondition.Status, expectedCondition.Reason, conditions)
	return false
}

// InferencePoolMustHaveCondition waits for the specified InferencePool resource
// to exist and report the expected status condition within one of its parent statuses.
// It polls the InferencePool's status until the condition is met or the timeout occurs.
func InferencePoolMustHaveCondition(t *testing.T, c client.Client, poolNN types.NamespacedName, expectedCondition metav1.Condition) {
	t.Helper() // Marks this function as a test helper

	var timeoutConfig config.InferenceExtensionTimeoutConfig = config.DefaultInferenceExtensionTimeoutConfig()
	var lastObservedPool *inferenceapi.InferencePool
	var lastError error
	var conditionFound bool

	waitErr := wait.PollUntilContextTimeout(
		context.Background(),
		timeoutConfig.InferencePoolMustHaveConditionInterval,
		timeoutConfig.InferencePoolMustHaveConditionTimeout,
		true, func(ctx context.Context) (bool, error) {
			pool := &inferenceapi.InferencePool{} // This is the type instance used for Get
			err := c.Get(ctx, poolNN, pool)
			if err != nil {
				if apierrors.IsNotFound(err) {
					t.Logf("InferencePool %s not found yet. Retrying.", poolNN.String())
					lastError = err
					return false, nil
				}
				t.Logf("Error fetching InferencePool %s (type: %s): %v. Retrying.", poolNN.String(), reflect.TypeOf(pool).String(), err)
				lastError = err
				return false, nil
			}
			lastObservedPool = pool
			lastError = nil
			conditionFound = false

			if len(pool.Status.Parents) == 0 {
				t.Logf("InferencePool %s has no parent statuses reported yet.", poolNN.String())
				return false, nil
			}

			for _, parentStatus := range pool.Status.Parents {
				if checkCondition(t, parentStatus.Conditions, expectedCondition) {
					conditionFound = true
					return true, nil
				}
			}
			return false, nil
		})

	if waitErr != nil || !conditionFound {
		debugMsg := ""
		if waitErr != nil {
			debugMsg += fmt.Sprintf(" Polling error: %v.", waitErr)
		}
		if lastError != nil {
			debugMsg += fmt.Sprintf(" Last error during fetching: %v.", lastError)
		}

		if lastObservedPool != nil {
			debugMsg += "\nLast observed InferencePool status:"
			if len(lastObservedPool.Status.Parents) == 0 {
				debugMsg += " (No parent statuses reported)"
			}
			for i, parentStatus := range lastObservedPool.Status.Parents {
				debugMsg += fmt.Sprintf("\n  Parent %d (Gateway: %s/%s):", i, parentStatus.GatewayRef.Namespace, parentStatus.GatewayRef.Name)
				if len(parentStatus.Conditions) == 0 {
					debugMsg += " (No conditions reported for this parent)"
				}
				for _, cond := range parentStatus.Conditions {
					debugMsg += fmt.Sprintf("\n    - Type: %s, Status: %s, Reason: %s, Message: %s", cond.Type, cond.Status, cond.Reason, cond.Message)
				}
			}
		} else if lastError == nil || !apierrors.IsNotFound(lastError) {
			debugMsg += "\nInferencePool was not found or not observed successfully during polling."
		}

		finalMsg := fmt.Sprintf("timed out or condition not met for InferencePool %s to have condition Type=%s, Status=%s",
			poolNN.String(), expectedCondition.Type, expectedCondition.Status)
		if expectedCondition.Reason != "" {
			finalMsg += fmt.Sprintf(", Reason='%s'", expectedCondition.Reason)
		}
		finalMsg += "." + debugMsg
		require.FailNow(t, finalMsg)
	}

	logMsg := fmt.Sprintf("InferencePool %s successfully has condition Type=%s, Status=%s",
		poolNN.String(), expectedCondition.Type, expectedCondition.Status)
	if expectedCondition.Reason != "" {
		logMsg += fmt.Sprintf(", Reason='%s'", expectedCondition.Reason)
	}
	t.Log(logMsg)
}

// InferencePoolMustHaveNoParents waits for the specified InferencePool resource
// to exist and report that it has no parent references in its status.
// This typically indicates it is no longer referenced by any Gateway API resources.
func InferencePoolMustHaveNoParents(t *testing.T, c client.Client, poolNN types.NamespacedName) {
	t.Helper()

	var lastObservedPool *inferenceapi.InferencePool
	var lastError error
	var timeoutConfig config.InferenceExtensionTimeoutConfig = config.DefaultInferenceExtensionTimeoutConfig()

	ctx := context.Background()
	waitErr := wait.PollUntilContextTimeout(
		ctx,

		timeoutConfig.InferencePoolMustHaveConditionInterval,
		timeoutConfig.InferencePoolMustHaveConditionTimeout,
		true,
		func(pollCtx context.Context) (bool, error) {
			pool := &inferenceapi.InferencePool{}
			err := c.Get(pollCtx, poolNN, pool)
			if err != nil {
				if apierrors.IsNotFound(err) {
					t.Logf("InferencePool %s not found. Considering this as having no parents.", poolNN.String())
					lastError = nil
					return true, nil
				}
				t.Logf("Error fetching InferencePool %s: %v. Retrying.", poolNN.String(), err)
				lastError = err
				return false, nil
			}
			lastObservedPool = pool
			lastError = nil

			if len(pool.Status.Parents) == 0 {
				t.Logf("InferencePool %s successfully has no parent statuses.", poolNN.String())
				return true, nil
			}
			t.Logf("InferencePool %s still has %d parent statuses. Waiting...", poolNN.String(), len(pool.Status.Parents))
			return false, nil
		})

	if waitErr != nil {
		debugMsg := fmt.Sprintf("Timed out waiting for InferencePool %s to have no parent statuses.", poolNN.String())
		if lastError != nil {
			debugMsg += fmt.Sprintf(" Last error during fetching: %v.", lastError)
		}
		if lastObservedPool != nil && len(lastObservedPool.Status.Parents) > 0 {
			debugMsg += fmt.Sprintf(" Last observed InferencePool still had %d parent(s):", len(lastObservedPool.Status.Parents))
		} else if lastError == nil && (lastObservedPool == nil || len(lastObservedPool.Status.Parents) == 0) {
			debugMsg += " Polling completed without timeout, but an unexpected waitErr occurred."
		}
		require.FailNow(t, debugMsg, waitErr)
	}
	t.Logf("Successfully verified that InferencePool %s has no parent statuses.", poolNN.String())
}

// HTTPRouteMustBeAcceptedAndResolved waits for the specified HTTPRoute
// to be Accepted and have its references resolved by the specified Gateway.
// It uses the upstream Gateway API's HTTPRouteMustHaveCondition helper.
func HTTPRouteMustBeAcceptedAndResolved(t *testing.T, c client.Client, timeoutConfig gatewayapiconfig.TimeoutConfig, routeNN, gatewayNN types.NamespacedName) {
	t.Helper()

	acceptedCondition := metav1.Condition{
		Type:   string(gatewayv1.RouteConditionAccepted),
		Status: metav1.ConditionTrue,
		Reason: string(gatewayv1.RouteReasonAccepted),
	}

	resolvedRefsCondition := metav1.Condition{
		Type:   string(gatewayv1.RouteConditionResolvedRefs),
		Status: metav1.ConditionTrue,
		Reason: string(gatewayv1.RouteReasonResolvedRefs),
	}

	t.Logf("Waiting for HTTPRoute %s to be Accepted by Gateway %s", routeNN.String(), gatewayNN.String())
	gatewayk8sutils.HTTPRouteMustHaveCondition(t, c, timeoutConfig, routeNN, gatewayNN, acceptedCondition)

	t.Logf("Waiting for HTTPRoute %s to have ResolvedRefs by Gateway %s", routeNN.String(), gatewayNN.String())
	gatewayk8sutils.HTTPRouteMustHaveCondition(t, c, timeoutConfig, routeNN, gatewayNN, resolvedRefsCondition)

	t.Logf("HTTPRoute %s is now Accepted and has ResolvedRefs by Gateway %s", routeNN.String(), gatewayNN.String())
}

// HTTPRouteMustHaveConditions waits for the specified HTTPRoute to have a set of conditions
// reported by the specified Gateway. It checks each condition sequentially.
func HTTPRouteMustHaveConditions(t *testing.T, c client.Client, timeoutConfig gatewayapiconfig.TimeoutConfig, routeNN, gatewayNN types.NamespacedName, expectedConditions []metav1.Condition) {
	t.Helper()

	for _, condition := range expectedConditions {
		t.Logf("Waiting for HTTPRoute %s to have condition: Type=%s, Status=%s, Reason=%s",
			routeNN.String(), condition.Type, condition.Status, condition.Reason)
		gatewayk8sutils.HTTPRouteMustHaveCondition(t, c, timeoutConfig, routeNN, gatewayNN, condition)
	}
	t.Logf("Successfully verified all expected conditions for HTTPRoute %s on Gateway %s",
		routeNN.String(), gatewayNN.String())
}

// InferencePoolMustBeAcceptedByParent waits for the specified InferencePool
// to report an Accepted condition with status True and reason "Accepted"
// from at least one of its parent Gateways.
func InferencePoolMustBeAcceptedByParent(t *testing.T, c client.Client, poolNN types.NamespacedName) {
	t.Helper()

	expectedPoolCondition := metav1.Condition{
		Type:   string(gatewayv1.RouteConditionAccepted),
		Status: metav1.ConditionTrue,
		Reason: string(gatewayv1.RouteReasonAccepted),
	}

	// Call the existing generic helper with the predefined condition
	InferencePoolMustHaveCondition(t, c, poolNN, expectedPoolCondition)
	t.Logf("InferencePool %s successfully verified with RouteAccepted condition (Type: %s, Status: %s, Reason: %s).",
		poolNN.String(), expectedPoolCondition.Type, expectedPoolCondition.Status, expectedPoolCondition.Reason)
}

// HTTPRouteAndInferencePoolMustBeAcceptedAndRouteAccepted waits for the specified HTTPRoute
// to be Accepted and have its references resolved by the specified Gateway,
// AND for the specified InferencePool to be "RouteAccepted" using the specific
// RouteConditionAccepted criteria.
func HTTPRouteAndInferencePoolMustBeAcceptedAndRouteAccepted(
	t *testing.T,
	c client.Client,
	routeNN types.NamespacedName,
	gatewayNN types.NamespacedName,
	poolNN types.NamespacedName) {
	t.Helper()
	var timeoutConfig config.InferenceExtensionTimeoutConfig = config.DefaultInferenceExtensionTimeoutConfig()

	HTTPRouteMustBeAcceptedAndResolved(t, c, timeoutConfig.TimeoutConfig, routeNN, gatewayNN)
	InferencePoolMustBeRouteAccepted(t, c, poolNN)
	t.Logf("Successfully verified: HTTPRoute %s (Gateway %s) is Accepted & Resolved, and InferencePool %s is RouteAccepted.",
		routeNN.String(), gatewayNN.String(), poolNN.String())
}

// InferencePoolMustBeRouteAccepted waits for the specified InferencePool resource
// to exist and report an Accepted condition with Type=RouteConditionAccepted,
// Status=True, and Reason=RouteReasonAccepted within one of its parent statuses.
func InferencePoolMustBeRouteAccepted(t *testing.T, c client.Client, poolNN types.NamespacedName) {
	t.Helper()

	expectedPoolCondition := metav1.Condition{
		Type:   string(gatewayv1.RouteConditionAccepted),
		Status: metav1.ConditionTrue,
		Reason: string(gatewayv1.RouteReasonAccepted),
	}

	// Call the existing generic helper with the predefined condition
	InferencePoolMustHaveCondition(t, c, poolNN, expectedPoolCondition)
	t.Logf("InferencePool %s successfully verified with RouteAccepted condition (Type: %s, Status: %s, Reason: %s).",
		poolNN.String(), expectedPoolCondition.Type, expectedPoolCondition.Status, expectedPoolCondition.Reason)
}

// GetGatewayEndpoint waits for the specified Gateway to have at least one address
// and returns the address in "host:port" format.
// It leverages the upstream Gateway API's WaitForGatewayAddress.
func GetGatewayEndpoint(t *testing.T, k8sClient client.Client, timeoutConfig gatewayapiconfig.TimeoutConfig, gatewayNN types.NamespacedName) string {
	t.Helper()

	t.Logf("Waiting for Gateway %s/%s to get an address...", gatewayNN.Namespace, gatewayNN.Name)
	gwAddr, err := gatewayk8sutils.WaitForGatewayAddress(t, k8sClient, timeoutConfig, gatewayk8sutils.NewGatewayRef(gatewayNN))
	require.NoError(t, err, "failed to get Gateway address for %s", gatewayNN.String())
	require.NotEmpty(t, gwAddr, "Gateway %s has no address", gatewayNN.String())

	t.Logf("Gateway %s/%s has address: %s", gatewayNN.Namespace, gatewayNN.Name, gwAddr)
	return gwAddr
}

// GetPod waits for a Pod matching the specified labels to exist in the given
// namespace and have an IP address assigned. It's a test helper that fails the
// test on timeout or error.
func GetPod(t *testing.T, c client.Client, namespace string, selector labels.Selector, timeout time.Duration) *corev1.Pod {
	t.Helper()

	var pods corev1.PodList
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	waitErr := wait.PollUntilContextTimeout(ctx, 1*time.Second, timeout, true, func(ctx context.Context) (bool, error) {
		if err := c.List(ctx, &pods, &client.ListOptions{
			LabelSelector: selector,
			Namespace:     namespace,
		}); err != nil {
			t.Logf("Error listing pods with selector %s: %v. Retrying.", selector.String(), err)
			return false, nil
		}

		if len(pods.Items) > 0 {
			pod := pods.Items[0]
			if pod.Status.PodIP != "" && pod.Status.Phase == corev1.PodRunning {
				return true, nil
			}
			t.Logf("Pod %s found, but not yet running or has no IP. Current phase: %s, IP: '%s'. Retrying.", pod.Name, pod.Status.Phase, pod.Status.PodIP)
		} else {
			t.Logf("No pods found with selector %s yet. Retrying.", selector.String())
		}
		return false, nil
	})

	require.NoErrorf(t, waitErr, "timed out waiting for Pod with selector %s in namespace %s to be ready", selector.String(), namespace)
	require.NotEmpty(t, pods.Items, "expected at least one pod for selector %s in namespace %s, but found none", selector.String(), namespace)

	pod := &pods.Items[0]
	t.Logf("Successfully found ready Pod %s with IP %s for selector %s", pod.Name, pod.Status.PodIP, selector.String())
	return pod
}
