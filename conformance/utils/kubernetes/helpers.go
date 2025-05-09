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

	"github.com/stretchr/testify/require"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"sigs.k8s.io/controller-runtime/pkg/client"

	// Import the Inference Extension API types
	inferenceapi "sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2" // Adjust if your API version is different

	// Import necessary utilities from the core Gateway API conformance suite
	"sigs.k8s.io/gateway-api-inference-extension/conformance/utils/config"
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

// InferencePoolMustBeDeleted waits for the specified InferencePool resource to be fully deleted from the cluster.
// It polls until a Get request returns a "NotFound" error.
func InferencePoolMustBeDeleted(t *testing.T, c client.Client, poolNN types.NamespacedName) {
	t.Helper()

	var lastError error
	var timeoutConfig config.InferenceExtensionTimeoutConfig = config.DefaultInferenceExtensionTimeoutConfig()
	interval := timeoutConfig.InferencePoolMustHaveConditionInterval
	timeout := timeoutConfig.InferencePoolMustHaveConditionTimeout

	ctx := context.Background()

	waitErr := wait.PollUntilContextTimeout(ctx, interval, timeout, true, func(ctx context.Context) (bool, error) {
		pool := &inferenceapi.InferencePool{}
		err := c.Get(ctx, poolNN, pool)
		if err != nil {
			if apierrors.IsNotFound(err) {
				t.Logf("InferencePool %s successfully deleted (NotFound).", poolNN.String())
				return true, nil
			}
			lastError = fmt.Errorf("error fetching InferencePool %s: %w", poolNN.String(), err)
			t.Logf("%s. Retrying.", lastError.Error())
			return false, nil // Continue polling for other errors, hoping it resolves or becomes NotFound
		}
		lastError = fmt.Errorf("InferencePool %s still exists", poolNN.String())
		t.Logf("InferencePool %s still found. Retrying.", poolNN.String())
		return false, nil
	})

	if waitErr != nil {
		finalMsg := fmt.Sprintf("timed out waiting for InferencePool %s to be deleted.", poolNN.String())
		if lastError != nil {
			finalMsg += fmt.Sprintf(" Last observed error: %v.", lastError)
		}
		require.FailNow(t, finalMsg)
	}
}

// InferencePoolMustHaveSelector waits until the specified InferencePool's selector
// matches the expectedSelector or a timeout occurs.
func InferencePoolMustHaveSelector(t *testing.T, c client.Client, poolNN types.NamespacedName, expectedSelector map[inferenceapi.LabelKey]inferenceapi.LabelValue) {
	t.Helper()

	timeoutConf := config.DefaultInferenceExtensionTimeoutConfig()
	timeout := timeoutConf.InferencePoolMustHaveConditionTimeout
	interval := timeoutConf.InferencePoolMustHaveConditionInterval

	var lastObservedPool *inferenceapi.InferencePool
	var lastK8sError error
	var conditionMet bool

	ctx := context.Background()
	waitErr := wait.PollUntilContextTimeout(ctx, interval, timeout, true, func(pollCtx context.Context) (bool, error) {
		pool := &inferenceapi.InferencePool{}
		err := c.Get(pollCtx, poolNN, pool)
		if err != nil {
			if apierrors.IsNotFound(err) {
				t.Logf("InferencePool %s not found yet while waiting for selector. Retrying.", poolNN.String())
				lastK8sError = err
				lastObservedPool = nil
				return false, nil // Continue polling
			}
			t.Logf("Error fetching InferencePool %s during selector verification: %v. Retrying.", poolNN.String(), err)
			lastK8sError = err
			// lastObservedPool remains as is or nil if never fetched
			return false, nil // Continue polling, hoping it's transient
		}
		lastObservedPool = pool
		lastK8sError = nil // Clear last error if fetch is successful

		// Perform the check
		if expectedSelector == nil {
			conditionMet = (pool.Spec.Selector == nil)
			return conditionMet, nil
		}
		// If expectedSelector is not nil, pool.Spec.Selector also must not be nil for a match
		if pool.Spec.Selector == nil {
			conditionMet = false
			return false, nil
		}

		conditionMet = reflect.DeepEqual(pool.Spec.Selector, expectedSelector)
		return conditionMet, nil
	})

	if waitErr != nil || !conditionMet {
		errMsg := fmt.Sprintf("timed out or condition not met for InferencePool %s to have selector %+v.",
			poolNN.String(), expectedSelector)
		if waitErr != nil {
			errMsg += fmt.Sprintf(" Polling error: %v.", waitErr)
		}
		if lastK8sError != nil {
			errMsg += fmt.Sprintf(" Last K8s API error: %v.", lastK8sError)
		}
		if lastObservedPool != nil {
			errMsg += fmt.Sprintf(" Last observed selector: %+v.", lastObservedPool.Spec.Selector)
		} else if lastK8sError == nil || !apierrors.IsNotFound(lastK8sError) {
			errMsg += " InferencePool was not found or not observed successfully during polling."
		}
		require.FailNow(t, errMsg)
	}

	t.Logf("Successfully observed InferencePool %s/%s with selector %+v",
		poolNN.Namespace, poolNN.Name, expectedSelector)
}
