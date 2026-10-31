/*
Copyright 2026 The Kubernetes Authors.

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

package kubernetes

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/lwepp/metadata"
)

// WaitForServedEndpoint observes response metadata independently of the client response.
// Only records from this test's unique request ID count; backend echo headers cannot
// satisfy the assertion. The caller bounds both polling and Kubernetes API requests.
func WaitForServedEndpoint(ctx context.Context, client kubernetes.Interface, namespace, selector, requestID, expected string, since time.Time, interval time.Duration) error {
	var lastAPIError error
	err := wait.PollUntilContextCancel(ctx, interval, true, func(ctx context.Context) (bool, error) {
		pods, err := client.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{LabelSelector: selector})
		if err != nil {
			if apierrors.IsUnauthorized(err) || apierrors.IsForbidden(err) {
				return false, fmt.Errorf("list EPP pods: %w", err)
			}
			lastAPIError = fmt.Errorf("list EPP pods: %w", err)
			return false, nil
		}
		for _, pod := range pods.Items {
			data, err := client.CoreV1().Pods(namespace).GetLogs(pod.Name, &corev1.PodLogOptions{Container: "epp", SinceTime: &metav1.Time{Time: since}}).DoRaw(ctx)
			if err != nil {
				if apierrors.IsUnauthorized(err) || apierrors.IsForbidden(err) {
					return false, fmt.Errorf("read EPP %s logs: %w", pod.Name, err)
				}
				lastAPIError = fmt.Errorf("read EPP %s logs: %w", pod.Name, err)
				continue
			}
			found, err := servedEndpointObserved(string(data), requestID, expected)
			if err != nil || found {
				return found, err
			}
		}
		return false, nil
	})
	if err != nil && lastAPIError != nil {
		return fmt.Errorf("%w (last Kubernetes API error: %v)", err, lastAPIError)
	}
	return err
}

func servedEndpointObserved(logs, requestID, expected string) (bool, error) {
	scanner := bufio.NewScanner(strings.NewReader(logs))
	// Unrelated verbose log entries may exceed Scanner's default limit.
	scanner.Buffer(make([]byte, 4096), 1024*1024)
	found := false
	var lastEndpoint string
	for scanner.Scan() {
		var event map[string]json.RawMessage
		if json.Unmarshal(scanner.Bytes(), &event) != nil {
			continue
		}
		var message, id, endpoint string
		if json.Unmarshal(event["msg"], &message) != nil || message != metadata.ServedEndpointObservation ||
			json.Unmarshal(event[metadata.ConformanceRequestIDLogKey], &id) != nil || id != requestID {
			continue
		}
		// Stabilization sends multiple requests with the same ID. Only the last
		// observation determines the result once the gateway becomes consistent.
		_ = json.Unmarshal(event[metadata.ServedEndpointLogKey], &endpoint)
		lastEndpoint = endpoint
		found = true
	}
	if err := scanner.Err(); err != nil {
		return false, err
	}
	if found && lastEndpoint != expected {
		return false, fmt.Errorf("request %s: EPP reported served endpoint %q, expected %q", requestID, lastEndpoint, expected)
	}
	return found, nil
}
