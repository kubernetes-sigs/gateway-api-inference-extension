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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/lwepp/metadata"
)

// WaitForServedEndpoint observes response metadata independently of the client response.
// Only records from this test's unique request ID count; backend echo headers cannot
// satisfy the assertion. The caller bounds both polling and Kubernetes API requests.
func WaitForServedEndpoint(ctx context.Context, client kubernetes.Interface, namespace, selector, requestID, expected string, since time.Time, interval time.Duration) error {
	return wait.PollUntilContextCancel(ctx, interval, true, func(ctx context.Context) (bool, error) {
		pods, err := client.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{LabelSelector: selector})
		if err != nil {
			return false, fmt.Errorf("list EPP pods: %w", err)
		}
		for _, pod := range pods.Items {
			data, err := client.CoreV1().Pods(namespace).GetLogs(pod.Name, &corev1.PodLogOptions{Container: "epp", SinceTime: &metav1.Time{Time: since}}).DoRaw(ctx)
			if err != nil {
				return false, fmt.Errorf("read EPP %s logs: %w", pod.Name, err)
			}
			found, err := servedEndpointObserved(string(data), requestID, expected)
			if err != nil || found {
				return found, err
			}
		}
		return false, nil
	})
}

func servedEndpointObserved(logs, requestID, expected string) (bool, error) {
	scanner := bufio.NewScanner(strings.NewReader(logs))
	// Unrelated verbose log entries may exceed Scanner's default limit.
	scanner.Buffer(make([]byte, 4096), 1024*1024)
	found := false
	for scanner.Scan() {
		var event struct {
			Message   string `json:"msg"`
			RequestID string `json:"conformanceRequestID"`
			Endpoint  string `json:"servedEndpoint"`
		}
		if json.Unmarshal(scanner.Bytes(), &event) != nil || event.Message != metadata.ServedEndpointObservation || event.RequestID != requestID {
			continue
		}
		if event.Endpoint != expected {
			return false, fmt.Errorf("request %s: EPP reported served endpoint %q, expected %q", requestID, event.Endpoint, expected)
		}
		found = true
	}
	return found, scanner.Err()
}
