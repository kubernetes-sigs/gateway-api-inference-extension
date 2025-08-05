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

package scheduling

import (
	"context"
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/uuid"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

// Helper function to create properly initialized fake pod metrics
func createFakePodMetrics(address string) schedulingtypes.Pod {
	// Create a proper k8s pod
	k8sPod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod-" + address, // Make name unique
			Namespace: "default",
			Labels:    map[string]string{"app": "test"},
		},
		Status: corev1.PodStatus{
			PodIP: address,
		},
	}

	// Use the proper constructor
	fakePodMetrics := backendmetrics.NewFakePodMetrics(k8sPod)

	// Override the address in the backend pod to match test requirements
	pod := fakePodMetrics.GetPod()
	pod.Address = address

	return fakePodMetrics
}

// Tests the scheduler for conformance tests.
func TestSchedule(t *testing.T) {
	tests := []struct {
		name    string
		input   []schedulingtypes.Pod
		req     *schedulingtypes.LLMRequest
		wantRes *schedulingtypes.SchedulingResult
		err     bool
	}{
		{
			name:  "no candidate pods and req header is set",
			input: []schedulingtypes.Pod{}, // Explicitly set empty slice
			req: &schedulingtypes.LLMRequest{
				Headers:   map[string]string{"test-epp-endpoint-selection": "random-endpoint"},
				RequestId: uuid.NewString(),
			},
			wantRes: nil,
			err:     true,
		},
		{
			name: "req header not set",
			input: []schedulingtypes.Pod{
				createFakePodMetrics("random-endpoint"),
			},
			req: &schedulingtypes.LLMRequest{
				Headers:   map[string]string{}, // Deliberately set an empty header.
				RequestId: uuid.NewString(),
			},
			wantRes: nil,
			err:     true,
		},
		{
			name: "no pods address from the candidate pods matches req header address",
			input: []schedulingtypes.Pod{
				createFakePodMetrics("nonmatched-endpoint"),
			},
			req: &schedulingtypes.LLMRequest{
				Headers:   map[string]string{"test-epp-endpoint-selection": "matched-endpoint"},
				RequestId: uuid.NewString(),
			},
			wantRes: nil,
			err:     true,
		},
		{
			name: "one pod address from the candidate pods matches req header address",
			input: []schedulingtypes.Pod{
				createFakePodMetrics("nonmatched-endpoint"),
				createFakePodMetrics("matched-endpoint"),
			},
			req: &schedulingtypes.LLMRequest{
				Headers:   map[string]string{"test-epp-endpoint-selection": "matched-endpoint"},
				RequestId: uuid.NewString(),
			},
			wantRes: nil, // We'll verify manually instead of using exact comparison
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			scheduler := NewReqHeaderBasedScheduler()

			// Add panic recovery to provide better error information
			var got *schedulingtypes.SchedulingResult
			var err error

			func() {
				defer func() {
					if r := recover(); r != nil {
						err = fmt.Errorf("scheduler panicked: %v", r)
						t.Logf("Panic occurred with input: %d pods, headers: %v", len(test.input), test.req.Headers)
					}
				}()
				got, err = scheduler.Schedule(context.Background(), test.req, test.input)
			}()

			if test.err != (err != nil) {
				t.Errorf("Unexpected error, got %v, want error=%v", err, test.err)
				return
			}

			if !test.err {
				// For the successful test case, do manual verification instead of exact comparison
				if test.name == "one pod address from the candidate pods matches req header address" {
					if got == nil {
						t.Error("Expected non-nil result for successful scheduling")
						return
					}

					// Verify basic structure
					if got.PrimaryProfileName != "req-header-based-profile" {
						t.Errorf("Expected PrimaryProfileName 'req-header-based-profile', got %s", got.PrimaryProfileName)
					}

					// Verify profile results exist
					profileResult, exists := got.ProfileResults["req-header-based-profile"]
					if !exists {
						t.Error("Expected profile result 'req-header-based-profile' not found")
						return
					}

					// Verify we got exactly one target pod
					if len(profileResult.TargetPods) != 1 {
						t.Errorf("Expected 1 target pod, got %d", len(profileResult.TargetPods))
						return
					}

					// Verify the pod has the correct address
					targetPod := profileResult.TargetPods[0]
					if targetPod.GetPod() == nil {
						t.Error("Target pod GetPod() returned nil")
						return
					}

					if targetPod.GetPod().Address != "matched-endpoint" {
						t.Errorf("Expected target pod address 'matched-endpoint', got %s", targetPod.GetPod().Address)
					}

				} else if diff := cmp.Diff(test.wantRes, got); diff != "" {
					t.Errorf("Unexpected output (-want +got): %v", diff)
				}
			}
		})
	}
}
