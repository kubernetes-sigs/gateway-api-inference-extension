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

package pod

import (
	"reflect"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
)

func TestExtractActivePorts(t *testing.T) {
	tests := []struct {
		name           string
		pod            *corev1.Pod
		expectedPorts  sets.Set[int]
		expectedExists bool
	}{
		{
			name: "Pod without active ports annotation",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "test-pod",
					Namespace:   "default",
					Annotations: map[string]string{},
				},
			},
			expectedPorts:  nil,
			expectedExists: false,
		},
		{
			name: "Pod with empty active ports annotation",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "test-pod",
					Namespace:   "default",
					Annotations: map[string]string{activePortsAnnotation: ""},
				},
			},
			expectedPorts:  sets.New[int](),
			expectedExists: true,
		},
		{
			name: "Pod with single port in annotation",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "test-pod",
					Namespace:   "default",
					Annotations: map[string]string{activePortsAnnotation: "8000"},
				},
			},
			expectedPorts:  sets.New[int](8000),
			expectedExists: true,
		},
		{
			name: "Pod with multiple ports in annotation",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "test-pod",
					Namespace:   "default",
					Annotations: map[string]string{activePortsAnnotation: "8000,8001,8002"},
				},
			},
			expectedPorts:  sets.New[int](8000, 8001, 8002),
			expectedExists: true,
		},
		{
			name: "Pod with multiple ports with spaces in annotation",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "test-pod",
					Namespace:   "default",
					Annotations: map[string]string{activePortsAnnotation: "8000, 8001 , 8002"},
				},
			},
			expectedPorts:  sets.New[int](8000, 8001, 8002),
			expectedExists: true,
		},
		{
			name: "Pod with invalid port in annotation (non-numeric)",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "test-pod",
					Namespace:   "default",
					Annotations: map[string]string{activePortsAnnotation: "8000,invalid,8002"},
				},
			},
			expectedPorts:  sets.New[int](8000, 8002),
			expectedExists: true,
		},
		{
			name: "Pod with invalid port in annotation (negative number)",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "test-pod",
					Namespace:   "default",
					Annotations: map[string]string{activePortsAnnotation: "8000,-1,8002"},
				},
			},
			expectedPorts:  sets.New[int](8000, 8002),
			expectedExists: true,
		},
		{
			name: "Pod with duplicate ports in annotation",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "test-pod",
					Namespace:   "default",
					Annotations: map[string]string{activePortsAnnotation: "8000,8001,8000"},
				},
			},
			expectedPorts:  sets.New[int](8000, 8001),
			expectedExists: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ports, exists := ExtractActivePorts(tt.pod)

			if exists != tt.expectedExists {
				t.Errorf("ExtractActivePorts() exists = %v, want %v", exists, tt.expectedExists)
			}

			if !reflect.DeepEqual(ports, tt.expectedPorts) {
				t.Errorf("ExtractActivePorts() ports = %v, want %v", ports, tt.expectedPorts)
			}
		})
	}
}

func TestIsPodReady(t *testing.T) {
	tests := []struct {
		name     string
		pod      *corev1.Pod
		expected bool
	}{
		{
			name: "Pod with Ready condition True",
			pod: &corev1.Pod{
				Status: corev1.PodStatus{
					Conditions: []corev1.PodCondition{
						{
							Type:   corev1.PodReady,
							Status: corev1.ConditionTrue,
						},
					},
				},
			},
			expected: true,
		},
		{
			name: "Pod with Ready condition False",
			pod: &corev1.Pod{
				Status: corev1.PodStatus{
					Conditions: []corev1.PodCondition{
						{
							Type:   corev1.PodReady,
							Status: corev1.ConditionFalse,
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "Pod with Ready condition Unknown",
			pod: &corev1.Pod{
				Status: corev1.PodStatus{
					Conditions: []corev1.PodCondition{
						{
							Type:   corev1.PodReady,
							Status: corev1.ConditionUnknown,
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "Pod with deletion timestamp",
			pod: &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					DeletionTimestamp: &metav1.Time{Time: time.Now()},
				},
				Status: corev1.PodStatus{
					Conditions: []corev1.PodCondition{
						{
							Type:   corev1.PodReady,
							Status: corev1.ConditionTrue,
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "Pod without Ready condition",
			pod: &corev1.Pod{
				Status: corev1.PodStatus{
					Conditions: []corev1.PodCondition{
						{
							Type:   corev1.PodScheduled,
							Status: corev1.ConditionTrue,
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "Pod with no conditions",
			pod: &corev1.Pod{
				Status: corev1.PodStatus{
					Conditions: []corev1.PodCondition{},
				},
			},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := IsPodReady(tt.pod)
			if result != tt.expected {
				t.Errorf("IsPodReady() = %v, want %v", result, tt.expected)
			}
		})
	}
}
