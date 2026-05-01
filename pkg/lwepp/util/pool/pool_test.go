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

package pool

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/lwepp/datastore"
)

func TestInferencePoolToEndpointPool(t *testing.T) {
	tests := []struct {
		name     string
		pool     *v1.InferencePool
		expected *datastore.EndpointPool
	}{
		{
			name:     "nil input returns nil",
			pool:     nil,
			expected: nil,
		},
		{
			name: "single port and single label",
			pool: &v1.InferencePool{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pool",
					Namespace: "default",
				},
				Spec: v1.InferencePoolSpec{
					TargetPorts: []v1.Port{{Number: 8080}},
					Selector: v1.LabelSelector{
						MatchLabels: map[v1.LabelKey]v1.LabelValue{
							"app": "model-server",
						},
					},
				},
			},
			expected: &datastore.EndpointPool{
				Namespace:   "default",
				TargetPorts: []int{8080},
				Selector:    map[string]string{"app": "model-server"},
			},
		},
		{
			name: "multiple ports preserve order",
			pool: &v1.InferencePool{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "multi-port-pool",
					Namespace: "production",
				},
				Spec: v1.InferencePoolSpec{
					TargetPorts: []v1.Port{
						{Number: 8080},
						{Number: 9090},
						{Number: 5000},
					},
					Selector: v1.LabelSelector{
						MatchLabels: map[v1.LabelKey]v1.LabelValue{
							"app": "llm",
						},
					},
				},
			},
			expected: &datastore.EndpointPool{
				Namespace:   "production",
				TargetPorts: []int{8080, 9090, 5000},
				Selector:    map[string]string{"app": "llm"},
			},
		},
		{
			name: "multiple labels all converted",
			pool: &v1.InferencePool{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "labeled-pool",
					Namespace: "staging",
				},
				Spec: v1.InferencePoolSpec{
					TargetPorts: []v1.Port{{Number: 8000}},
					Selector: v1.LabelSelector{
						MatchLabels: map[v1.LabelKey]v1.LabelValue{
							"app":     "model-server",
							"version": "v1",
							"env":     "staging",
						},
					},
				},
			},
			expected: &datastore.EndpointPool{
				Namespace:   "staging",
				TargetPorts: []int{8000},
				Selector: map[string]string{
					"app":     "model-server",
					"version": "v1",
					"env":     "staging",
				},
			},
		},
		{
			name: "empty selector produces empty map",
			pool: &v1.InferencePool{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "no-labels-pool",
					Namespace: "default",
				},
				Spec: v1.InferencePoolSpec{
					TargetPorts: []v1.Port{{Number: 8000}},
					Selector:    v1.LabelSelector{},
				},
			},
			expected: &datastore.EndpointPool{
				Namespace:   "default",
				TargetPorts: []int{8000},
				Selector:    map[string]string{},
			},
		},
		{
			name: "name field is not carried over",
			pool: &v1.InferencePool{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "named-pool",
					Namespace: "default",
				},
				Spec: v1.InferencePoolSpec{
					TargetPorts: []v1.Port{{Number: 8000}},
					Selector: v1.LabelSelector{
						MatchLabels: map[v1.LabelKey]v1.LabelValue{
							"app": "server",
						},
					},
				},
			},
			expected: &datastore.EndpointPool{
				Namespace:   "default",
				TargetPorts: []int{8000},
				Selector:    map[string]string{"app": "server"},
			},
		},
		{
			name: "max valid port number converts correctly",
			pool: &v1.InferencePool{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "edge-port-pool",
					Namespace: "default",
				},
				Spec: v1.InferencePoolSpec{
					TargetPorts: []v1.Port{{Number: 65535}},
					Selector: v1.LabelSelector{
						MatchLabels: map[v1.LabelKey]v1.LabelValue{
							"app": "server",
						},
					},
				},
			},
			expected: &datastore.EndpointPool{
				Namespace:   "default",
				TargetPorts: []int{65535},
				Selector:    map[string]string{"app": "server"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := InferencePoolToEndpointPool(tt.pool)
			if diff := cmp.Diff(tt.expected, result); diff != "" {
				t.Errorf("InferencePoolToEndpointPool() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
