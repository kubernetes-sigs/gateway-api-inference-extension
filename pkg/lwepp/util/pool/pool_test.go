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
		name  string
		input *v1.InferencePool
		want  *datastore.EndpointPool
	}{
		{
			name:  "nil InferencePool returns nil",
			input: nil,
			want:  nil,
		},
		{
			name: "selector, ports and namespace are all carried over",
			input: &v1.InferencePool{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pool-1",
					Namespace: "ns-1",
				},
				Spec: v1.InferencePoolSpec{
					Selector: v1.LabelSelector{
						MatchLabels: map[v1.LabelKey]v1.LabelValue{
							"app":  "vllm",
							"tier": "backend",
						},
					},
					TargetPorts: []v1.Port{{Number: 8000}, {Number: 8001}},
				},
			},
			want: &datastore.EndpointPool{
				Selector:    map[string]string{"app": "vllm", "tier": "backend"},
				TargetPorts: []int{8000, 8001},
				Namespace:   "ns-1",
			},
		},
		{
			name: "target port order is preserved",
			input: &v1.InferencePool{
				ObjectMeta: metav1.ObjectMeta{Namespace: "ns-2"},
				Spec: v1.InferencePoolSpec{
					Selector: v1.LabelSelector{
						MatchLabels: map[v1.LabelKey]v1.LabelValue{"app": "vllm"},
					},
					TargetPorts: []v1.Port{{Number: 9002}, {Number: 9000}, {Number: 9001}},
				},
			},
			want: &datastore.EndpointPool{
				Selector:    map[string]string{"app": "vllm"},
				TargetPorts: []int{9002, 9000, 9001},
				Namespace:   "ns-2",
			},
		},
		{
			// The API permits an empty label value. It selects pods carrying the key
			// with an empty value, so the key must survive the conversion.
			name: "empty label value is preserved",
			input: &v1.InferencePool{
				ObjectMeta: metav1.ObjectMeta{Namespace: "ns-4"},
				Spec: v1.InferencePoolSpec{
					Selector: v1.LabelSelector{
						MatchLabels: map[v1.LabelKey]v1.LabelValue{"app": ""},
					},
					TargetPorts: []v1.Port{{Number: 8000}},
				},
			},
			want: &datastore.EndpointPool{
				Selector:    map[string]string{"app": ""},
				TargetPorts: []int{8000},
				Namespace:   "ns-4",
			},
		},
		{
			// PortNumber is an int32 narrowed to int; pin both ends of the valid range.
			name: "port numbers at both bounds of the valid range",
			input: &v1.InferencePool{
				ObjectMeta: metav1.ObjectMeta{Namespace: "ns-5"},
				Spec: v1.InferencePoolSpec{
					Selector: v1.LabelSelector{
						MatchLabels: map[v1.LabelKey]v1.LabelValue{"app": "vllm"},
					},
					TargetPorts: []v1.Port{{Number: 1}, {Number: 65535}},
				},
			},
			want: &datastore.EndpointPool{
				Selector:    map[string]string{"app": "vllm"},
				TargetPorts: []int{1, 65535},
				Namespace:   "ns-5",
			},
		},
		{
			name: "empty selector and no target ports yield empty, non-nil fields",
			input: &v1.InferencePool{
				ObjectMeta: metav1.ObjectMeta{Namespace: "ns-3"},
				Spec:       v1.InferencePoolSpec{},
			},
			want: &datastore.EndpointPool{
				Selector:    map[string]string{},
				TargetPorts: []int{},
				Namespace:   "ns-3",
			},
		},
		{
			name: "empty namespace is carried over as empty",
			input: &v1.InferencePool{
				Spec: v1.InferencePoolSpec{
					Selector: v1.LabelSelector{
						MatchLabels: map[v1.LabelKey]v1.LabelValue{"app": "vllm"},
					},
					TargetPorts: []v1.Port{{Number: 8000}},
				},
			},
			want: &datastore.EndpointPool{
				Selector:    map[string]string{"app": "vllm"},
				TargetPorts: []int{8000},
				Namespace:   "",
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := InferencePoolToEndpointPool(test.input)
			if diff := cmp.Diff(test.want, got); diff != "" {
				t.Errorf("InferencePoolToEndpointPool() returned unexpected result (-want +got): %v", diff)
			}
		})
	}
}
