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

package v1alpha2

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
)

var (
	group       = Group("my-group")
	kind        = Kind("MyKind")
	failureMode = ExtensionFailureMode("Deny")
	portNumber  = PortNumber(9000)
	timestamp   = metav1.Unix(0, 0)

	v1Group       = v1.Group("my-group")
	v1Kind        = v1.Kind("MyKind")
	v1FailureMode = v1.ExtensionFailureMode("Deny")
	v1PortNumber  = v1.PortNumber(9000)
)

func TestInferencePoolConvertTo(t *testing.T) {
	tests := []struct {
		name    string
		src     *InferencePool
		want    *v1.InferencePool
		wantErr bool
	}{
		{
			name: "full conversion from v1alpha2 to v1 including status",
			src: &InferencePool{
				TypeMeta: metav1.TypeMeta{
					Kind:       "InferencePool",
					APIVersion: "inference.networking.x-k8s.io/v1alpha2",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pool",
					Namespace: "test-ns",
				},
				Spec: InferencePoolSpec{
					Selector: map[LabelKey]LabelValue{
						"app": "my-model-server",
					},
					TargetPortNumber: 8080,
					ExtensionRef: &Extension{
						Group:       &group,
						Kind:        &kind,
						Name:        "my-epp-service",
						PortNumber:  &portNumber,
						FailureMode: &failureMode,
					},
				},
				Status: InferencePoolStatus{
					Parents: []PoolStatus{
						{
							GatewayRef: ParentGatewayReference{Name: "my-gateway"},
							Conditions: []metav1.Condition{
								{
									Type:               string(InferencePoolConditionAccepted),
									Status:             metav1.ConditionTrue,
									Reason:             string(InferencePoolReasonAccepted),
									LastTransitionTime: timestamp,
								},
							},
						},
					},
				},
			},
			want: &v1.InferencePool{
				TypeMeta: metav1.TypeMeta{
					Kind:       "InferencePool",
					APIVersion: "inference.networking.x-k8s.io/v1alpha2",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pool",
					Namespace: "test-ns",
				},
				Spec: v1.InferencePoolSpec{
					Selector: v1.LabelSelector{
						MatchLabels: map[v1.LabelKey]v1.LabelValue{
							"app": "my-model-server",
						},
					},
					TargetPortNumber: 8080,
					ExtensionRef: &v1.Extension{
						Group:       &v1Group,
						Kind:        &v1Kind,
						Name:        "my-epp-service",
						PortNumber:  &v1PortNumber,
						FailureMode: &v1FailureMode,
					},
				},
				Status: v1.InferencePoolStatus{
					Parents: []v1.PoolStatus{
						{
							GatewayRef: v1.ParentGatewayReference{Name: "my-gateway"},
							Conditions: []metav1.Condition{
								{
									Type:               string(v1.InferencePoolConditionAccepted),
									Status:             metav1.ConditionTrue,
									Reason:             string(v1.InferencePoolReasonAccepted),
									LastTransitionTime: timestamp,
								},
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "conversion from v1alpha2 to v1 with nil extensionRef",
			src: &InferencePool{
				TypeMeta: metav1.TypeMeta{
					Kind:       "InferencePool",
					APIVersion: "inference.networking.x-k8s.io/v1alpha2",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pool",
					Namespace: "test-ns",
				},
				Spec: InferencePoolSpec{
					Selector: map[LabelKey]LabelValue{
						"app": "my-model-server",
					},
					TargetPortNumber: 8080,
				},
				Status: InferencePoolStatus{
					Parents: []PoolStatus{
						{
							GatewayRef: ParentGatewayReference{Name: "my-gateway"},
							Conditions: []metav1.Condition{
								{
									Type:               string(InferencePoolConditionAccepted),
									Status:             metav1.ConditionTrue,
									Reason:             string(InferencePoolReasonAccepted),
									LastTransitionTime: timestamp,
								},
							},
						},
					},
				},
			},
			want: &v1.InferencePool{
				TypeMeta: metav1.TypeMeta{
					Kind:       "InferencePool",
					APIVersion: "inference.networking.x-k8s.io/v1alpha2",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pool",
					Namespace: "test-ns",
				},
				Spec: v1.InferencePoolSpec{
					Selector: v1.LabelSelector{
						MatchLabels: map[v1.LabelKey]v1.LabelValue{
							"app": "my-model-server",
						},
					},
					TargetPortNumber: 8080,
				},
				Status: v1.InferencePoolStatus{
					Parents: []v1.PoolStatus{
						{
							GatewayRef: v1.ParentGatewayReference{Name: "my-gateway"},
							Conditions: []metav1.Condition{
								{
									Type:               string(v1.InferencePoolConditionAccepted),
									Status:             metav1.ConditionTrue,
									Reason:             string(v1.InferencePoolReasonAccepted),
									LastTransitionTime: timestamp,
								},
							},
						},
					},
				},
			},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := &v1.InferencePool{}
			err := tt.src.ConvertTo(got)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ConvertTo() error = %v, wantErr %v", err, tt.wantErr)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("ConvertTo() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func TestInferencePoolConvertFrom(t *testing.T) {
	tests := []struct {
		name    string
		src     *v1.InferencePool
		want    *InferencePool
		wantErr bool
	}{
		{
			name: "full conversion from v1 to v1alpha2 including status",
			src: &v1.InferencePool{
				TypeMeta: metav1.TypeMeta{
					Kind:       "InferencePool",
					APIVersion: "inference.networking.k8s.io/v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pool",
					Namespace: "test-ns",
				},
				Spec: v1.InferencePoolSpec{
					Selector: v1.LabelSelector{
						MatchLabels: map[v1.LabelKey]v1.LabelValue{
							"app": "my-model-server",
						},
					},
					TargetPortNumber: 8080,
					ExtensionRef: &v1.Extension{
						Group:       &v1Group,
						Kind:        &v1Kind,
						Name:        "my-epp-service",
						PortNumber:  &v1PortNumber,
						FailureMode: &v1FailureMode,
					},
				},
				Status: v1.InferencePoolStatus{
					Parents: []v1.PoolStatus{
						{
							GatewayRef: v1.ParentGatewayReference{Name: "my-gateway"},
							Conditions: []metav1.Condition{
								{
									Type:               string(v1.InferencePoolConditionAccepted),
									Status:             metav1.ConditionTrue,
									Reason:             string(v1.InferencePoolReasonAccepted),
									LastTransitionTime: timestamp,
								},
							},
						},
					},
				},
			},
			want: &InferencePool{
				TypeMeta: metav1.TypeMeta{
					Kind:       "InferencePool",
					APIVersion: "inference.networking.k8s.io/v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pool",
					Namespace: "test-ns",
				},
				Spec: InferencePoolSpec{
					Selector: map[LabelKey]LabelValue{
						"app": "my-model-server",
					},
					TargetPortNumber: 8080,
					ExtensionRef: &Extension{
						Group:       &group,
						Kind:        &kind,
						Name:        "my-epp-service",
						PortNumber:  &portNumber,
						FailureMode: &failureMode,
					},
				},
				Status: InferencePoolStatus{
					Parents: []PoolStatus{
						{
							GatewayRef: ParentGatewayReference{Name: "my-gateway"},
							Conditions: []metav1.Condition{
								{
									Type:               string(InferencePoolConditionAccepted),
									Status:             metav1.ConditionTrue,
									Reason:             string(InferencePoolReasonAccepted),
									LastTransitionTime: timestamp,
								},
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "conversion from v1 to v1alpha2 with nil extensionRef",
			src: &v1.InferencePool{
				TypeMeta: metav1.TypeMeta{
					Kind:       "InferencePool",
					APIVersion: "inference.networking.k8s.io/v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pool",
					Namespace: "test-ns",
				},
				Spec: v1.InferencePoolSpec{
					Selector: v1.LabelSelector{
						MatchLabels: map[v1.LabelKey]v1.LabelValue{
							"app": "my-model-server",
						},
					},
					TargetPortNumber: 8080,
				},
				Status: v1.InferencePoolStatus{
					Parents: []v1.PoolStatus{
						{
							GatewayRef: v1.ParentGatewayReference{Name: "my-gateway"},
							Conditions: []metav1.Condition{
								{
									Type:               string(v1.InferencePoolConditionAccepted),
									Status:             metav1.ConditionTrue,
									Reason:             string(v1.InferencePoolReasonAccepted),
									LastTransitionTime: timestamp,
								},
							},
						},
					},
				},
			},
			want: &InferencePool{
				TypeMeta: metav1.TypeMeta{
					Kind:       "InferencePool",
					APIVersion: "inference.networking.k8s.io/v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-pool",
					Namespace: "test-ns",
				},
				Spec: InferencePoolSpec{
					Selector: map[LabelKey]LabelValue{
						"app": "my-model-server",
					},
					TargetPortNumber: 8080,
				},
				Status: InferencePoolStatus{
					Parents: []PoolStatus{
						{
							GatewayRef: ParentGatewayReference{Name: "my-gateway"},
							Conditions: []metav1.Condition{
								{
									Type:               string(InferencePoolConditionAccepted),
									Status:             metav1.ConditionTrue,
									Reason:             string(InferencePoolReasonAccepted),
									LastTransitionTime: timestamp,
								},
							},
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name:    "nil source",
			src:     nil,
			want:    &InferencePool{},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := &InferencePool{}
			err := got.ConvertFrom(tt.src)
			if (err != nil) != tt.wantErr {
				t.Fatalf("ConvertFrom() error = %v, wantErr %v", err, tt.wantErr)
			}
			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("ConvertFrom() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
