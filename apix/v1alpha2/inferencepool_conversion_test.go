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
	v1FailureMode = v1.EndpointPickerFailureMode("Deny")
	v1Port        = v1.Port{Number: 9000}
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
					APIVersion: GroupVersion.String(),
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
					ExtensionRef: Extension{
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
								{ // v1alpha2 default condition is skipped from conversion.
									Type:               string(v1.InferencePoolConditionAccepted),
									Status:             metav1.ConditionUnknown,
									Reason:             string(InferencePoolReasonPending),
									LastTransitionTime: timestamp,
								},
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
					APIVersion: v1.GroupVersion.String(),
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
					TargetPorts: []v1.Port{{Number: v1.PortNumber(int32(8080))}},
					EndpointPickerRef: v1.EndpointPickerRef{
						Group:       &v1Group,
						Kind:        v1Kind,
						Name:        "my-epp-service",
						Port:        &v1Port,
						FailureMode: v1FailureMode,
					},
				},
				Status: v1.InferencePoolStatus{
					Parents: []v1.ParentStatus{
						{
							ParentRef: v1.ParentReference{Name: "my-gateway"},
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
			name: "conversion from v1alpha2 to v1 with empty extensionRef and default v1alpha2 status condition",
			src: &InferencePool{
				TypeMeta: metav1.TypeMeta{
					Kind:       "InferencePool",
					APIVersion: GroupVersion.String(),
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
									Type:               string(v1.InferencePoolConditionAccepted),
									Status:             metav1.ConditionUnknown,
									Reason:             string(InferencePoolReasonPending),
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
					APIVersion: v1.GroupVersion.String(),
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
					TargetPorts: []v1.Port{{Number: v1.PortNumber(int32(8080))}},
				},
				Status: v1.InferencePoolStatus{
					Parents: []v1.ParentStatus{
						{
							ParentRef: v1.ParentReference{Name: "my-gateway"},
							// Conditions omitted (nil) after dropping the synthetic default.
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "v1alpha2 -> v1 maps NotSupportedByGateway to NotSupportedByParent",
			src: &InferencePool{
				TypeMeta: metav1.TypeMeta{
					Kind:       "InferencePool",
					APIVersion: GroupVersion.String(),
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pool",
					Namespace: "ns",
				},
				Spec: InferencePoolSpec{
					Selector:         map[LabelKey]LabelValue{"app": "m"},
					TargetPortNumber: 8080,
				},
				Status: InferencePoolStatus{
					Parents: []PoolStatus{
						{
							GatewayRef: ParentGatewayReference{Name: "gw"},
							Conditions: []metav1.Condition{
								{
									Type:               string(InferencePoolConditionAccepted),
									Status:             metav1.ConditionFalse,
									Reason:             string(InferencePoolReasonNotSupportedByGateway),
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
					APIVersion: v1.GroupVersion.String(),
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pool",
					Namespace: "ns",
				},
				Spec: v1.InferencePoolSpec{
					Selector: v1.LabelSelector{MatchLabels: map[v1.LabelKey]v1.LabelValue{"app": "m"}},
					TargetPorts: []v1.Port{
						{Number: v1.PortNumber(8080)},
					},
				},
				Status: v1.InferencePoolStatus{
					Parents: []v1.ParentStatus{
						{
							ParentRef: v1.ParentReference{Name: "gw"},
							Conditions: []metav1.Condition{
								{
									Type:               string(v1.InferencePoolConditionAccepted),
									Status:             metav1.ConditionFalse,
									Reason:             string(v1.InferencePoolReasonNotSupportedByParent),
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
			name: "v1alpha2 -> v1 drops synthetic default parent",
			src: &InferencePool{
				TypeMeta: metav1.TypeMeta{
					Kind:       "InferencePool",
					APIVersion: GroupVersion.String(),
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pool",
					Namespace: "ns",
				},
				Spec: InferencePoolSpec{
					Selector:         map[LabelKey]LabelValue{"app": "m"},
					TargetPortNumber: 8080,
				},
				Status: InferencePoolStatus{
					Parents: []PoolStatus{
						{
							GatewayRef: func() ParentGatewayReference {
								k := Kind("Status")
								return ParentGatewayReference{
									Kind: &k, Name: "default",
								}
							}(),
							Conditions: []metav1.Condition{
								{
									Type:               string(InferencePoolConditionAccepted),
									Status:             metav1.ConditionUnknown,
									Reason:             string(InferencePoolReasonPending),
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
					APIVersion: v1.GroupVersion.String(),
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pool",
					Namespace: "ns",
				},
				Spec: v1.InferencePoolSpec{
					Selector:    v1.LabelSelector{MatchLabels: map[v1.LabelKey]v1.LabelValue{"app": "m"}},
					TargetPorts: []v1.Port{{Number: v1.PortNumber(8080)}},
				},
				Status: v1.InferencePoolStatus{
					// All parents dropped -> empty status.
				},
			},
			wantErr: false,
		},
		{
			name: "v1alpha2 -> v1 drops synthetic default parent but keeps real parent",
			src: &InferencePool{
				TypeMeta: metav1.TypeMeta{
					Kind:       "InferencePool",
					APIVersion: GroupVersion.String(),
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pool",
					Namespace: "ns",
				},
				Spec: InferencePoolSpec{
					Selector:         map[LabelKey]LabelValue{"app": "m"},
					TargetPortNumber: 8080,
				},
				Status: InferencePoolStatus{
					Parents: []PoolStatus{
						{
							GatewayRef: func() ParentGatewayReference {
								k := Kind("Status")
								return ParentGatewayReference{Kind: &k, Name: "default"}
							}(),
							Conditions: []metav1.Condition{
								{
									Type:               string(InferencePoolConditionAccepted),
									Status:             metav1.ConditionUnknown,
									Reason:             string(InferencePoolReasonPending),
									LastTransitionTime: timestamp,
								},
							},
						},
						{
							GatewayRef: ParentGatewayReference{Name: "real-gw"},
							Conditions: []metav1.Condition{
								{
									Type:               string(InferencePoolConditionResolvedRefs),
									Status:             metav1.ConditionTrue,
									Reason:             string(InferencePoolReasonResolvedRefs),
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
					APIVersion: v1.GroupVersion.String(),
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pool",
					Namespace: "ns",
				},
				Spec: v1.InferencePoolSpec{
					Selector:    v1.LabelSelector{MatchLabels: map[v1.LabelKey]v1.LabelValue{"app": "m"}},
					TargetPorts: []v1.Port{{Number: v1.PortNumber(8080)}},
				},
				Status: v1.InferencePoolStatus{
					Parents: []v1.ParentStatus{
						{
							ParentRef: v1.ParentReference{Name: "real-gw"},
							Conditions: []metav1.Condition{
								{
									Type:               string(v1.InferencePoolConditionResolvedRefs),
									Status:             metav1.ConditionTrue,
									Reason:             string(v1.InferencePoolReasonResolvedRefs),
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
					APIVersion: v1.GroupVersion.String(),
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
					TargetPorts: []v1.Port{{Number: v1.PortNumber(int32(8080))}},
					EndpointPickerRef: v1.EndpointPickerRef{
						Group:       &v1Group,
						Kind:        v1Kind,
						Name:        "my-epp-service",
						Port:        &v1Port,
						FailureMode: v1FailureMode,
					},
				},
				Status: v1.InferencePoolStatus{
					Parents: []v1.ParentStatus{
						{
							ParentRef: v1.ParentReference{Name: "my-gateway"},
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
					APIVersion: GroupVersion.String(),
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
					ExtensionRef: Extension{
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
			name: "conversion from v1 to v1alpha2 with empty extensionRef and nil status condition",
			src: &v1.InferencePool{
				TypeMeta: metav1.TypeMeta{
					Kind:       "InferencePool",
					APIVersion: v1.GroupVersion.String(),
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
					TargetPorts: []v1.Port{{Number: v1.PortNumber(int32(8080))}},
				},
				Status: v1.InferencePoolStatus{
					Parents: []v1.ParentStatus{
						{
							ParentRef: v1.ParentReference{Name: "my-gateway"},
						},
					},
				},
			},
			want: &InferencePool{
				TypeMeta: metav1.TypeMeta{
					Kind:       "InferencePool",
					APIVersion: GroupVersion.String(),
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
						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "conversion from v1 to v1alpha2 with empty extensionRef and empty status condition",
			src: &v1.InferencePool{
				TypeMeta: metav1.TypeMeta{
					Kind:       "InferencePool",
					APIVersion: v1.GroupVersion.String(),
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
					TargetPorts: []v1.Port{{Number: v1.PortNumber(int32(8080))}},
				},
				Status: v1.InferencePoolStatus{
					Parents: []v1.ParentStatus{
						{
							ParentRef:  v1.ParentReference{Name: "my-gateway"},
							Conditions: []metav1.Condition{},
						},
					},
				},
			},
			want: &InferencePool{
				TypeMeta: metav1.TypeMeta{
					Kind:       "InferencePool",
					APIVersion: GroupVersion.String(),
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
							Conditions: nil, // Conditions omitted because the converter leaves it nil when v1 had [].

						},
					},
				},
			},
			wantErr: false,
		},
		{
			name: "v1 -> v1alpha2 maps NotSupportedByParent to NotSupportedByGateway",
			src: &v1.InferencePool{
				TypeMeta: metav1.TypeMeta{
					Kind:       "InferencePool",
					APIVersion: v1.GroupVersion.String(),
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pool",
					Namespace: "ns",
				},
				Spec: v1.InferencePoolSpec{
					Selector:    v1.LabelSelector{MatchLabels: map[v1.LabelKey]v1.LabelValue{"app": "m"}},
					TargetPorts: []v1.Port{{Number: v1.PortNumber(8080)}},
				},
				Status: v1.InferencePoolStatus{
					Parents: []v1.ParentStatus{
						{
							ParentRef: v1.ParentReference{Name: "gw"},
							Conditions: []metav1.Condition{
								{
									Type:               string(v1.InferencePoolConditionAccepted),
									Status:             metav1.ConditionFalse,
									Reason:             string(v1.InferencePoolReasonNotSupportedByParent),
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
					APIVersion: GroupVersion.String(),
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pool",
					Namespace: "ns",
				},
				Spec: InferencePoolSpec{
					Selector:         map[LabelKey]LabelValue{"app": "m"},
					TargetPortNumber: 8080,
				},
				Status: InferencePoolStatus{
					Parents: []PoolStatus{
						{
							GatewayRef: ParentGatewayReference{Name: "gw"},
							Conditions: []metav1.Condition{
								{
									Type:               string(InferencePoolConditionAccepted),
									Status:             metav1.ConditionFalse,
									Reason:             string(InferencePoolReasonNotSupportedByGateway),
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
