/*
Copyright 2024 The Kubernetes Authors.

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

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// InferencePool is the Schema for the InferencePools API.
//
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +genclient
type InferencePool struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   InferencePoolSpec   `json:"spec,omitempty"`
	Status InferencePoolStatus `json:"status,omitempty"`
}

// InferencePoolList contains a list of InferencePool.
//
// +kubebuilder:object:root=true
type InferencePoolList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []InferencePool `json:"items"`
}

// InferencePoolSpec defines the desired state of InferencePool
type InferencePoolSpec struct {
	// Selector defines a map of label to watch model server pods
	// that should be included in the InferencePool. ModelServers should not
	// be with any other Service or InferencePool, that behavior is not supported
	// and will result in sub-optimal utilization.
	// In some cases, implementations may translate this to a Service selector, so this matches the simple
	// map used for Service selectors instead of the full Kubernetes LabelSelector type.
	//
	// +kubebuilder:validation:Required
	Selector map[LabelKey]LabelValue `json:"selector"`

	// TargetPortNumber defines the port number to access the selected model servers.
	// The number must be in the range 1 to 65535.
	//
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=65535
	// +kubebuilder:validation:Required
	TargetPortNumber int32 `json:"targetPortNumber"`
}

// LabelKey was originally copied from: https://github.com/kubernetes-sigs/gateway-api/blob/99a3934c6bc1ce0874f3a4c5f20cafd8977ffcb4/apis/v1/shared_types.go#L694-L731
// Duplicated as to not take an unexpected dependency on gw's API.
//
// LabelKey is the key of a label. This is used for validation
// of maps. This matches the Kubernetes "qualified name" validation that is used for labels.
// // Labels are case sensitive, so: my-label and My-Label are considered distinct.
//
// Valid values include:
//
// * example
// * example.com
// * example.com/path
// * example.com/path.html
//
// Invalid values include:
//
// * example~ - "~" is an invalid character
// * example.com. - can not start or end with "."
//
// +kubebuilder:validation:MinLength=1
// +kubebuilder:validation:MaxLength=253
// +kubebuilder:validation:Pattern=`^([a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*/)?([A-Za-z0-9][-A-Za-z0-9_.]{0,61})?[A-Za-z0-9]$`
type LabelKey string

// LabelValue is the value of a label. This is used for validation
// of maps. This matches the Kubernetes label validation rules:
// * must be 63 characters or less (can be empty),
// * unless empty, must begin and end with an alphanumeric character ([a-z0-9A-Z]),
// * could contain dashes (-), underscores (_), dots (.), and alphanumerics between.
//
// Valid values include:
//
// * MyValue
// * my.name
// * 123-my-value
//
// +kubebuilder:validation:MinLength=0
// +kubebuilder:validation:MaxLength=63
// +kubebuilder:validation:Pattern=`^(([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9])?$`
type LabelValue string

// InferencePoolStatus defines the observed state of InferencePool
type InferencePoolStatus struct {
	// Conditions track the state of the InferencePool.
	//
	// Known condition types are:
	//
	// * "Ready"
	//
	// +optional
	// +listType=map
	// +listMapKey=type
	// +kubebuilder:validation:MaxItems=8
	// +kubebuilder:default={{type: "Ready", status: "Unknown", reason:"Pending", message:"Waiting for controller", lastTransitionTime: "1970-01-01T00:00:00Z"}}
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// InferencePoolConditionType is a type of condition for the InferencePool
type InferencePoolConditionType string

// InferencePoolConditionReason is the reason for a given InferencePoolConditionType
type InferencePoolConditionReason string

const (
	// This condition indicates whether the pool is ready for traffic or not, and why.
	PoolConditionReady InferencePoolConditionType = "Ready"

	// Desired state. The pool and its components are initialized and ready for traffic.
	PoolReasonReady InferencePoolConditionReason = "Ready"

	// This reason is used when the EPP has not yet passed health checks, or has started failing them.
	PoolReasonEPPNotHealthy InferencePoolConditionReason = "EndpointPickerNotHealthy"

	// This reason is the initial state, and indicates that the controller has not yet reconciled this pool.
	PoolReasonPending InferencePoolConditionReason = "Pending"
)

func init() {
	SchemeBuilder.Register(&InferencePool{}, &InferencePoolList{})
}
