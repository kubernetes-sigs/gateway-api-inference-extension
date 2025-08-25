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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// InferencePool is the Schema for the InferencePools API.
// +kubebuilder:object:root=true
// +kubebuilder:metadata:annotations="api-approved.kubernetes.io=unapproved, experimental-only"
// +kubebuilder:resource:shortName=xinfpool
// +kubebuilder:subresource:status
// +kubebuilder:storageversion
// +genclient
type InferencePool struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec InferencePoolSpec `json:"spec,omitempty"`

	// Status defines the observed state of InferencePool.
	//
	// +kubebuilder:default={parent: {{parentRef: {kind: "Status", name: "default"}, conditions: {{type: "Accepted", status: "Unknown", reason: "Pending", message: "Waiting for controller", lastTransitionTime: "1970-01-01T00:00:00Z"}}}}}
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
	// Selector defines a map of labels to watch model server Pods
	// that should be included in the InferencePool.
	// In some cases, implementations may translate this field to a Service selector, so this matches the simple
	// map used for Service selectors instead of the full Kubernetes LabelSelector type.
	// If specified, it will be applied to match the model server pods in the same namespace as the InferencePool.
	// Cross namesoace selector is not supported.
	//
	// +kubebuilder:validation:Required
	Selector map[LabelKey]LabelValue `json:"selector"`

	// TargetPortNumber defines the port number to access the selected model server Pods.
	// The number must be in the range 1 to 65535.
	//
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=65535
	// +kubebuilder:validation:Required
	TargetPortNumber int32 `json:"targetPortNumber"`

	// Extension configures an endpoint picker as an extension service.
	// +required
	ExtensionRef Extension `json:"extensionRef,omitempty"`
}

// Extension specifies how to configure an extension that runs the endpoint picker.
type Extension struct {
	// Group is the group of the referent.
	// The default value is "", representing the Core API group.
	//
	// +optional
	// +kubebuilder:default=""
	Group *Group `json:"group,omitempty"`

	// Kind is the Kubernetes resource kind of the referent.
	//
	// Defaults to "Service" when not specified.
	//
	// ExternalName services can refer to CNAME DNS records that may live
	// outside of the cluster and as such are difficult to reason about in
	// terms of conformance. They also may not be safe to forward to (see
	// CVE-2021-25740 for more information). Implementations MUST NOT
	// support ExternalName Services.
	//
	// +optional
	// +kubebuilder:default=Service
	Kind *Kind `json:"kind,omitempty"`

	// Name is the name of the referent.
	//
	// +kubebuilder:validation:Required
	Name ObjectName `json:"name"`

	// The port number on the service running the extension. When unspecified,
	// implementations SHOULD infer a default value of 9002 when the Kind is
	// Service.
	//
	// +optional
	PortNumber *PortNumber `json:"portNumber,omitempty"`

	// Configures how the gateway handles the case when the extension is not responsive.
	// Defaults to failClose.
	//
	// +optional
	// +kubebuilder:default="FailClose"
	FailureMode *ExtensionFailureMode `json:"failureMode"`
}

// ExtensionFailureMode defines the options for how the gateway handles the case when the extension is not
// responsive.
// +kubebuilder:validation:Enum=FailOpen;FailClose
type ExtensionFailureMode string

const (
	// FailOpen specifies that the proxy should forward the request to an endpoint of its picking when the Endpoint Picker fails.
	FailOpen ExtensionFailureMode = "FailOpen"
	// FailClose specifies that the proxy should drop the request when the Endpoint Picker fails.
	FailClose ExtensionFailureMode = "FailClose"
)

// InferencePoolStatus defines the observed state of InferencePool.
type InferencePoolStatus struct {
	// Parents is a list of parent resources (usually Gateways) that are
	// associated with the InferencePool, and the status of the InferencePool with respect to
	// each parent.
	//
	// A maximum of 32 Gateways will be represented in this list. When the list contains
	// `kind: Status, name: default`, it indicates that the InferencePool is not
	// associated with any Gateway and a controller must perform the following:
	//
	//  - Remove the parent when setting the "Accepted" condition.
	//  - Add the parent when the controller will no longer manage the InferencePool
	//    and no other parents exist.
	//
	// +kubebuilder:validation:MaxItems=32
	Parents []PoolStatus `json:"parent,omitempty"`
}

// PoolStatus defines the observed state of InferencePool from a Gateway.
type PoolStatus struct {
	// GatewayRef indicates the gateway that observed state of InferencePool.
	GatewayRef ParentGatewayReference `json:"parentRef"`

	// Conditions track the state of the InferencePool.
	//
	// Known condition types are:
	//
	// * "Accepted"
	// * "ResolvedRefs"
	//
	// +optional
	// +listType=map
	// +listMapKey=type
	// +kubebuilder:validation:MaxItems=8
	// +kubebuilder:default={{type: "Accepted", status: "Unknown", reason:"Pending", message:"Waiting for controller", lastTransitionTime: "1970-01-01T00:00:00Z"}}
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// InferencePoolConditionType is a type of condition for the InferencePool
type InferencePoolConditionType string

// InferencePoolReason is the reason for a given InferencePoolConditionType
type InferencePoolReason string

const (
	// This condition indicates whether the InferencePool has been accepted or rejected
	// by a Gateway, and why.
	//
	// Possible reasons for this condition to be True are:
	//
	// * "Accepted"
	//
	// Possible reasons for this condition to be False are:
	//
	// * "NotSupportedByGateway"
	// * "HTTPRouteNotAccepted"
	//
	// Possible reasons for this condition to be Unknown are:
	//
	// * "Pending"
	//
	// Controllers MAY raise this condition with other reasons, but should
	// prefer to use the reasons listed above to improve interoperability.
	InferencePoolConditionAccepted InferencePoolConditionType = "Accepted"

	// This reason is used with the "Accepted" condition when the InferencePool has been
	// accepted by the Gateway.
	InferencePoolReasonAccepted InferencePoolReason = "Accepted"

	// This reason is used with the "Accepted" condition when the InferencePool
	// has not been accepted by a Gateway because the Gateway does not support
	// InferencePool as a backend.
	InferencePoolReasonNotSupportedByGateway InferencePoolReason = "NotSupportedByGateway"

	// This reason is used with the "Accepted" condition when the InferencePool is
	// referenced by an HTTPRoute that has been rejected by the Gateway. The user
	// should inspect the status of the referring HTTPRoute for the specific reason.
	InferencePoolReasonHTTPRouteNotAccepted InferencePoolReason = "HTTPRouteNotAccepted"

	// This reason is used with the "Accepted" when a controller has not yet
	// reconciled the InferencePool.
	InferencePoolReasonPending InferencePoolReason = "Pending"
)

const (
	// This condition indicates whether the controller was able to resolve all
	// the object references for the InferencePool.
	//
	// Possible reasons for this condition to be True are:
	//
	// * "ResolvedRefs"
	//
	// Possible reasons for this condition to be False are:
	//
	// * "InvalidExtensionRef"
	//
	// Controllers MAY raise this condition with other reasons, but should
	// prefer to use the reasons listed above to improve interoperability.
	InferencePoolConditionResolvedRefs InferencePoolConditionType = "ResolvedRefs"

	// This reason is used with the "ResolvedRefs" condition when the condition
	// is true.
	InferencePoolReasonResolvedRefs InferencePoolReason = "ResolvedRefs"

	// This reason is used with the "ResolvedRefs" condition when the
	// Extension is invalid in some way. This can include an unsupported kind
	// or API group, or a reference to a resource that can not be found.
	InferencePoolReasonInvalidExtensionRef InferencePoolReason = "InvalidExtensionRef"
)

// ParentGatewayReference identifies an API object including its namespace,
// defaulting to Gateway.
type ParentGatewayReference struct {
	// Group is the group of the referent.
	//
	// +optional
	// +kubebuilder:default="gateway.networking.k8s.io"
	Group *Group `json:"group"`

	// Kind is kind of the referent. For example "Gateway".
	//
	// +optional
	// +kubebuilder:default=Gateway
	Kind *Kind `json:"kind"`

	// Name is the name of the referent.
	Name ObjectName `json:"name"`

	// Namespace is the namespace of the referent.  If not present,
	// the namespace of the referent is assumed to be the same as
	// the namespace of the referring object.
	//
	// +optional
	Namespace *Namespace `json:"namespace,omitempty"`
}
