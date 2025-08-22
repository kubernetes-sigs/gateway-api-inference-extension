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

package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// InferencePool is the Schema for the InferencePools API.
//
// +kubebuilder:object:root=true
// TODO: change the annotation once it gets officially approved
// +kubebuilder:metadata:annotations="api-approved.kubernetes.io=unapproved, experimental-only"
// +kubebuilder:resource:shortName=infpool
// +kubebuilder:subresource:status
// +kubebuilder:storageversion
// +genclient
type InferencePool struct {
	metav1.TypeMeta `json:",inline"`

	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the desired state of the InferencePool.
	//
	// +required
	Spec InferencePoolSpec `json:"spec,omitzero"`

	// Status defines the observed state of the InferencePool.
	//
	// +optional
	Status *InferencePoolStatus `json:"status,omitempty"`
}

// InferencePoolList contains a list of InferencePools.
//
// +kubebuilder:object:root=true
type InferencePoolList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []InferencePool `json:"items"`
}

// InferencePoolSpec defines the desired state of the InferencePool.
type InferencePoolSpec struct {
	// Selector determines which Pods are members of this inference pool.
	// It matches Pods by their labels only within the same namespace; cross-namespace
	// selection is not supported.
	//
	// The structure of this LabelSelector is intentionally simple to be compatible
	// with Kubernetes Service selectors, as some implementations may translate
	// this configuration into a Service resource.
	//
	// +required
	Selector LabelSelector `json:"selector,omitzero"`

	// TargetPorts defines a list of ports that are exposed by this InferencePool.
	// Currently, the list may only include a single port definition.
	//
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:MaxItems=1
	// +listType=atomic
	// +required
	TargetPorts []Port `json:"targetPorts,omitempty"`

	// EndpointPickerRef is a reference to the Endpoint Picker extension and its
	// associated configuration.
	//
	// +required
	EndpointPickerRef EndpointPickerRef `json:"endpointPickerRef,omitzero"`
}

// Port defines the network port that will be exposed by this InferencePool.
type Port struct {
	// Number defines the port number to access the selected model server Pods.
	// The number must be in the range 1 to 65535.
	//
	// +required
	Number PortNumber `json:"number,omitempty"`
}

// EndpointPickerRef specifies a reference to an Endpoint Picker extension and its
// associated configuration.
type EndpointPickerRef struct {
	// Group is the group of the referent API object. When unspecified, the default value
	// is "", representing the Core API group.
	//
	// +optional
	// +kubebuilder:default=""
	Group *Group `json:"group,omitempty"`

	// Kind is the Kubernetes resource kind of the referent.
	//
	// Required if the referent is ambiguous, e.g. service with multiple ports.
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
	//nolint:kubeapilinter // ignore kubeapilinter here as we want to use pointer for optional struct.
	Kind *Kind `json:"kind,omitempty"`

	// Name is the name of the referent API object.
	//
	// +required
	Name ObjectName `json:"name,omitempty"`

	// PortNumber is the port number of the Endpoint Picker extension service. When unspecified,
	// implementations SHOULD infer a default value of 9002 when the kind field is "Service" or
	// unspecified (defaults to "Service").
	//
	// +optional
	//nolint:kubeapilinter // ignore kubeapilinter here as we want to use pointer for optional struct.
	PortNumber *PortNumber `json:"portNumber,omitempty"`

	// FailureMode configures how the parent handles the case when the Endpoint Picker extension
	// is non-responsive. When unspecified, defaults to "FailClose".
	//
	// +optional
	// +kubebuilder:default="FailClose"
	//nolint:kubeapilinter // ignore kubeapilinter here as we want to use pointer for optional struct.
	FailureMode *EndpointPickerFailureMode `json:"failureMode,omitempty"`
}

// EndpointPickerFailureMode defines the options for how the parent handles the case when the
// Endpoint Picker extension is non-responsive.
//
// +kubebuilder:validation:Enum=FailOpen;FailClose
type EndpointPickerFailureMode string

const (
	// EndpointPickerFailOpen specifies that the parent should forward the request to an endpoint
	// of its picking when the Endpoint Picker extension fails.
	EndpointPickerFailOpen EndpointPickerFailureMode = "FailOpen"
	// EndpointPickerFailClose specifies that the parent should drop the request when the Endpoint
	// Picker extension fails.
	EndpointPickerFailClose EndpointPickerFailureMode = "FailClose"
)

// InferencePoolStatus defines the observed state of the InferencePool.
type InferencePoolStatus struct {
	// Parents is a list of parent resources, typically Gateways, that are associated with
	// the InferencePool, and the status of the InferencePool with respect to each parent.
	//
	// A controller that manages the InferencePool, must add an entry for each parent it manages
	// and remove the parent entry when the controller no longer considers the InferencePool to
	// be associated with that parent.
	//
	// A maximum of 32 parents will be represented in this list. When the list is empty,
	// it indicates that the InferencePool is not associated with any parents.
	//
	// +kubebuilder:validation:MaxItems=32
	// +optional
	// +listType=atomic
	Parents []ParentStatus `json:"parents,omitempty"`
}

// ParentStatus defines the observed state of InferencePool from a Parent, i.e. Gateway.
type ParentStatus struct {
	// Conditions is a list of status conditions that provide information about the observed
	// state of the InferencePool. This field is required to be set by the controller that
	// manages the InferencePool.
	//
	// Supported condition types are:
	//
	// * "Accepted"
	// * "ResolvedRefs"
	//
	// +optional
	// +listType=map
	// +listMapKey=type
	// +kubebuilder:validation:MaxItems=8
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// ParentRef is used to identify the parent resource that this status
	// is associated with. It is used to match the InferencePool with the parent
	// resource, such as a Gateway.
	//
	// +required
	ParentRef ParentReference `json:"parentRef,omitzero"`
}

// InferencePoolConditionType is a type of status condition for the InferencePool.
type InferencePoolConditionType string

// InferencePoolReason is the reason for a type of InferencePool status condition.
type InferencePoolReason string

const (
	// InferencePoolConditionAccepted is a type of condition that indicates whether
	// the InferencePool has been accepted or rejected by a Parent, and why.
	//
	// Possible reasons for this condition to be True are:
	//
	// * "SupportedByParent"
	//
	// Possible reasons for this condition to be False are:
	//
	// * "Accepted"
	// * "HTTPRouteNotAccepted"
	//
	// Possible reasons for this condition to be Unknown are:
	//
	// * "Pending"
	//
	// Controllers MAY raise this condition with other reasons, but should
	// prefer to use the reasons listed above to improve interoperability.
	InferencePoolConditionAccepted InferencePoolConditionType = "Accepted"

	// InferencePoolReasonAccepted is a reason used with the "Accepted" condition
	// when the InferencePool is accepted by a Parent because the Parent supports
	// InferencePool as a backend.
	InferencePoolReasonAccepted InferencePoolReason = "Accepted"

	// InferencePoolReasonNotSupportedByParent is a reason used with the "Accepted"
	// condition when the InferencePool has not been accepted by a Parent because
	// the Parent does not support InferencePool as a backend.
	InferencePoolReasonNotSupportedByParent InferencePoolReason = "NotSupportedByParent"

	// InferencePoolReasonHTTPRouteNotAccepted is an optional reason used with the
	// "Accepted" condition when the InferencePool is referenced by an HTTPRoute that
	// has been rejected by the Parent. The user should inspect the status of the
	// referring HTTPRoute for the specific reason.
	InferencePoolReasonHTTPRouteNotAccepted InferencePoolReason = "HTTPRouteNotAccepted"
)

const (
	// InferencePoolConditionResolvedRefs is a type of condition that indicates whether
	// the controller was able to resolve all the object references for the InferencePool.
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

	// InferencePoolReasonResolvedRefs is a reason used with the "ResolvedRefs"
	// condition when the condition is true.
	InferencePoolReasonResolvedRefs InferencePoolReason = "ResolvedRefs"

	// InferencePoolReasonInvalidExtensionRef is a reason used with the "ResolvedRefs"
	// condition when the Extension is invalid in some way. This can include an
	// unsupported kind or API group, or a reference to a resource that cannot be found.
	InferencePoolReasonInvalidExtensionRef InferencePoolReason = "InvalidExtensionRef"
)

// ParentReference identifies an API object. It is used to associate the InferencePool with a
// parent resource, such as a Gateway.
type ParentReference struct {
	// Group is the group of the referent API object. When unspecified, the referent is assumed
	// to be in the "gateway.networking.k8s.io" API group.
	//
	// +optional
	// +kubebuilder:default="gateway.networking.k8s.io"
	Group *Group `json:"group,omitempty"`

	// Kind is the kind of the referent API object. When unspecified, the referent is assumed
	// to be a "Gateway" kind.
	//
	// +optional
	// +kubebuilder:default=Gateway
	//nolint:kubeapilinter // ignore kubeapilinter here as we want to use pointer for optional struct.
	Kind *Kind `json:"kind,omitempty"`

	// Name is the name of the referent API object.
	//
	// +required
	Name ObjectName `json:"name,omitempty"`

	// Namespace is the namespace of the referenced object. When unspecified, the local
	// namespace is inferred.
	//
	// Note that when a namespace different than the local namespace is specified,
	// a ReferenceGrant object is required in the referent namespace to allow that
	// namespace's owner to accept the reference. See the ReferenceGrant
	// documentation for details: https://gateway-api.sigs.k8s.io/api-types/referencegrant/
	//
	// +optional
	//nolint:kubeapilinter // ignore kubeapilinter here as we want to use pointer for optional struct.
	Namespace *Namespace `json:"namespace,omitempty"`
}
