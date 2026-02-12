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

// Group refers to a Kubernetes Group. It must either be an empty string or a
// RFC 1123 subdomain.
//
// This validation is based off of the corresponding Kubernetes validation:
// https://github.com/kubernetes/apimachinery/blob/02cfb53916346d085a6c6c7c66f882e3c6b0eca6/pkg/util/validation/validation.go#L208
//
// Valid values include:
//
// * "" - empty string implies core Kubernetes API group
// * "gateway.networking.k8s.io"
// * "foo.example.com"
//
// Invalid values include:
//
// * "example.com/bar" - "/" is an invalid character
//
// +kubebuilder:validation:MaxLength=253
// +kubebuilder:validation:Pattern=`^$|^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$`
type Group string

// Kind refers to a Kubernetes Kind.
//
// Valid values include:
//
// * "Service"
// * "HTTPRoute"
//
// Invalid values include:
//
// * "invalid/kind" - "/" is an invalid character
//
// +kubebuilder:validation:MinLength=1
// +kubebuilder:validation:MaxLength=63
// +kubebuilder:validation:Pattern=`^[a-zA-Z]([-a-zA-Z0-9]*[a-zA-Z0-9])?$`
type Kind string

// ObjectName refers to the name of a Kubernetes object.
// Object names can have a variety of forms, including RFC 1123 subdomains,
// RFC 1123 labels, or RFC 1035 labels.
//
// +kubebuilder:validation:MinLength=1
// +kubebuilder:validation:MaxLength=253
type ObjectName string

// Namespace refers to a Kubernetes namespace. It must be a RFC 1123 label.
//
// This validation is based off of the corresponding Kubernetes validation:
// https://github.com/kubernetes/apimachinery/blob/02cfb53916346d085a6c6c7c66f882e3c6b0eca6/pkg/util/validation/validation.go#L187
//
// This is used for Namespace name validation here:
// https://github.com/kubernetes/apimachinery/blob/02cfb53916346d085a6c6c7c66f882e3c6b0eca6/pkg/api/validation/generic.go#L63
//
// Valid values include:
//
// * "example"
//
// Invalid values include:
//
// * "example.com" - "." is an invalid character
//
// +kubebuilder:validation:Pattern=`^[a-z0-9]([-a-z0-9]*[a-z0-9])?$`
// +kubebuilder:validation:MinLength=1
// +kubebuilder:validation:MaxLength=63
type Namespace string

// PortNumber defines a network port.
//
// +kubebuilder:validation:Minimum=1
// +kubebuilder:validation:Maximum=65535
type PortNumber int32

// LabelKey was originally copied from: https://github.com/kubernetes-sigs/gateway-api/blob/99a3934c6bc1ce0874f3a4c5f20cafd8977ffcb4/apis/v1/shared_types.go#L694-L731
// Duplicated as to not take an unexpected dependency on gw's API.
//
// LabelKey is the key of a label. This is used for validation
// of maps. This matches the Kubernetes "qualified name" validation that is used for labels.
// Labels are case sensitive, so: my-label and My-Label are considered distinct.
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

// LabelSelectorOperator is a set of operators for label selector requirements.
//
// +kubebuilder:validation:Enum=In;NotIn;Exists;DoesNotExist
type LabelSelectorOperator string

const (
	// LabelSelectorOpIn matches labels where the key exists and the value is in the specified set.
	LabelSelectorOpIn LabelSelectorOperator = "In"
	// LabelSelectorOpNotIn matches labels where the key does not exist or the value is not in the specified set.
	LabelSelectorOpNotIn LabelSelectorOperator = "NotIn"
	// LabelSelectorOpExists matches labels where the key exists, regardless of value.
	LabelSelectorOpExists LabelSelectorOperator = "Exists"
	// LabelSelectorOpDoesNotExist matches labels where the key does not exist.
	LabelSelectorOpDoesNotExist LabelSelectorOperator = "DoesNotExist"
)

// LabelSelectorRequirement is a selector requirement that contains values, a key, and an operator
// that relates the key and values.
type LabelSelectorRequirement struct {
	// Key is the label key that the selector applies to.
	//
	// +kubebuilder:validation:Required
	Key LabelKey `json:"key"`

	// Operator represents a key's relationship to a set of values.
	// Valid operators are In, NotIn, Exists and DoesNotExist.
	//
	// +kubebuilder:validation:Required
	Operator LabelSelectorOperator `json:"operator"`

	// Values is an array of string values. If the operator is In or NotIn,
	// the values array must be non-empty. If the operator is Exists or DoesNotExist,
	// the values array must be empty.
	//
	// +optional
	// +kubebuilder:validation:MaxItems=64
	Values []LabelValue `json:"values,omitempty"`
}

// PoolSelector selects InferencePools based on labels and optionally filters by group/kind.
// Only pools within the same namespace as the referencing resource are considered.
type PoolSelector struct {
	// Group constrains the selector to only match pools of a specific API group.
	// If not specified, defaults to "inference.networking.k8s.io".
	//
	// +optional
	// +kubebuilder:default="inference.networking.k8s.io"
	Group Group `json:"group,omitempty"`

	// Kind constrains the selector to only match pools of a specific kind.
	// If not specified, defaults to "InferencePool".
	//
	// +optional
	// +kubebuilder:default="InferencePool"
	Kind Kind `json:"kind,omitempty"`

	// MatchLabels is a map of {key,value} pairs. A pool is selected if all
	// labels match. The matching is performed using an AND operation.
	//
	// +optional
	// +kubebuilder:validation:MaxProperties=64
	MatchLabels map[LabelKey]LabelValue `json:"matchLabels,omitempty"`

	// MatchExpressions is a list of label selector requirements.
	// The requirements are ANDed together.
	//
	// +optional
	// +kubebuilder:validation:MaxItems=16
	MatchExpressions []LabelSelectorRequirement `json:"matchExpressions,omitempty"`
}

// PoolObjectReference identifies an API object within the namespace of the
// referrer.
type PoolObjectReference struct {
	// Group is the group of the referent.
	//
	// +optional
	// +kubebuilder:default="inference.networking.k8s.io"
	Group Group `json:"group,omitempty"`

	// Kind is kind of the referent. For example "InferencePool".
	//
	// +optional
	// +kubebuilder:default="InferencePool"
	Kind Kind `json:"kind,omitempty"`

	// Name is the name of the referent.
	//
	// +kubebuilder:validation:Required
	Name ObjectName `json:"name"`
}
