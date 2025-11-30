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

// InferenceModelRewrite is the Schema for the InferenceModelRewrite API.
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Inference Pool",type=string,JSONPath=`.spec.poolRef.name`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`
// +genclient
type InferenceModelRewrite struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   InferenceModelRewriteSpec   `json:"spec,omitempty"`
	Status InferenceModelRewriteStatus `json:"status,omitempty"`
}

// InferenceModelRewriteList contains a list of InferenceModelRewrite.
//
// +kubebuilder:object:root=true
type InferenceModelRewriteList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []InferenceModelRewrite `json:"items"`
}

// InferenceModelRewriteSpec defines the desired state of InferenceModelRewrite.
type InferenceModelRewriteSpec struct {
	// PoolRef is a reference to the inference pool.
	// +kubebuilder:validation:Required
	PoolRef *PoolObjectReference `json:"poolRef"`

	// Rules are the ordered set of rules for rewriting inference requests.
	// The first rule to match a request will be used.

	//
	// --- Precedence and Conflict Resolution ---
	// If multiple InferenceModelRewrite resources target the same
	// InferencePool, the controller will merge them based on precedence.
	//
	// Across all rules specified on applicable rewrites, precedence MUST be
	// given to the match having an "Exact" model match over a generic match
	// (a rule with an empty `matches` array).
	//
	// If ties still exist across multiple InferenceModelRewrite resources (e.g.
	// two rewrites both have an exact match for the same model), matching
	// precedence MUST be determined by the oldest resource based on
	// creation timestamp.
	//
	// If ties still exist within a single InferenceModelRewrite resource, the
	// FIRST matching rule (in list order) is used.
	// +required
	Rules []InferenceModelRewriteRule `json:"rules"`
}

// InferenceModelRewriteRule defines the match criteria and corresponding action.
// For details on how precedence is determined across multiple rules and
// InferenceModelRewrite resources, see the "Precedence and Conflict Resolution"
// section in InferenceModelRewriteSpec.
type InferenceModelRewriteRule struct {
	// Matches defines the criteria for matching a request.
	// If multiple match criteria are specified, a request matches if
	// ANY of the criteria are satisfied (logical OR).
	// If empty, the rule matches all requests.

	// +optional
	Matches []Match `json:"matches,omitempty"`

	// --- Actions ---
	// Targets defines how to distribute traffic across a set of
	// weighted model targets. This is used for traffic splitting, A/B tests,
	// or canary rollouts.
	// +optional
	// +kubebuilder:validation:MinItems=1
	//
	Targets []TargetModel `json:"targets,omitempty"`
}

// TargetModel defines a weighted model destination for traffic distribution.
type TargetModel struct {
	// (The following comment is copied from the original targetModel)
	// Weight is used to determine the proportion of traffic that should be
	// sent to this model when multiple target models are specified.
	//
	// Weight defines the proportion of requests forwarded to the specified
	// model. This is computed as weight/(sum of all weights in this
	// TargetModels list). For non-zero values, there may be some epsilon from
	// the exact proportion defined here depending on the precision an
	// implementation supports. Weight is not a percentage and the sum of
	// weights does not need to equal 100.
	//
	// If a weight is set for any targetModel, it must be set for all targetModels.
	// Conversely weights are optional, so long as ALL targetModels do not specify a weight.
	//
	// +optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=1000000
	Weight int32 `json:"weight"`

	// --- Destination Types ---
	// ModelRewrite specifies a static model name destination.
	// +required
	ModelRewrite string `json:"modelRewrite"`
}

// Match defines the criteria for matching the LLM requests.
type Match struct {
	// Model specifies the criteria for matching the 'model' field
	// within the JSON request body.
	// +required
	Model *ModelMatch `json:"model,omitempty"`
}

// ModelMatch defines how to match against the model name in the request body.
type ModelMatch struct {
	// Type specifies the kind of string matching to use.
	// Supported value is "Exact". Defaults to "Exact".
	// +optional
	// +kubebuilder:default=Exact
	Type *MatchValidationType `json:"type,omitempty"`

	// Value is the model name string to match against.
	// +required
	// +kubebuilder:validation:MinLength=1
	Value string `json:"value"`
}

// MatchValidationType specifies the type of string matching to use.
// +kubebuilder:validation:Enum=Exact
type MatchValidationType string

const (
	// MatchExact indicates that the model name must match exactly.
	MatchExact MatchValidationType = "Exact"
)

// InferenceModelRewriteStatus defines the observed state of InferenceModelRewrite.
type InferenceModelRewriteStatus struct {
	// Conditions track the state of the InferenceModelRewrite.
	//
	// Known condition types are:
	//
	// * "Accepted"
	//
	// +optional
	// +listType=map
	// +listMapKey=type
	// +kubebuilder:validation:MaxItems=8
	// +kubebuilder:default={{type: "Accepted", status: "Unknown", reason:"Pending", message:"Waiting for controller", lastTransitionTime: "1970-01-01T00:00:00Z"}}
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// InferenceModelRewriteConditionType is a type of condition for the InferenceModelRewrite.
type InferenceModelRewriteConditionType string

// InferenceModelRewriteConditionReason is the reason for a given InferenceModelRewriteConditionType.
type InferenceModelRewriteConditionReason string

const (
	// RewriteConditionAccepted indicates if the rewrite is accepted, and if not, why.
	// This is the primary condition for this resource.
	//
	// Possible reasons for this condition to be True are:
	//
	// * "Accepted"
	//
	// Possible reasons for this condition to be Unknown are:
	//
	// * "Pending"
	//
	RewriteConditionAccepted InferenceModelRewriteConditionType = "Accepted"

	// RewriteReasonAccepted indicates the rewrite is valid, non-conflicting,
	// and has been successfully applied to the inference pool.
	RewriteReasonAccepted InferenceModelRewriteConditionReason = "Accepted"

	// RewriteReasonPending is the initial state, and indicates that the
	// controller has not yet reconciled the InferenceModelRewrite.
	RewriteReasonPending InferenceModelRewriteConditionReason = "Pending"
)
