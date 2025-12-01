# Inference Pool Level Model Name Redirect and Traffic Splitting

**Status:** Proposed

For the full, detailed proposal, please see the [original proposal](https://docs.google.com/document/d/12yR_nAWM-Tg2ZmgGYX1h-dlUNi0AqYoACUjNElipl0M/edit?usp=sharing).

## Summary

The original `InferenceModel` API ([v1alpha2](https://github.com/capri-xiyue/gateway-api-inference-extension/blob/0189c333c2d4076f099fda81bc37f41996426fa9/apix/v1alpha2/inferencemodel_types.go)) provided initial support for model routing. This proposal reintroduces and enhances the core functionalities of model name redirection and traffic splitting within an inference pool. This is essential for modern use cases such as model name aliasing/versioning and granular traffic splitting for gradual rollouts.

### Goals
*   Enable configurable model redirection/aliasing within an `InferencePool`.
*   Enable traffic splitting within an `InferencePool`.

### Non-Goals
*   Traffic splitting between `InferencePools` will be handled by `HTTPRoute`.

## Proposal

We propose introducing a new Custom Resource Definition (CRD), `InferenceModelRewrite`, to define rules for model redirection and traffic splitting. The Endpoint Picker Proxy (EPP) will be responsible for watching these resources and performing the necessary request body mutations and traffic distribution.

This approach provides the cleanest separation of concerns:
*   **BBR / `HTTPRoute`:** Handles "global" (inter-pool) routing. It directs the user-facing model name to the correct logical `InferencePool`.
*   **EPP / `InferenceModelRewrite`:** Manages "local" (intra-pool) implementation details, such as routing to different model versions (`v1` vs. `v2`) within that pool.

For a more detailed discussion on the execution architecture (BBR vs. EPP) and naming conventions, please refer to the [original proposal document](https://docs.google.com/document/d/12yR_nAWM-Tg2ZmgGYX1h-dlUNi0AqYoACUjNElipl0M/edit?usp=sharing).

### CRD Specification

```go
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
	//If empty, the rule matches all requests.

	// +optional
	Matches []Match `json:"matches,omitempty"`

	// --- Actions ---
	// Targets defines how to distribute traffic across a set of
	// weighted model targets. This is used for traffic splitting, A/B tests,
	// or canary rollouts.
	// +optional
	// +kubebuilder:validation:MinItems=1
	//
	Targets []TargetModel `json:"split,omitempty"`
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
```

### Example: Traffic Splitting

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: InferenceModelRewrite
metadata:
  name: food-review-canary-rollout
spec:
  poolRef:
    name: main-food-review-pool
  rules:
  - matches:
    - model:
        type: Exact
        value: "foodreview"
    targets:
    - modelRewrite: "foodreview-v1"
      weight: 10
    - modelRewrite: "foodreview-v2"
      weight: 90
```

## Implementation Phases

*   **Phase 1: EPP-Driven Intra-Pool Rewrite:** This phase delivers the core functionality for the most common use case: model rewrite and traffic splitting within a single `InferencePool`.
    *   The `InferenceModelRewrite` CRD will be created.
    *   The EPP will be enhanced to be a read-only controller for this CRD, executing the request body mutation and traffic splitting.
    *   **Key Point:** The EPP will be a **read-only** consumer of the CRD and will not write to its status field. Status updates (e.g., marking conflicts) must be handled by a new, separate controller if needed.

*   **Phase 2 (Conditional): Promote Rewrite Logic to a Shared Library:** If user patterns show a strong need for inter-pool traffic splitting based on rewritten model names, the logic can be moved into a shared library used by both BBR and EPP.
    *   An idempotency mechanism, like an `X-Gateway-Model-Name` header, would be added to prevent the EPP from re-executing the logic.
