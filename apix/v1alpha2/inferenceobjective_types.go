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

// InferenceObjective is the Schema for the InferenceObjectives API.
//
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:storageversion
// +kubebuilder:printcolumn:name="Inference Pool",type=string,JSONPath=`.spec.poolRef.name`
// +kubebuilder:printcolumn:name="Priority",type=string,JSONPath=`.spec.priority`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`
// +genclient
type InferenceObjective struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   InferenceObjectiveSpec   `json:"spec,omitempty"`
	Status InferenceObjectiveStatus `json:"status,omitempty"`
}

// InferenceObjectiveList contains a list of InferenceObjective.
//
// +kubebuilder:object:root=true
type InferenceObjectiveList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []InferenceObjective `json:"items"`
}

// InferenceObjectiveSpec represents the desired state of a specific model use case. This resource is
// managed by the "Inference Workload Owner" persona.
//
// The Inference Workload Owner persona is someone that trains, verifies, and
// leverages a large language model from a model frontend, drives the lifecycle
// and rollout of new versions of those models, and defines the specific
// performance and latency goals for the model. These workloads are
// expected to operate within an InferencePool sharing compute capacity with other
// InferenceObjectives, defined by the Inference Platform Admin.
type InferenceObjectiveSpec struct {

	// Priority defines how important it is to serve the request compared to other requests in the same pool.
	// Priority is an integer value that defines the priority of the request.
	// The higher the value, the more critical the request is; negative values _are_ allowed.
	// No default value is set for this field, allowing for future additions of new fields that may 'one of' with this field.
	// However, implementations that consume this field (such as the Endpoint Picker) will treat an unset value as '0'.
	// Priority is used in flow control, primarily in the event of resource scarcity(requests need to be queued).
	// All requests will be queued, and flow control will _always_ allow requests of higher priority to be served first.
	// Fairness is only enforced and tracked between requests of the same priority.
	//
	// Example: requests with Priority 10 will always be served before
	// requests with Priority of 0 (the value used if Priority is unset or no InfereneceObjective is specified).
	// Similarly requests with a Priority of -10 will always be served after requests with Priority of 0.
	// +optional
	Priority *int `json:"priority,omitempty"`

	// PoolRef is a reference to the inference pool, the pool must exist in the same namespace.
	//
	// +kubebuilder:validation:Required
	PoolRef PoolObjectReference `json:"poolRef"`
}

// InferenceObjectiveStatus defines the observed state of InferenceObjective
type InferenceObjectiveStatus struct {
	// Conditions track the state of the InferenceObjective.
	//
	// Known condition types are:
	//
	// * "Accepted"
	//
	// +optional
	// +listType=map
	// +listMapKey=type
	// +kubebuilder:validation:MaxItems=8
	// +kubebuilder:default={{type: "Ready", status: "Unknown", reason:"Pending", message:"Waiting for controller", lastTransitionTime: "1970-01-01T00:00:00Z"}}
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// InferenceObjectiveConditionType is a type of condition for the InferenceObjective.
type InferenceObjectiveConditionType string

// InferenceObjectiveConditionReason is the reason for a given InferenceObjectiveConditionType.
type InferenceObjectiveConditionReason string

const (
	// ObjectiveConditionAccepted indicates if the objective config is accepted, and if not, why.
	//
	// Possible reasons for this condition to be True are:
	//
	// * "Accepted"
	//
	// Possible reasons for this condition to be Unknown are:
	//
	// * "Pending"
	//
	ObjectiveConditionAccepted InferenceObjectiveConditionType = "Accepted"

	// ObjectiveReasonAccepted is the desired state. Model conforms to the state of the pool.
	ObjectiveReasonAccepted InferenceObjectiveConditionReason = "Accepted"

	// ObjectiveReasonPending is the initial state, and indicates that the controller has not yet reconciled the InferenceObjective.
	ObjectiveReasonPending InferenceObjectiveConditionReason = "Pending"
)
