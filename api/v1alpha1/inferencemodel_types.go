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

// InferenceModel is the Schema for the InferenceModels API.
// The InferenceModel is intended to represent a model workload (also referred to as a model use case) within Kubernetes.
// The management of the model server is not done by the InferenceModel. Instead, the
// focus of the InferenceModel is to provide the tools needed to effectively manage multiple models
// that share the same base model (currently the focus is LoRA adapters). Fields such as TargetModel
// are intended to simplify A/B testing and version rollout of adapters. While Criticality assists with
// governance of multiplexing many usecases over shared hardware.
//
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="ModelName",type=string,JSONPath=`.spec.modelName`
// +kubebuilder:printcolumn:name="Accepted",type=string,JSONPath=`.status.conditions[?(@.type=="Accepted")].status`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`
// +genclient
type InferenceModel struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   InferenceModelSpec   `json:"spec,omitempty"`
	Status InferenceModelStatus `json:"status,omitempty"`
}

// InferenceModelList contains a list of InferenceModel.
//
// +kubebuilder:object:root=true
type InferenceModelList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []InferenceModel `json:"items"`
}

// InferenceModelSpec represents the desired state of an InferenceModel. This resource is
// managed by the "Inference Workload Owner" persona.
//
// The Inference Workload Owner persona is someone that trains, verifies, and
// leverages a large language model focusing on model fidelity performance, and
// less on inference performance (which is managed by the Inference Platform Admin).
// They also drive the lifecycle and rollout of new versions of those models, and defines the specific
// performance and latency goals for the model. These workloads are
// expected to operate within an InferencePool sharing compute capacity with other
// InferenceModels, with specific governance defined by the Inference Platform Admin.
type InferenceModelSpec struct {
	// ModelName is the name of the model as the users set in the "model" parameter in the requests.
	// The name should be unique among the workloads that reference the same backend pool.
	// This is the parameter that will be used to match the request with.
	// Names can be reserved without implementing an actual model in the pool.
	// This can be done by specifying a target model and setting the weight to zero,
	// an error will be returned specifying that no valid target model is found.
	//
	// +kubebuilder:validation:MaxLength=256
	// +kubebuilder:validation:Required
	ModelName string `json:"modelName"`

	// Criticality defines how important it is to serve the model compared to other models referencing the same pool.
	// TODO: Update field upon resolution of: https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/213
	//
	// Default values for this field will not be set, to allow for future additions of new field that may 'one of' with this field.
	// Any implementations that may consume this field may treat an unset value as the 'Standard' range.
	// +optional
	Criticality *Criticality `json:"criticality,omitempty"`

	// TargetModels allow multiple versions of a model for traffic splitting.
	// Traffic splitting is handled via weights. The targetModel field is optional, however,
	// if not specified, the target model name is defaulted to the modelName parameter.
	// modelName is often in reference to a LoRA adapter.
	//
	// Examples:
	// - A model server serving `llama2-7b` may be represented by:
	//   - setting the modelName to `llama2-7b` and setting no targetModels
	//   - setting the modelName to `hello-world` and setting a single targetModel to `llama2-7b`, and setting no weights
	//   - setting modelName to 'my-fine-tune', setting 2 targetModels 'fine-tune-v1' & 'fine-tune-v2', and setting no weights.
	//       This has the effect of weighing the two models equally
	//   - setting modelName to 'my-fine-tune', setting 2 targetModels 'fine-tune-v1' w/weight: 10 & 'fine-tune-v2' w/weight: 1.
	//       This has the effect of the fine-tune-v1 being selected 10x as often as v2
	//
	// +optional
	// +kubebuilder:validation:MaxItems=10
	// +kubebuilder:validation:XValidation:message="Weights should be set for all models, or none of the models.",rule="self.all(model, has(model.weight)) || self.all(model, !has(model.weight))"
	TargetModels []TargetModel `json:"targetModels,omitempty"`

	// PoolRef is a reference to the inference pool, the pool must exist in the same namespace.
	//
	// +kubebuilder:validation:Required
	PoolRef PoolObjectReference `json:"poolRef"`
}

// PoolObjectReference identifies an API object within the namespace of the
// referrer.
type PoolObjectReference struct {
	// Group is the group of the referent.
	//
	// +optional
	// +kubebuilder:default="inference.networking.x-k8s.io"
	// +kubebuilder:validation:MaxLength=253
	// +kubebuilder:validation:Pattern=`^$|^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$`
	Group string `json:"group,omitempty"`

	// Kind is kind of the referent. For example "InferencePool".
	//
	// +optional
	// +kubebuilder:default="InferencePool"
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=63
	// +kubebuilder:validation:Pattern=`^[a-zA-Z]([-a-zA-Z0-9]*[a-zA-Z0-9])?$`
	Kind string `json:"kind,omitempty"`

	// Name is the name of the referent.
	//
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=253
	// +kubebuilder:validation:Required
	Name string `json:"name"`
}

// Criticality defines how important it is to serve the model compared to other models.
// Criticality is intentionally a bounded enum to contain the possibilities that need to be supported by the load balancing algorithm. Any reference to the Criticality field must be optional(use a pointer), and set no default.
// This allows us to union this with a oneOf field in the future should we wish to adjust/extend this behavior.
// +kubebuilder:validation:Enum=Critical;Standard;Sheddable
type Criticality string

const (
	// Critical defines the highest level of criticality. Requests to this band will be shed last.
	Critical Criticality = "Critical"

	// Standard defines the base criticality level and is more important than Sheddable but less
	// important than Critical. Requests in this band will be shed before critical traffic.
	// Most models are expected to fall within this band.
	Standard Criticality = "Standard"

	// Sheddable defines the lowest level of criticality. Requests to this band will be shed before
	// all other bands.
	Sheddable Criticality = "Sheddable"
)

// TargetModel represents a deployed model or a LoRA adapter. The
// Name field is expected to match the name of the LoRA adapter
// (or base model) as it is registered within the model server. Inference
// Gateway assumes that the model exists on the model server and it's the
// responsibility of the user to validate a correct match. Should a model fail
// to exist at request time, the error is processed by the Inference Gateway
// and emitted on the appropriate InferenceModel object.
type TargetModel struct {
	// Name is the name of the LoRA adapter or base model, as expected by the ModelServer.
	//
	// +kubebuilder:validation:MaxLength=253
	// +kubebuilder:validation:Required
	Name string `json:"name"`

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
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1000000
	Weight *int32 `json:"weight,omitempty"`
}

// InferenceModelStatus defines the observed state of InferenceModel
type InferenceModelStatus struct {
	// Conditions track the state of the InferenceModel.
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

// InferenceModelConditionType is a type of condition for the InferenceModel.
type InferenceModelConditionType string

// InferenceModelConditionReason is the reason for a given InferenceModelConditionType.
type InferenceModelConditionReason string

const (
	// This condition indicates if the model config is accepted, and if not, why.
	//
	// Possible reasons for this condition to be True are:
	//
	// * "Accepted"
	//
	// Possible reasons for this condition to be False are:
	//
	// * "ModelNameInUse"
	//
	// Possible reasons for this condition to be Unknown are:
	//
	// * "Pending"
	//
	ModelConditionAccepted InferenceModelConditionType = "Accepted"

	// Desired state. Model conforms to the state of the pool.
	ModelReasonAccepted InferenceModelConditionReason = "Accepted"

	// This reason is used when a given ModelName already exists within the pool.
	// Details about naming conflict resolution are on the ModelName field itself.
	ModelReasonNameInUse InferenceModelConditionReason = "ModelNameInUse"

	// This reason is the initial state, and indicates that the controller has not yet reconciled the InferenceModel.
	ModelReasonPending InferenceModelConditionReason = "Pending"
)

func init() {
	SchemeBuilder.Register(&InferenceModel{}, &InferenceModelList{})
}
