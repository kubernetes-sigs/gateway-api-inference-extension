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

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
)

// InferencePoolImport is the Schema for the InferencePoolImports API.
//
// +kubebuilder:object:root=true
// +kubebuilder:resource:shortName=infpimp
// +kubebuilder:subresource:status
// +kubebuilder:storageversion
// +genclient
type InferencePoolImport struct {
	metav1.TypeMeta `json:",inline"`

	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// Status defines the observed state of the InferencePoolImport.
	//
	// +optional
	//nolint:kubeapilinter // status should not be a pointer.
	Status InferencePoolImportStatus `json:"status,omitempty"`
}

// InferencePoolImportList contains a list of InferencePoolImports.
//
// +kubebuilder:object:root=true
type InferencePoolImportList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []InferencePoolImport `json:"items"`
}

// InferencePoolImportStatus defines the observed state of the InferencePoolImport.
type InferencePoolImportStatus struct {
	// Controllers is a list of controllers that are responsible for managing the InferencePoolImport.
	//
	// +listType=map
	// +listMapKey=name
	// +kubebuilder:validation:MaxItems=8
	// +kubebuilder:validation:Required
	Controllers []ImportController `json:"controllers"`
}

// ImportController defines a controller that is responsible for managing the InferencePoolImport.
type ImportController struct {
	// Name is a domain/path string that indicates the name of the controller that manages the
	// InferencePoolImport. Name corresponds to the GatewayClass controllerName field when the
	// controller will manage parents of type "Gateway". Otherwise, the name is implementation-specific.
	//
	// Example: "example.net/import-controller".
	//
	// The format of this field is DOMAIN "/" PATH, where DOMAIN and PATH are valid Kubernetes
	// names (https://kubernetes.io/docs/concepts/overview/working-with-objects/names/#names).
	//
	// A controller MUST populate this field when writing status and ensure that entries to status
	// populated with their controller name are removed when they are no longer necessary.
	//
	// +required
	Name ControllerName `json:"name"`

	// ExportingClusters is a list of clusters that exported the InferencePool(s) that back the
	// InferencePoolImport. Required when the controller is responsible for CRUD'ing the InferencePoolImport
	// from the exported InferencePool(s).
	//
	// +optional
	ExportingClusters []ExportingCluster `json:"exportingClusters,omitempty"`

	// Parents is a list of parent resources, typically Gateways, that are associated with the
	// InferencePoolImport, and the status of the InferencePoolImport with respect to each parent.
	//
	// Ancestor would be a more accurate name, but Parent is consistent with InferencePool terminology.
	//
	// Required when the controller manages the InferencePoolImport as an HTTPRoute backendRef. The controller
	// must add an entry for each parent it manages and remove the parent entry when the controller no longer
	// considers the InferencePoolImport to be associated with that parent.
	//
	// +optional
	// +listType=atomic
	Parents []v1.ParentStatus `json:"parents,omitempty"`

	// Conditions track the state of the InferencePoolImport.
	//
	// Known condition types are:
	//
	//  * "Accepted"
	//
	// +optional
	// +listType=map
	// +listMapKey=type
	// +kubebuilder:validation:MaxItems=8
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// ControllerName is the name of a controller that manages a resource. It must be a domain prefixed path.
//
// Valid values include:
//
//   - "example.com/bar"
//
// Invalid values include:
//
//   - "example.com" - must include path
//   - "foo.example.com" - must include path
//
// +kubebuilder:validation:MinLength=1
// +kubebuilder:validation:MaxLength=253
// +kubebuilder:validation:Pattern=`^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*\/[A-Za-z0-9\/\-._~%!$&'()*+,;=:]+$`
type ControllerName string

// ClusterName is the name of a cluster that exported the InferencePool.
//
// +kubebuilder:validation:MinLength=1
// +kubebuilder:validation:MaxLength=253
type ClusterName string

// ExportingCluster defines a cluster that exported the InferencePool that backs this InferencePoolImport.
type ExportingCluster struct {
	// Name of the exporting cluster (must be unique within the list).
	//
	// +kubebuilder:validation:Required
	Name ClusterName `json:"name"`
}
