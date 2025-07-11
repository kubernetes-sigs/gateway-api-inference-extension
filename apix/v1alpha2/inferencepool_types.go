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
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
)

// InferencePool is the Schema for the InferencePools API.
//
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:storageversion
// +genclient
type InferencePool v1.InferencePool

// InferencePoolList contains a list of InferencePool.
//
// +kubebuilder:object:root=true
type InferencePoolList v1.InferencePoolList

// InferencePoolSpec defines the desired state of InferencePool
type InferencePoolSpec = v1.InferencePoolSpec

// EndpointPickerConfig specifies the configuration needed by the proxy to discover and connect to the endpoint picker extension.
// This type is intended to be a union of mutually exclusive configuration options that we may add in the future.
type EndpointPickerConfig = v1.EndpointPickerConfig

// Extension specifies how to configure an extension that runs the endpoint picker.
type Extension = v1.Extension

// ExtensionReference is a reference to the extension.
//
// If a reference is invalid, the implementation MUST update the `ResolvedRefs`
// Condition on the InferencePool's status to `status: False`. A 5XX status code MUST be returned
// for the request that would have otherwise been routed to the invalid backend.
type ExtensionReference = v1.ExtensionReference

// ExtensionConnection encapsulates options that configures the connection to the extension.
type ExtensionConnection = v1.ExtensionConnection

// ExtensionFailureMode defines the options for how the gateway handles the case when the extension is not
// responsive.
// +kubebuilder:validation:Enum=FailOpen;FailClose
type ExtensionFailureMode = v1.ExtensionFailureMode

// InferencePoolStatus defines the observed state of InferencePool.
type InferencePoolStatus = v1.InferencePoolStatus

// PoolStatus defines the observed state of InferencePool from a Gateway.
type PoolStatus = v1.PoolStatus

// InferencePoolConditionType is a type of condition for the InferencePool
type InferencePoolConditionType = v1.InferencePoolConditionType

// InferencePoolReason is the reason for a given InferencePoolConditionType
type InferencePoolReason = v1.InferencePoolReason
