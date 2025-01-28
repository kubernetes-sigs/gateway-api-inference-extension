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
// Code generated by applyconfiguration-gen. DO NOT EDIT.

package v1alpha1

import (
	apiv1alpha1 "inference.networking.x-k8s.io/gateway-api-inference-extension/api/v1alpha1"
)

// ExtensionConfigApplyConfiguration represents a declarative configuration of the ExtensionConfig type for use
// with apply.
type ExtensionConfigApplyConfiguration struct {
	ExtensionRef                           *ExtensionReferenceApplyConfiguration `json:"extensionRef,omitempty"`
	*ExtensionConnectionApplyConfiguration `json:"extensionConnection,omitempty"`
}

// ExtensionConfigApplyConfiguration constructs a declarative configuration of the ExtensionConfig type for use with
// apply.
func ExtensionConfig() *ExtensionConfigApplyConfiguration {
	return &ExtensionConfigApplyConfiguration{}
}

// WithExtensionRef sets the ExtensionRef field in the declarative configuration to the given value
// and returns the receiver, so that objects can be built by chaining "With" function invocations.
// If called multiple times, the ExtensionRef field is set to the value of the last call.
func (b *ExtensionConfigApplyConfiguration) WithExtensionRef(value *ExtensionReferenceApplyConfiguration) *ExtensionConfigApplyConfiguration {
	b.ExtensionRef = value
	return b
}

// WithTargetPortNumber sets the TargetPortNumber field in the declarative configuration to the given value
// and returns the receiver, so that objects can be built by chaining "With" function invocations.
// If called multiple times, the TargetPortNumber field is set to the value of the last call.
func (b *ExtensionConfigApplyConfiguration) WithTargetPortNumber(value int32) *ExtensionConfigApplyConfiguration {
	b.ensureExtensionConnectionApplyConfigurationExists()
	b.ExtensionConnectionApplyConfiguration.TargetPortNumber = &value
	return b
}

// WithFailureMode sets the FailureMode field in the declarative configuration to the given value
// and returns the receiver, so that objects can be built by chaining "With" function invocations.
// If called multiple times, the FailureMode field is set to the value of the last call.
func (b *ExtensionConfigApplyConfiguration) WithFailureMode(value apiv1alpha1.ExtensionFailureMode) *ExtensionConfigApplyConfiguration {
	b.ensureExtensionConnectionApplyConfigurationExists()
	b.ExtensionConnectionApplyConfiguration.FailureMode = &value
	return b
}

func (b *ExtensionConfigApplyConfiguration) ensureExtensionConnectionApplyConfigurationExists() {
	if b.ExtensionConnectionApplyConfiguration == nil {
		b.ExtensionConnectionApplyConfiguration = &ExtensionConnectionApplyConfiguration{}
	}
}
