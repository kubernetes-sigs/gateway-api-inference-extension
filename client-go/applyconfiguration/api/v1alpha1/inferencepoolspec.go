/*
Copyright 2024.

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
	v1alpha1 "inference.networking.x-k8s.io/llm-instance-gateway/api/v1alpha1"
)

// InferencePoolSpecApplyConfiguration represents a declarative configuration of the InferencePoolSpec type for use
// with apply.
type InferencePoolSpecApplyConfiguration struct {
	Selector         map[v1alpha1.LabelKey]v1alpha1.LabelValue `json:"selector,omitempty"`
	TargetPortNumber *int32                                    `json:"targetPortNumber,omitempty"`
}

// InferencePoolSpecApplyConfiguration constructs a declarative configuration of the InferencePoolSpec type for use with
// apply.
func InferencePoolSpec() *InferencePoolSpecApplyConfiguration {
	return &InferencePoolSpecApplyConfiguration{}
}

// WithSelector puts the entries into the Selector field in the declarative configuration
// and returns the receiver, so that objects can be build by chaining "With" function invocations.
// If called multiple times, the entries provided by each call will be put on the Selector field,
// overwriting an existing map entries in Selector field with the same key.
func (b *InferencePoolSpecApplyConfiguration) WithSelector(entries map[v1alpha1.LabelKey]v1alpha1.LabelValue) *InferencePoolSpecApplyConfiguration {
	if b.Selector == nil && len(entries) > 0 {
		b.Selector = make(map[v1alpha1.LabelKey]v1alpha1.LabelValue, len(entries))
	}
	for k, v := range entries {
		b.Selector[k] = v
	}
	return b
}

// WithTargetPortNumber sets the TargetPortNumber field in the declarative configuration to the given value
// and returns the receiver, so that objects can be built by chaining "With" function invocations.
// If called multiple times, the TargetPortNumber field is set to the value of the last call.
func (b *InferencePoolSpecApplyConfiguration) WithTargetPortNumber(value int32) *InferencePoolSpecApplyConfiguration {
	b.TargetPortNumber = &value
	return b
}
