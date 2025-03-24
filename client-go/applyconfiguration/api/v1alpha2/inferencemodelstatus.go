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
// Code generated by applyconfiguration-gen. DO NOT EDIT.

package v1alpha2

import (
	v1 "k8s.io/client-go/applyconfigurations/meta/v1"
)

// InferenceModelStatusApplyConfiguration represents a declarative configuration of the InferenceModelStatus type for use
// with apply.
type InferenceModelStatusApplyConfiguration struct {
	Conditions []v1.ConditionApplyConfiguration `json:"conditions,omitempty"`
}

// InferenceModelStatusApplyConfiguration constructs a declarative configuration of the InferenceModelStatus type for use with
// apply.
func InferenceModelStatus() *InferenceModelStatusApplyConfiguration {
	return &InferenceModelStatusApplyConfiguration{}
}

// WithConditions adds the given value to the Conditions field in the declarative configuration
// and returns the receiver, so that objects can be build by chaining "With" function invocations.
// If called multiple times, values provided by each call will be appended to the Conditions field.
func (b *InferenceModelStatusApplyConfiguration) WithConditions(values ...*v1.ConditionApplyConfiguration) *InferenceModelStatusApplyConfiguration {
	for i := range values {
		if values[i] == nil {
			panic("nil value passed to WithConditions")
		}
		b.Conditions = append(b.Conditions, *values[i])
	}
	return b
}
