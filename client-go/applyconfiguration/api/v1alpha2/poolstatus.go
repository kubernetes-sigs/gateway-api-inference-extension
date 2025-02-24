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

package v1alpha2

import (
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/client-go/applyconfigurations/meta/v1"
)

// PoolStatusApplyConfiguration represents a declarative configuration of the PoolStatus type for use
// with apply.
type PoolStatusApplyConfiguration struct {
	GatewayRef *v1.ObjectReference                  `json:"parentRef,omitempty"`
	Conditions []metav1.ConditionApplyConfiguration `json:"conditions,omitempty"`
}

// PoolStatusApplyConfiguration constructs a declarative configuration of the PoolStatus type for use with
// apply.
func PoolStatus() *PoolStatusApplyConfiguration {
	return &PoolStatusApplyConfiguration{}
}

// WithGatewayRef sets the GatewayRef field in the declarative configuration to the given value
// and returns the receiver, so that objects can be built by chaining "With" function invocations.
// If called multiple times, the GatewayRef field is set to the value of the last call.
func (b *PoolStatusApplyConfiguration) WithGatewayRef(value v1.ObjectReference) *PoolStatusApplyConfiguration {
	b.GatewayRef = &value
	return b
}

// WithConditions adds the given value to the Conditions field in the declarative configuration
// and returns the receiver, so that objects can be build by chaining "With" function invocations.
// If called multiple times, values provided by each call will be appended to the Conditions field.
func (b *PoolStatusApplyConfiguration) WithConditions(values ...*metav1.ConditionApplyConfiguration) *PoolStatusApplyConfiguration {
	for i := range values {
		if values[i] == nil {
			panic("nil value passed to WithConditions")
		}
		b.Conditions = append(b.Conditions, *values[i])
	}
	return b
}
