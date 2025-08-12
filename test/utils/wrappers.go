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

package utils

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
)

// InferenceObjectiveWrapper wraps an InferenceObjective.
type InferenceObjectiveWrapper struct {
	v1alpha2.InferenceObjective
}

// MakeModelWrapper creates a wrapper for an MakeModelWrapper.
func MakeModelWrapper(namespacedName types.NamespacedName) *InferenceObjectiveWrapper {
	return &InferenceObjectiveWrapper{
		v1alpha2.InferenceObjective{
			ObjectMeta: metav1.ObjectMeta{
				Name:      namespacedName.Name,
				Namespace: namespacedName.Namespace,
			},
			Spec: v1alpha2.InferenceObjectiveSpec{
				PoolRef: v1alpha2.PoolObjectReference{},
			},
		},
	}
}

// SetPriority sets the value of the InferenceObjective.spec.priority.
func (m *InferenceObjectiveWrapper) SetPriority(level int) *InferenceObjectiveWrapper {
	m.Spec.Priority = &level
	return m
}

// SetPoolRef sets the value of the InferenceObjective.spec.poolRef using defaults
// for group/kind and name as the PoolObjectReference name.
func (m *InferenceObjectiveWrapper) SetPoolRef(name string) *InferenceObjectiveWrapper {
	ref := v1alpha2.PoolObjectReference{
		Group: v1alpha2.Group(v1.GroupVersion.Group),
		Kind:  "inferencepools",
		Name:  v1alpha2.ObjectName(name),
	}
	m.Spec.PoolRef = ref
	return m
}

// Obj returns the inner InferenceObjective.
func (m *InferenceObjectiveWrapper) Obj() *v1alpha2.InferenceObjective {
	return &m.InferenceObjective
}
