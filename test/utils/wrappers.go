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
	infextv1a1 "sigs.k8s.io/gateway-api-inference-extension/api/v1alpha1"
)

// InferenceModelWrapper wraps an InferenceModel.
type InferenceModelWrapper struct {
	infextv1a1.InferenceModel
}

// MakeModelWrapper creates a wrapper for an MakeModelWrapper.
func MakeModelWrapper(name, ns string) *InferenceModelWrapper {
	return &InferenceModelWrapper{
		infextv1a1.InferenceModel{
			ObjectMeta: metav1.ObjectMeta{
				Name:      name,
				Namespace: ns,
			},
			Spec: infextv1a1.InferenceModelSpec{
				ModelName: "",
				PoolRef:   infextv1a1.PoolObjectReference{},
			},
		},
	}
}

// SetModelName sets the value of the inferenceModel.spec.modelName.
func (m *InferenceModelWrapper) SetModelName(name string) *InferenceModelWrapper {
	m.Spec.ModelName = name
	return m
}

// SetCriticality sets the value of the inferenceModel.spec.criticality.
func (m *InferenceModelWrapper) SetCriticality(level infextv1a1.Criticality) *InferenceModelWrapper {
	m.Spec.Criticality = &level
	return m
}

// SetPoolRef sets the value of the inferenceModel.spec.poolRef using defaults
// for group/kind and name as the PoolObjectReference name.
func (m *InferenceModelWrapper) SetPoolRef(name string) *InferenceModelWrapper {
	ref := infextv1a1.PoolObjectReference{
		Group: infextv1a1.GroupVersion.Group,
		Kind:  "inferencepools",
		Name:  name,
	}
	m.Spec.PoolRef = ref
	return m
}

// SetTargetModels sets the value of the inferenceModel.spec.targetModels.
func (m *InferenceModelWrapper) SetTargetModels(models []infextv1a1.TargetModel) *InferenceModelWrapper {
	m.Spec.TargetModels = models
	return m
}

// Obj returns the inner InferenceModel.
func (m *InferenceModelWrapper) Obj() *infextv1a1.InferenceModel {
	return &m.InferenceModel
}
