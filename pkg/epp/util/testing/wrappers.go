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

package testing

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
)

// PodWrapper wraps a Pod.
type PodWrapper struct {
	corev1.Pod
}

func FromBase(pod *corev1.Pod) *PodWrapper {
	return &PodWrapper{
		Pod: *pod,
	}
}

// MakePod creates a wrapper for a Pod.
func MakePod(podName string) *PodWrapper {
	return &PodWrapper{
		corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec:   corev1.PodSpec{},
			Status: corev1.PodStatus{},
		},
	}
}

// Complete sets necessary fields for a Pod to make it not denied by the apiserver
func (p *PodWrapper) Complete() *PodWrapper {
	if p.Pod.Namespace == "" {
		p.Namespace("default")
	}
	p.Spec.Containers = []corev1.Container{
		{
			Name:  "mock-vllm",
			Image: "mock-vllm:latest",
		},
	}
	return p
}

func (p *PodWrapper) Namespace(ns string) *PodWrapper {
	p.ObjectMeta.Namespace = ns
	return p
}

// Labels sets the pod labels.
func (p *PodWrapper) Labels(labels map[string]string) *PodWrapper {
	p.ObjectMeta.Labels = labels
	return p
}

// SetReadyCondition sets a PodReay=true condition.
func (p *PodWrapper) ReadyCondition() *PodWrapper {
	p.Status.Conditions = []corev1.PodCondition{{
		Type:   corev1.PodReady,
		Status: corev1.ConditionTrue,
	}}
	return p
}

func (p *PodWrapper) IP(ip string) *PodWrapper {
	p.Status.PodIP = ip
	return p
}

func (p *PodWrapper) DeletionTimestamp() *PodWrapper {
	now := metav1.Now()
	p.ObjectMeta.DeletionTimestamp = &now
	p.ObjectMeta.Finalizers = []string{"finalizer"}
	return p
}

// Obj returns the wrapped Pod.
func (p *PodWrapper) ObjRef() *corev1.Pod {
	return &p.Pod
}

// InferenceModelWrapper wraps an InferenceModel.
type InferenceModelWrapper struct {
	v1alpha2.InferenceModel
}

// MakeInferenceModel creates a wrapper for a InferenceModel.
func MakeInferenceModel(name string) *InferenceModelWrapper {
	return &InferenceModelWrapper{
		v1alpha2.InferenceModel{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
			Spec: v1alpha2.InferenceModelSpec{},
		},
	}
}

func (m *InferenceModelWrapper) Namespace(ns string) *InferenceModelWrapper {
	m.ObjectMeta.Namespace = ns
	return m
}

// Obj returns the wrapped InferenceModel.
func (m *InferenceModelWrapper) ObjRef() *v1alpha2.InferenceModel {
	return &m.InferenceModel
}

func (m *InferenceModelWrapper) ModelName(modelName string) *InferenceModelWrapper {
	m.Spec.ModelName = modelName
	return m
}

func (m *InferenceModelWrapper) PoolName(poolName string) *InferenceModelWrapper {
	m.Spec.PoolRef = v1alpha2.PoolObjectReference{Name: v1alpha2.ObjectName(poolName)}
	return m
}

func (m *InferenceModelWrapper) Criticality(criticality v1alpha2.Criticality) *InferenceModelWrapper {
	m.Spec.Criticality = &criticality
	return m
}

func (m *InferenceModelWrapper) DeletionTimestamp() *InferenceModelWrapper {
	now := metav1.Now()
	m.ObjectMeta.DeletionTimestamp = &now
	m.ObjectMeta.Finalizers = []string{"finalizer"}
	return m
}

func (m *InferenceModelWrapper) CreationTimestamp(t metav1.Time) *InferenceModelWrapper {
	m.ObjectMeta.CreationTimestamp = t
	return m
}

// InferencePoolWrapper wraps an InferencePool.
type InferencePoolWrapper struct {
	v1alpha2.InferencePool
}

// MakeInferencePool creates a wrapper for a InferencePool.
func MakeInferencePool(name string) *InferencePoolWrapper {
	return &InferencePoolWrapper{
		v1alpha2.InferencePool{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
			Spec: v1alpha2.InferencePoolSpec{},
		},
	}
}

func (m *InferencePoolWrapper) Namespace(ns string) *InferencePoolWrapper {
	m.ObjectMeta.Namespace = ns
	return m
}

func (m *InferencePoolWrapper) Selector(selector map[string]string) *InferencePoolWrapper {
	s := make(map[v1alpha2.LabelKey]v1alpha2.LabelValue)
	for k, v := range selector {
		s[v1alpha2.LabelKey(k)] = v1alpha2.LabelValue(v)
	}
	m.Spec.Selector = s
	return m
}

func (m *InferencePoolWrapper) TargetPortNumber(p int32) *InferencePoolWrapper {
	m.Spec.TargetPortNumber = p
	return m
}

// Obj returns the wrapped InferencePool.
func (m *InferencePoolWrapper) ObjRef() *v1alpha2.InferencePool {
	return &m.InferencePool
}
