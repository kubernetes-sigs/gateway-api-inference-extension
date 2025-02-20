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
)

// PodWrapper wraps a Pod.
type PodWrapper struct {
	corev1.Pod
}

// MakePod creates a wrapper for a Pod.
func MakePod(podName, ns string) *PodWrapper {
	return &PodWrapper{
		corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      podName,
				Namespace: ns,
			},
			Spec:   corev1.PodSpec{},
			Status: corev1.PodStatus{},
		},
	}
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

// Obj returns the wrapped Pod.
func (p *PodWrapper) Obj() corev1.Pod {
	return p.Pod
}
