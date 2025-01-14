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
	infextv1a1 "inference.networking.x-k8s.io/gateway-api-inference-extension/api/v1alpha1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	klog "k8s.io/klog/v2"
)

// InferencePoolWrapper wraps an InferencePool.
type InferencePoolWrapper struct {
	infextv1a1.InferencePool
}

// MakePoolWrapper creates a wrapper for an InferencePool.
func MakePoolWrapper(name, ns string) *InferencePoolWrapper {
	return &InferencePoolWrapper{
		infextv1a1.InferencePool{
			ObjectMeta: metav1.ObjectMeta{
				Name:      name,
				Namespace: ns,
			},
			Spec: infextv1a1.InferencePoolSpec{
				Selector: map[infextv1a1.LabelKey]infextv1a1.LabelValue{},
			},
		},
	}
}

// SetTargetPort sets the value of the inferencePool.spec.targetPortNumber.
func (p *InferencePoolWrapper) SetTargetPort(port int) *InferencePoolWrapper {
	p.Spec.TargetPortNumber = int32(port)
	return p
}

// SetSelector sets the value of the inferencePool.spec.selector.
func (p *InferencePoolWrapper) SetSelector(s map[string]string) *InferencePoolWrapper {
	// Convert the map[string]string to map[LabelKey]LabelValue
	converted := make(map[infextv1a1.LabelKey]infextv1a1.LabelValue, len(s))
	for k, v := range s {
		converted[infextv1a1.LabelKey(k)] = infextv1a1.LabelValue(v)
	}

	p.Spec.Selector = converted
	return p
}

// AddSelector adds a k/v pair to the inferencePool.spec.selector.
func (p *InferencePoolWrapper) AddSelector(key, value string) *InferencePoolWrapper {
	p.Spec.Selector[infextv1a1.LabelKey(key)] = infextv1a1.LabelValue(value)
	return p
}

// SetLabels sets the value of the inferencePool.metadata.labels.
func (p *InferencePoolWrapper) SetLabels(labels map[string]string) *InferencePoolWrapper {
	p.Labels = labels
	return p
}

// Obj returns the inner InferencePool.
func (p *InferencePoolWrapper) Obj() *infextv1a1.InferencePool {
	return &p.InferencePool
}

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
		Kind:  infextv1a1.KindInferencePool,
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

// PodWrapper wraps a Pod.
type PodWrapper struct {
	corev1.Pod
}

// MakePodWrapper creates a wrapper for a Pod.
func MakePodWrapper(name, ns string) *PodWrapper {
	return &PodWrapper{
		corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      name,
				Namespace: ns,
			},
			Spec: corev1.PodSpec{},
		},
	}
}

// AddLabel adds a pod label.
func (p *PodWrapper) AddLabel(key, value string) *PodWrapper {
	p.ObjectMeta.Labels[key] = value
	return p
}

// SetLabels sets the pod labels.
func (p *PodWrapper) SetLabels(labels map[string]string) *PodWrapper {
	p.ObjectMeta.Labels = labels
	return p
}

// SetSpec sets the pod spec.
func (p *PodWrapper) SetSpec(spec *corev1.PodSpec) *PodWrapper {
	p.Spec = *spec
	return p
}

// Obj returns the wrapped Pod.
func (p *PodWrapper) Obj() *corev1.Pod {
	return &p.Pod
}

// PodWrapper wraps a Deployment.
type DeploymentWrapper struct {
	appsv1.Deployment
}

// MakeDeploymentWrapper creates a wrapper for a Deployment.
func MakeDeploymentWrapper(name, ns string) *DeploymentWrapper {
	return &DeploymentWrapper{
		appsv1.Deployment{
			ObjectMeta: metav1.ObjectMeta{
				Name:      name,
				Namespace: ns,
			},
			Spec: appsv1.DeploymentSpec{},
		},
	}
}

// AddLabel adds a deployment label.
func (d *DeploymentWrapper) AddLabel(key, value string) *DeploymentWrapper {
	d.ObjectMeta.Labels[key] = value
	return d
}

// SetLabels sets the deployment labels.
func (d *DeploymentWrapper) SetLabels(labels map[string]string) *DeploymentWrapper {
	d.ObjectMeta.Labels = labels
	return d
}

// SetSpec sets the deployment spec.
func (d *DeploymentWrapper) SetSpec(spec *appsv1.DeploymentSpec) *DeploymentWrapper {
	if spec == nil {
		klog.Errorf("SetSpec called with a nil spec; no changes were made")
		return d
	}
	d.Spec = *spec
	return d
}

// SetPodSpec sets the deployment pod spec.
func (d *DeploymentWrapper) SetPodSpec(spec *corev1.PodSpec) *DeploymentWrapper {
	if spec == nil {
		klog.Errorf("SetPodSpec called with a nil spec; no changes were made")
		return d
	}
	d.Spec.Template.Spec = *spec
	return d
}

// Obj returns the wrapped Deployment.
func (d *DeploymentWrapper) Obj() *appsv1.Deployment {
	return &d.Deployment
}

// ServiceWrapper wraps a Service.
type ServiceWrapper struct {
	corev1.Service
}

// MakeServiceWrapper creates a wrapper for a Service.
func MakeServiceWrapper(name, ns string) *ServiceWrapper {
	return &ServiceWrapper{
		corev1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name:      name,
				Namespace: ns,
			},
			Spec: corev1.ServiceSpec{},
		},
	}
}

// AddLabel adds a service label.
func (s *ServiceWrapper) AddLabel(key, value string) *ServiceWrapper {
	s.ObjectMeta.Labels[key] = value
	return s
}

// SetLabels sets the service labels.
func (s *ServiceWrapper) SetLabels(labels map[string]string) *ServiceWrapper {
	s.ObjectMeta.Labels = labels
	return s
}

// SetSpec sets the service spec.
func (s *ServiceWrapper) SetSpec(spec *corev1.ServiceSpec) *ServiceWrapper {
	if spec == nil {
		klog.Errorf("SetSpec called with a nil spec; no changes were made")
		return s
	}
	s.Spec = *spec
	return s
}

// Obj returns the wrapped Service.
func (s *ServiceWrapper) Obj() *corev1.Service {
	return &s.Service
}

// SecretWrapper wraps a Secret.
type SecretWrapper struct {
	corev1.Secret
}

// MakeSecretWrapper creates a wrapper for a Secret.
func MakeSecretWrapper(name, ns string) *SecretWrapper {
	return &SecretWrapper{
		corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{
				Name:      name,
				Namespace: ns,
			},
			Data: map[string][]byte{},
		},
	}
}

// AddLabel adds a secret label.
func (s *SecretWrapper) AddLabel(key, value string) *SecretWrapper {
	s.ObjectMeta.Labels[key] = value
	return s
}

// SetLabels sets the secret labels.
func (s *SecretWrapper) SetLabels(labels map[string]string) *SecretWrapper {
	s.ObjectMeta.Labels = labels
	return s
}

// SetSpec sets the secret data.
func (s *SecretWrapper) SetData(data map[string][]byte) *SecretWrapper {
	s.Data = data
	return s
}

// Obj returns the wrapped Secret.
func (s *SecretWrapper) Obj() *corev1.Secret {
	return &s.Secret
}

// ConfigMapWrapper wraps a ConfigMap.
type ConfigMapWrapper struct {
	corev1.ConfigMap
}

// MakeConfigMapWrapper creates a wrapper for a ConfigMap.
func MakeConfigMapWrapper(name, ns string) *ConfigMapWrapper {
	return &ConfigMapWrapper{
		corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      name,
				Namespace: ns,
			},
			Data: map[string]string{},
		},
	}
}

// SetLabels sets the configmap labels.
func (c *ConfigMapWrapper) SetLabels(labels map[string]string) *ConfigMapWrapper {
	c.ObjectMeta.Labels = labels
	return c
}

// SetSpec sets the configmap data.
func (c *ConfigMapWrapper) SetData(data map[string]string) *ConfigMapWrapper {
	c.Data = data
	return c
}

// Obj returns the wrapped ConfigMap.
func (c *ConfigMapWrapper) Obj() *corev1.ConfigMap {
	return &c.ConfigMap
}

// ClusterRoleWrapper wraps a ClusterRole.
type ClusterRoleWrapper struct {
	rbacv1.ClusterRole
}

// MakeClusterRoleWrapper creates a wrapper for a ClusterRole.
func MakeClusterRoleWrapper(name string) *ClusterRoleWrapper {
	return &ClusterRoleWrapper{
		rbacv1.ClusterRole{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
			Rules: []rbacv1.PolicyRule{},
		},
	}
}

// SetLabels sets the ClusterRole labels.
func (c *ClusterRoleWrapper) SetLabels(labels map[string]string) *ClusterRoleWrapper {
	c.ObjectMeta.Labels = labels
	return c
}

// SetRules sets the ClusterRole rules.
func (c *ClusterRoleWrapper) SetRules(rules []rbacv1.PolicyRule) *ClusterRoleWrapper {
	c.Rules = rules
	return c
}

// Obj returns the wrapped ClusterRole.
func (c *ClusterRoleWrapper) Obj() *rbacv1.ClusterRole {
	return &c.ClusterRole
}

// ClusterRoleBindingWrapper wraps a ClusterRolBinding.
type ClusterRoleBindingWrapper struct {
	rbacv1.ClusterRoleBinding
}

// MakeClusterRoleBindingWrapper creates a wrapper for a ClusterRoleBinding.
func MakeClusterRoleBindingWrapper(name string) *ClusterRoleBindingWrapper {
	return &ClusterRoleBindingWrapper{
		rbacv1.ClusterRoleBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name: name,
			},
			Subjects: []rbacv1.Subject{},
			RoleRef:  rbacv1.RoleRef{},
		},
	}
}

// SetLabels sets the ClusterRoleBinding labels.
func (c *ClusterRoleBindingWrapper) SetLabels(labels map[string]string) *ClusterRoleBindingWrapper {
	c.ObjectMeta.Labels = labels
	return c
}

// SetSubjects sets the ClusterRoleBinding subjects.
func (c *ClusterRoleBindingWrapper) SetSubjects(subjects []rbacv1.Subject) *ClusterRoleBindingWrapper {
	c.Subjects = subjects
	return c
}

// SetRoleRef sets the ClusterRoleBinding roleRef to the provided name, using "ClusterRole" as the kind.
func (c *ClusterRoleBindingWrapper) SetRoleRef(ref string) *ClusterRoleBindingWrapper {
	c.RoleRef = rbacv1.RoleRef{
		Kind: "ClusterRole",
		Name: ref,
	}
	return c
}

// Obj returns the wrapped ClusterRoleBinding.
func (c *ClusterRoleBindingWrapper) Obj() *rbacv1.ClusterRoleBinding {
	return &c.ClusterRoleBinding
}
