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
	"strconv"

	infextv1a1 "inference.networking.x-k8s.io/gateway-api-inference-extension/api/v1alpha1"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"
)

// NewTestClientPodSpec returns a PodSpec for the test client.
func NewTestClientPodSpec() *corev1.PodSpec {
	return &corev1.PodSpec{
		RestartPolicy: corev1.RestartPolicyNever,
		Containers: []corev1.Container{
			{
				Name:            "curl",
				Image:           "curlimages/curl:7.83.1",
				ImagePullPolicy: corev1.PullIfNotPresent,
				Command:         []string{"tail", "-f", "/dev/null"},
			},
		},
	}
}

// NewTestModelDeploySpec returns a DeploymentSpec for the model server.
func NewTestModelDeploySpec(name string) *appsv1.DeploymentSpec {
	return &appsv1.DeploymentSpec{
		Replicas: ptr.To(int32(1)),
		Selector: &metav1.LabelSelector{
			MatchLabels: map[string]string{"app": name},
		},
		Template: corev1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Labels: map[string]string{"app": name},
			},
			Spec: newTestModelPodSpec(),
		},
	}
}

// newTestModelPodSpec returns a PodSpec for the model server.
func newTestModelPodSpec() corev1.PodSpec {
	return corev1.PodSpec{
		InitContainers: []corev1.Container{
			{
				Name:            "adapter-loader",
				Image:           "ghcr.io/tomatillo-and-multiverse/adapter-puller:demo",
				ImagePullPolicy: corev1.PullAlways,
				Command:         []string{"python"},
				Args: []string{
					"./pull_adapters.py",
					"--adapter", "yard1/llama-2-7b-sql-lora-test",
					"--adapter", "vineetsharma/qlora-adapter-Llama-2-7b-hf-TweetSumm",
					"--duplicate-count", "5",
				},
				Env: []corev1.EnvVar{
					{
						Name: "HF_TOKEN",
						ValueFrom: &corev1.EnvVarSource{
							SecretKeyRef: &corev1.SecretKeySelector{
								LocalObjectReference: corev1.LocalObjectReference{Name: "hf-token"},
								Key:                  "token",
							},
						},
					},
					{
						Name:  "HF_HOME",
						Value: "/adapters",
					},
				},
				VolumeMounts: []corev1.VolumeMount{
					{Name: "adapters", MountPath: "/adapters"},
				},
			},
		},
		Containers: []corev1.Container{
			{
				Name:            "lora",
				Image:           "vllm/vllm-openai:latest",
				ImagePullPolicy: corev1.PullAlways,
				Command:         []string{"python3", "-m", "vllm.entrypoints.openai.api_server"},
				Args: []string{
					"--model", "meta-llama/Llama-2-7b-hf",
					"--tensor-parallel-size", "1",
					"--port", "8000",
					"--enable-lora",
					"--max-loras", "4",
					"--max-cpu-loras", "12",
					"--lora-modules",
					"sql-lora=/adapters/hub/models--yard1--llama-2-7b-sql-lora-test/snapshots/0dfa347e8877a4d4ed19ee56c140fa518470028c/",
					"tweet-summary=/adapters/hub/models--vineetsharma--qlora-adapter-Llama-2-7b-hf-TweetSumm/snapshots/796337d8e866318c59e38f16416e3ecd11fe5403",
					"sql-lora-0=/adapters/yard1/llama-2-7b-sql-lora-test_0",
					"sql-lora-1=/adapters/yard1/llama-2-7b-sql-lora-test_1",
					"sql-lora-2=/adapters/yard1/llama-2-7b-sql-lora-test_2",
					"sql-lora-3=/adapters/yard1/llama-2-7b-sql-lora-test_3",
					"sql-lora-4=/adapters/yard1/llama-2-7b-sql-lora-test_4",
					"tweet-summary-0=/adapters/vineetsharma/qlora-adapter-Llama-2-7b-hf-TweetSumm_0",
					"tweet-summary-1=/adapters/vineetsharma/qlora-adapter-Llama-2-7b-hf-TweetSumm_1",
					"tweet-summary-2=/adapters/vineetsharma/qlora-adapter-Llama-2-7b-hf-TweetSumm_2",
					"tweet-summary-3=/adapters/vineetsharma/qlora-adapter-Llama-2-7b-hf-TweetSumm_3",
					"tweet-summary-4=/adapters/vineetsharma/qlora-adapter-Llama-2-7b-hf-TweetSumm_4",
				},
				Env: []corev1.EnvVar{
					{
						Name: "HUGGING_FACE_HUB_TOKEN",
						ValueFrom: &corev1.EnvVarSource{
							SecretKeyRef: &corev1.SecretKeySelector{
								LocalObjectReference: corev1.LocalObjectReference{Name: "hf-token"},
								Key:                  "token",
							},
						},
					},
					{Name: "PORT", Value: "8000"},
				},
				Ports: []corev1.ContainerPort{
					{ContainerPort: 8000, Name: "http", Protocol: corev1.ProtocolTCP},
				},
				LivenessProbe: &corev1.Probe{
					ProbeHandler: corev1.ProbeHandler{
						HTTPGet: &corev1.HTTPGetAction{
							Path: "/health", Port: intstr.FromString("http"), Scheme: corev1.URISchemeHTTP,
						},
					},
					InitialDelaySeconds: 5, PeriodSeconds: 5, FailureThreshold: 240,
				},
				ReadinessProbe: &corev1.Probe{
					ProbeHandler: corev1.ProbeHandler{
						HTTPGet: &corev1.HTTPGetAction{
							Path: "/health", Port: intstr.FromString("http"), Scheme: corev1.URISchemeHTTP,
						},
					},
					InitialDelaySeconds: 5, PeriodSeconds: 5, FailureThreshold: 600,
				},
				Resources: corev1.ResourceRequirements{
					Limits: corev1.ResourceList{
						corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("1"),
					},
					Requests: corev1.ResourceList{
						corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("1"),
					},
				},
				VolumeMounts: []corev1.VolumeMount{
					{
						Name:      "data",
						MountPath: "/data",
					},
					{
						Name:      "shm",
						MountPath: "/dev/shm",
					},
					{
						Name:      "adapters",
						MountPath: "/adapters",
					},
				},
			},
		},
		Volumes: []corev1.Volume{
			{Name: "data", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}},
			{Name: "shm", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{Medium: corev1.StorageMediumMemory}}},
			{Name: "adapters", VolumeSource: corev1.VolumeSource{EmptyDir: &corev1.EmptyDirVolumeSource{}}},
		},
	}
}

// NewTestModelServiceSpec returns a ServiceSpec for the model server.
func NewTestModelServiceSpec(name string) *corev1.ServiceSpec {
	return &corev1.ServiceSpec{
		Type:     corev1.ServiceTypeClusterIP,
		Selector: map[string]string{"app": name},
		Ports: []corev1.ServicePort{
			{
				Protocol:   corev1.ProtocolTCP,
				Port:       8000,
				TargetPort: intstr.FromInt(8000),
			},
		},
	}
}

// NewTestEnvoyDeploySpec returns a DeploymentSpec for the Envoy proxy.
func NewTestEnvoyDeploySpec(name, port string) (*appsv1.DeploymentSpec, error) {
	spec, err := newTestEnvoyPodSpec(name, port)
	if err != nil {
		return nil, err
	}
	return &appsv1.DeploymentSpec{
		Replicas: ptr.To(int32(1)),
		Selector: &metav1.LabelSelector{
			MatchLabels: map[string]string{"app": name},
		},
		Template: corev1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Labels: map[string]string{"app": name},
			},
			Spec: *spec,
		},
	}, nil
}

// newTestEnvoyPodSpec returns a PodSpec for the Envoy proxy.
func newTestEnvoyPodSpec(name, port string) (*corev1.PodSpec, error) {
	// Convert port to an integer.
	portInt, err := strconv.Atoi(port)
	if err != nil {
		klog.Errorf("Error converting port: %v", err)
		return nil, err
	}

	return &corev1.PodSpec{
		Containers: []corev1.Container{
			{
				Name:    name,
				Image:   "docker.io/envoyproxy/envoy:distroless-v1.32.2",
				Command: []string{"envoy"},
				Args: []string{
					"--service-cluster", "$(NS_NAME)",
					"--service-node", "$(POD_NAME)",
					"--log-level", "debug",
					"--cpuset-threads",
					"--drain-strategy", "immediate",
					"--drain-time-s", "60",
					"-c", "/etc/envoy/envoy.yaml",
				},
				Env: []corev1.EnvVar{
					{
						Name: "NS_NAME",
						ValueFrom: &corev1.EnvVarSource{
							FieldRef: &corev1.ObjectFieldSelector{
								FieldPath: "metadata.namespace",
							},
						},
					},
					{
						Name: "POD_NAME",
						ValueFrom: &corev1.EnvVarSource{
							FieldRef: &corev1.ObjectFieldSelector{
								FieldPath: "metadata.name",
							},
						},
					},
				},
				Ports: []corev1.ContainerPort{
					{ContainerPort: int32(portInt), Name: "http-" + port, Protocol: corev1.ProtocolTCP},
					{ContainerPort: 19001, Name: "metrics", Protocol: corev1.ProtocolTCP},
				},
				ReadinessProbe: &corev1.Probe{
					ProbeHandler: corev1.ProbeHandler{
						HTTPGet: &corev1.HTTPGetAction{
							Path: "/ready", Port: intstr.FromString("metrics"), Scheme: corev1.URISchemeHTTP,
						},
					},
					InitialDelaySeconds: 1, PeriodSeconds: 5, SuccessThreshold: 1, TimeoutSeconds: 1,
				},
				VolumeMounts: []corev1.VolumeMount{
					{Name: "config", MountPath: "/etc/envoy", ReadOnly: true},
				},
			},
		},
		Volumes: []corev1.Volume{
			{
				Name: "config",
				VolumeSource: corev1.VolumeSource{
					ConfigMap: &corev1.ConfigMapVolumeSource{
						LocalObjectReference: corev1.LocalObjectReference{Name: name},
						Items: []corev1.KeyToPath{
							{Key: "envoy.yaml", Path: "envoy.yaml"},
						},
					},
				},
			},
		},
	}, nil
}

// NewTestEnvoyServiceSpec returns a ServiceSpec for the Envoy proxy.
func NewTestEnvoyServiceSpec(name, port string) (*corev1.ServiceSpec, error) {
	// Convert port to an integer.
	portInt, err := strconv.Atoi(port)
	if err != nil {
		klog.Errorf("Error converting port: %v", err)
		return nil, err
	}

	return &corev1.ServiceSpec{
		Type:     corev1.ServiceTypeClusterIP,
		Selector: map[string]string{"app": name},
		Ports: []corev1.ServicePort{
			{
				Name:       "http-" + port,
				Protocol:   corev1.ProtocolTCP,
				Port:       int32(portInt),
				TargetPort: intstr.FromInt(portInt),
			},
		},
	}, nil
}

// NewTestInferExtDeploySpec returns a DeploymentSpec for the inference extension.
func NewTestInferExtDeploySpec(name, pool, svc string) *appsv1.DeploymentSpec {
	return &appsv1.DeploymentSpec{
		Replicas: ptr.To(int32(1)),
		Selector: &metav1.LabelSelector{
			MatchLabels: map[string]string{"app": name},
		},
		Template: corev1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Labels: map[string]string{"app": name},
			},
			Spec: newTestInferExtPodSpec(name, pool, svc),
		},
	}
}

// newTestInferExtPodSpec returns a PodSpec for the inference extension.
func newTestInferExtPodSpec(name, pool, svc string) corev1.PodSpec {
	return corev1.PodSpec{
		Containers: []corev1.Container{
			{
				Name:  name,
				Image: "us-central1-docker.pkg.dev/k8s-staging-images/llm-instance-gateway/epp:main",
				Args: []string{
					"-poolName", pool,
					"-poolNamespace", "$(NS_NAME)",
					"-v", "3",
					"-serviceName", svc,
					"-grpcPort", "9002",
					"-grpcHealthPort", "9003",
				},
				Ports: []corev1.ContainerPort{
					{ContainerPort: 9002, Protocol: corev1.ProtocolTCP},
					{ContainerPort: 9003, Protocol: corev1.ProtocolTCP},
				},
				Env: []corev1.EnvVar{
					{
						Name: "NS_NAME",
						ValueFrom: &corev1.EnvVarSource{
							FieldRef: &corev1.ObjectFieldSelector{
								FieldPath: "metadata.namespace",
							},
						},
					},
				},
				LivenessProbe: &corev1.Probe{
					ProbeHandler: corev1.ProbeHandler{
						GRPC: &corev1.GRPCAction{
							Port:    9003,
							Service: ptr.To("inference-extension"),
						},
					},
					InitialDelaySeconds: 5, PeriodSeconds: 10,
				},
				ReadinessProbe: &corev1.Probe{
					ProbeHandler: corev1.ProbeHandler{
						GRPC: &corev1.GRPCAction{
							Port:    9003,
							Service: ptr.To("inference-extension"),
						},
					},
					InitialDelaySeconds: 5, PeriodSeconds: 10,
				},
			},
		},
	}
}

// NewTestInferExtServiceSpec returns a ServiceSpec for the inference extension.
func NewTestInferExtServiceSpec(name string) *corev1.ServiceSpec {
	return &corev1.ServiceSpec{
		Type:     corev1.ServiceTypeClusterIP,
		Selector: map[string]string{"app": name},
		Ports: []corev1.ServicePort{
			{
				Protocol:   corev1.ProtocolTCP,
				Port:       9002,
				TargetPort: intstr.FromInt(9002),
			},
		},
	}
}

// NewTestInferExtRules returns the ClusterRole rules for the inference extension.
func NewTestInferExtRules() []rbacv1.PolicyRule {
	return []rbacv1.PolicyRule{
		{
			APIGroups: []string{infextv1a1.GroupVersion.Group},
			Resources: []string{infextv1a1.ResourceInferenceModel},
			Verbs:     []string{"get", "watch", "list"},
		},
		{
			APIGroups: []string{infextv1a1.GroupVersion.Group},
			Resources: []string{infextv1a1.ResourceInferencePool},
			Verbs:     []string{"get", "watch", "list"},
		},
		{
			APIGroups: []string{""},
			Resources: []string{"pods"},
			Verbs:     []string{"get", "watch", "list"},
		},
		{
			APIGroups: []string{discoveryv1.GroupName},
			Resources: []string{"endpointslices"},
			Verbs:     []string{"get", "watch", "list"},
		},
	}
}
