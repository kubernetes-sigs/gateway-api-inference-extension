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
	"bytes"
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apiextv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/remotecommand"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
)

// DeleteClusterResources deletes all cluster-scoped objects the tests typically create.
func DeleteClusterResources(ctx context.Context, cli client.Client) error {
	binding := &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod-read-binding",
		},
	}
	err := cli.Delete(ctx, binding, client.PropagationPolicy(metav1.DeletePropagationForeground))
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	role := &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod-read",
		},
	}
	err = cli.Delete(ctx, role, client.PropagationPolicy(metav1.DeletePropagationForeground))
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	model := &apiextv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: "inferencemodels.inference.networking.x-k8s.io",
		},
	}
	err = cli.Delete(ctx, model, client.PropagationPolicy(metav1.DeletePropagationForeground))
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	pool := &apiextv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: "inferencepools.inference.networking.x-k8s.io",
		},
	}
	err = cli.Delete(ctx, pool, client.PropagationPolicy(metav1.DeletePropagationForeground))
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	return nil
}

// DeleteNamespacedResources deletes all namespace-scoped objects the tests typically create.
// The given namespace will also be deleted if it's not "default".
func DeleteNamespacedResources(ctx context.Context, cli client.Client, ns string) error {
	if ns == "" {
		return nil
	}
	err := cli.DeleteAllOf(ctx, &appsv1.Deployment{}, client.InNamespace(ns), client.PropagationPolicy(metav1.DeletePropagationForeground))
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	err = cli.DeleteAllOf(ctx, &corev1.Service{}, client.InNamespace(ns), client.PropagationPolicy(metav1.DeletePropagationForeground))
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	err = cli.DeleteAllOf(ctx, &corev1.Pod{}, client.InNamespace(ns), client.PropagationPolicy(metav1.DeletePropagationForeground))
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	err = cli.DeleteAllOf(ctx, &corev1.ConfigMap{}, client.InNamespace(ns), client.PropagationPolicy(metav1.DeletePropagationForeground))
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	err = cli.DeleteAllOf(ctx, &corev1.Secret{}, client.InNamespace(ns), client.PropagationPolicy(metav1.DeletePropagationForeground))
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	err = cli.DeleteAllOf(ctx, &v1alpha2.InferencePool{}, client.InNamespace(ns), client.PropagationPolicy(metav1.DeletePropagationForeground))
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	err = cli.DeleteAllOf(ctx, &v1alpha2.InferenceModel{}, client.InNamespace(ns), client.PropagationPolicy(metav1.DeletePropagationForeground))
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	if ns != "default" {
		ns := &corev1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name: ns,
			},
		}
		if err := cli.Delete(ctx, ns, client.PropagationPolicy(metav1.DeletePropagationForeground)); err != nil && !apierrors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

// DeleteInferenceModelResources deletes all InferenceModel objects in the given namespace.
func DeleteInferenceModelResources(ctx context.Context, cli client.Client, ns string) error {
	if ns == "" {
		return nil
	}
	err := cli.DeleteAllOf(ctx, &v1alpha2.InferenceModel{}, client.InNamespace(ns), client.PropagationPolicy(metav1.DeletePropagationForeground))
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	return nil
}

// PodReady checks if the given Pod reports the "Ready" status condition before the given timeout.
func PodReady(ctx context.Context, cli client.Client, pod *corev1.Pod, timeout, interval time.Duration) {
	ginkgo.By(fmt.Sprintf("Checking pod %s/%s status is: %s", pod.Namespace, pod.Name, corev1.PodReady))
	conditions := []corev1.PodCondition{
		{
			Type:   corev1.PodReady,
			Status: corev1.ConditionTrue,
		},
	}
	gomega.Eventually(checkPodStatus, timeout, interval).WithArguments(ctx, cli, pod, conditions).Should(gomega.BeTrue())
}

// checkPodStatus checks if the given Pod status matches the expected conditions.
func checkPodStatus(ctx context.Context, cli client.Client, pod *corev1.Pod, conditions []corev1.PodCondition) (bool, error) {
	var fetchedPod corev1.Pod
	if err := cli.Get(ctx, types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}, &fetchedPod); err != nil {
		return false, err
	}
	found := 0
	for _, want := range conditions {
		for _, c := range fetchedPod.Status.Conditions {
			if c.Type == want.Type && c.Status == want.Status {
				found += 1
			}
		}
	}
	return found == len(conditions), nil
}

// DeploymentAvailable checks if the given Deployment reports the "Available" status condition before the given timeout.
func DeploymentAvailable(ctx context.Context, cli client.Client, deploy *appsv1.Deployment, timeout, interval time.Duration) {
	ginkgo.By(fmt.Sprintf("Checking if deployment %s/%s status is: %s", deploy.Namespace, deploy.Name, appsv1.DeploymentAvailable))
	conditions := []appsv1.DeploymentCondition{
		{
			Type:   appsv1.DeploymentAvailable,
			Status: corev1.ConditionTrue,
		},
	}
	gomega.Eventually(checkDeploymentStatus, timeout, interval).WithArguments(ctx, cli, deploy, conditions).Should(gomega.BeTrue())
}

// checkDeploymentStatus checks if the given Deployment status matches the expected conditions.
func checkDeploymentStatus(ctx context.Context, cli client.Client, deploy *appsv1.Deployment, conditions []appsv1.DeploymentCondition) (bool, error) {
	var fetchedDeploy appsv1.Deployment
	if err := cli.Get(ctx, types.NamespacedName{Namespace: deploy.Namespace, Name: deploy.Name}, &fetchedDeploy); err != nil {
		return false, err
	}
	found := 0
	for _, want := range conditions {
		for _, c := range fetchedDeploy.Status.Conditions {
			if c.Type == want.Type && c.Status == want.Status {
				found += 1
			}
		}
	}
	return found == len(conditions), nil
}

// CRDEstablished checks if the given CRD reports the "Established" status condition before the given timeout.
func CRDEstablished(ctx context.Context, cli client.Client, crd *apiextv1.CustomResourceDefinition, timeout, interval time.Duration) {
	ginkgo.By(fmt.Sprintf("Checking CRD %s status is: %s", crd.Name, apiextv1.Established))
	conditions := []apiextv1.CustomResourceDefinitionCondition{
		{
			Type:   apiextv1.Established,
			Status: apiextv1.ConditionTrue,
		},
	}
	gomega.Eventually(checkCrdStatus, timeout, interval).WithArguments(ctx, cli, crd, conditions).Should(gomega.BeTrue())
}

// checkCrdStatus checks if the given CRD status matches the expected conditions.
func checkCrdStatus(
	ctx context.Context,
	cli client.Client,
	crd *apiextv1.CustomResourceDefinition,
	conditions []apiextv1.CustomResourceDefinitionCondition,
) (bool, error) {
	var fetchedCrd apiextv1.CustomResourceDefinition
	if err := cli.Get(ctx, types.NamespacedName{Name: crd.Name}, &fetchedCrd); err != nil {
		return false, err
	}
	found := 0
	for _, want := range conditions {
		for _, c := range fetchedCrd.Status.Conditions {
			if c.Type == want.Type && c.Status == want.Status {
				found += 1
			}
		}
	}
	return found == len(conditions), nil
}

// ExecCommandInPod runs a command in a given container of a given Pod, returning combined stdout+stderr.
func ExecCommandInPod(
	ctx context.Context,
	cfg *rest.Config,
	scheme *runtime.Scheme,
	kubeClient *kubernetes.Clientset,
	podNamespace, podName, containerName string,
	cmd []string,
) (string, error) {

	parameterCodec := runtime.NewParameterCodec(scheme)

	req := kubeClient.CoreV1().RESTClient().
		Post().
		Resource("pods").
		Name(podName).
		Namespace(podNamespace).
		SubResource("exec").
		VersionedParams(&corev1.PodExecOptions{
			Container: containerName,
			Command:   cmd,
			Stdin:     false,
			Stdout:    true,
			Stderr:    true,
			TTY:       false,
		}, parameterCodec)

	exec, err := remotecommand.NewSPDYExecutor(cfg, "POST", req.URL())
	if err != nil {
		return "", fmt.Errorf("could not initialize executor: %w", err)
	}

	var stdout, stderr bytes.Buffer
	execErr := exec.StreamWithContext(ctx, remotecommand.StreamOptions{
		Stdout: &stdout,
		Stderr: &stderr,
	})

	combinedOutput := stdout.String() + stderr.String()

	if execErr != nil {
		return combinedOutput, fmt.Errorf("exec dial command %v failed: %w", cmd, execErr)
	}

	return combinedOutput, nil
}

// EventuallyExists checks if a Kubernetes resource exists and returns nil if successful.
// It takes a function `getResource` which retrieves the resource and returns an error if it doesn't exist.
func EventuallyExists(ctx context.Context, getResource func() error, timeout, interval time.Duration) {
	gomega.Eventually(func() error {
		return getResource()
	}, timeout, interval).Should(gomega.Succeed())
}
