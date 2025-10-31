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
	"os"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apiextv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/remotecommand"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/config"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/env"
)

const (
	// defaultExistsTimeout is the default timeout for a resource to exist in the api server.
	defaultExistsTimeout = 30 * time.Second
	// defaultReadyTimeout is the default timeout for a resource to report a ready state.
	defaultReadyTimeout = 3 * time.Minute
	// defaultModelReadyTimeout is the default timeout for the model server deployment to report a ready state.
	defaultModelReadyTimeout = 10 * time.Minute
	// defaultInterval is the default interval to check if a resource exists or ready conditions.
	defaultInterval = time.Millisecond * 250
)

// TestConfig groups various fields together for use in the test helpers
type TestConfig struct {
	Context           context.Context
	KubeCli           *kubernetes.Clientset
	K8sClient         client.Client
	RestConfig        *rest.Config
	NsName            string
	Scheme            *runtime.Scheme
	ExistsTimeout     time.Duration
	ReadyTimeout      time.Duration
	ModelReadyTimeout time.Duration
	Interval          time.Duration
}

// NewTestConfig creates a new TestConfig instance
func NewTestConfig(nsName string) *TestConfig {
	cfg := config.GetConfigOrDie()
	gomega.Expect(cfg).NotTo(gomega.BeNil())

	kubeCli, err := kubernetes.NewForConfig(cfg)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	gomega.Expect(kubeCli).NotTo(gomega.BeNil())

	return &TestConfig{
		Context:           context.Background(),
		KubeCli:           kubeCli,
		NsName:            nsName,
		RestConfig:        cfg,
		Scheme:            runtime.NewScheme(),
		ExistsTimeout:     env.GetEnvDuration("EXISTS_TIMEOUT", defaultExistsTimeout, ginkgo.GinkgoLogr),
		ReadyTimeout:      env.GetEnvDuration("READY_TIMEOUT", defaultReadyTimeout, ginkgo.GinkgoLogr),
		ModelReadyTimeout: env.GetEnvDuration("MODEL_READY_TIMEOUT", defaultModelReadyTimeout, ginkgo.GinkgoLogr),
		Interval:          defaultInterval,
	}
}

// CreateCli creates the Kubernetes client used in the tests, invoked after the scheme has been setup.
func (testConfig *TestConfig) CreateCli() {
	var err error
	testConfig.K8sClient, err = client.New(testConfig.RestConfig, client.Options{Scheme: testConfig.Scheme})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	gomega.Expect(testConfig.K8sClient).NotTo(gomega.BeNil())
}

// DeleteClusterResources deletes all cluster-scoped objects the tests typically create.
func DeleteClusterResources(testConfig *TestConfig) error {
	binding := &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "auth-reviewer-binding",
		},
	}
	err := testConfig.K8sClient.Delete(testConfig.Context, binding, client.PropagationPolicy(metav1.DeletePropagationForeground))
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	role := &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name: "auth-reviewer",
		},
	}
	err = testConfig.K8sClient.Delete(testConfig.Context, role, client.PropagationPolicy(metav1.DeletePropagationForeground))
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	metricsReaderBinding := &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "inference-gateway-sa-metrics-reader-role-binding",
		},
	}
	err = testConfig.K8sClient.Delete(testConfig.Context, metricsReaderBinding, client.PropagationPolicy(metav1.DeletePropagationForeground))
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	metricsReaderRole := &rbacv1.ClusterRole{
		ObjectMeta: metav1.ObjectMeta{
			Name: "inference-gateway-metrics-reader",
		},
	}
	err = testConfig.K8sClient.Delete(testConfig.Context, metricsReaderRole, client.PropagationPolicy(metav1.DeletePropagationForeground))
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	model := &apiextv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: "inferenceobjectives.inference.networking.x-k8s.io",
		},
	}
	err = testConfig.K8sClient.Delete(testConfig.Context, model, client.PropagationPolicy(metav1.DeletePropagationForeground))
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	pool := &apiextv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: "inferencepools.inference.networking.k8s.io",
		},
	}
	err = testConfig.K8sClient.Delete(testConfig.Context, pool, client.PropagationPolicy(metav1.DeletePropagationForeground))
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	return nil
}

// DeleteNamespacedResources deletes all namespace-scoped objects the tests typically create.
// The given namespace will also be deleted if it's not "default".
func DeleteNamespacedResources(testConfig *TestConfig) error {
	if testConfig.NsName == "" {
		return nil
	}
	err := testConfig.K8sClient.DeleteAllOf(testConfig.Context, &appsv1.Deployment{}, client.InNamespace(testConfig.NsName), client.PropagationPolicy(metav1.DeletePropagationForeground))
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	err = testConfig.K8sClient.DeleteAllOf(testConfig.Context, &corev1.Service{}, client.InNamespace(testConfig.NsName), client.PropagationPolicy(metav1.DeletePropagationForeground))
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	err = testConfig.K8sClient.DeleteAllOf(testConfig.Context, &corev1.Pod{}, client.InNamespace(testConfig.NsName), client.PropagationPolicy(metav1.DeletePropagationForeground))
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	err = testConfig.K8sClient.DeleteAllOf(testConfig.Context, &corev1.ConfigMap{}, client.InNamespace(testConfig.NsName), client.PropagationPolicy(metav1.DeletePropagationForeground))
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	err = testConfig.K8sClient.DeleteAllOf(testConfig.Context, &corev1.Secret{}, client.InNamespace(testConfig.NsName), client.PropagationPolicy(metav1.DeletePropagationForeground))
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	err = testConfig.K8sClient.DeleteAllOf(testConfig.Context, &corev1.ServiceAccount{}, client.InNamespace(testConfig.NsName), client.PropagationPolicy(metav1.DeletePropagationForeground))
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	err = testConfig.K8sClient.DeleteAllOf(testConfig.Context, &v1.InferencePool{}, client.InNamespace(testConfig.NsName), client.PropagationPolicy(metav1.DeletePropagationForeground))
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	err = testConfig.K8sClient.DeleteAllOf(testConfig.Context, &v1alpha2.InferenceObjective{}, client.InNamespace(testConfig.NsName), client.PropagationPolicy(metav1.DeletePropagationForeground))
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	if testConfig.NsName != "default" {
		ns := &corev1.Namespace{
			ObjectMeta: metav1.ObjectMeta{
				Name: testConfig.NsName,
			},
		}
		if err := testConfig.K8sClient.Delete(testConfig.Context, ns, client.PropagationPolicy(metav1.DeletePropagationForeground)); err != nil && !apierrors.IsNotFound(err) {
			return err
		}
	}
	return nil
}

// DeleteInferenceObjectiveResources deletes all InferenceObjective objects in the given namespace.
func DeleteInferenceObjectiveResources(testConfig *TestConfig) error {
	if testConfig.NsName == "" {
		return nil
	}
	err := testConfig.K8sClient.DeleteAllOf(testConfig.Context, &v1alpha2.InferenceObjective{}, client.InNamespace(testConfig.NsName), client.PropagationPolicy(metav1.DeletePropagationForeground))
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	return nil
}

// PodReady checks if the given Pod reports the "Ready" status condition before the given timeout.
func PodReady(testConfig *TestConfig, pod *corev1.Pod) {
	ginkgo.By(fmt.Sprintf("Checking pod %s/%s status is: %s", pod.Namespace, pod.Name, corev1.PodReady))
	conditions := []corev1.PodCondition{
		{
			Type:   corev1.PodReady,
			Status: corev1.ConditionTrue,
		},
	}
	gomega.Eventually(checkPodStatus, testConfig.ExistsTimeout, testConfig.Interval).
		WithArguments(testConfig, pod, conditions).Should(gomega.BeTrue())
}

// checkPodStatus checks if the given Pod status matches the expected conditions.
func checkPodStatus(testConfig *TestConfig, pod *corev1.Pod, conditions []corev1.PodCondition) (bool, error) {
	var fetchedPod corev1.Pod
	if err := testConfig.K8sClient.Get(testConfig.Context, types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}, &fetchedPod); err != nil {
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
func DeploymentAvailable(testConfig *TestConfig, deploy *appsv1.Deployment) {
	ginkgo.By(fmt.Sprintf("Checking if deployment %s/%s status is: %s", deploy.Namespace, deploy.Name, appsv1.DeploymentAvailable))
	conditions := []appsv1.DeploymentCondition{
		{
			Type:   appsv1.DeploymentAvailable,
			Status: corev1.ConditionTrue,
		},
	}
	gomega.Eventually(checkDeploymentStatus, testConfig.ModelReadyTimeout, testConfig.Interval).
		WithArguments(testConfig.Context, testConfig.K8sClient, deploy, conditions).
		Should(gomega.BeTrue())
}

// DeploymentReadyReplicas checks if the given Deployment has at least `count` ready replicas before the given timeout.
func DeploymentReadyReplicas(testConfig *TestConfig, deploy *appsv1.Deployment, count int) {
	ginkgo.By(fmt.Sprintf("Checking if deployment %s/%s has at least %d ready replica(s)", deploy.Namespace, deploy.Name, count))
	gomega.Eventually(func(g gomega.Gomega) {
		var fetchedDeploy appsv1.Deployment
		err := testConfig.K8sClient.Get(testConfig.Context, types.NamespacedName{Namespace: deploy.Namespace, Name: deploy.Name}, &fetchedDeploy)
		g.Expect(err).NotTo(gomega.HaveOccurred())
		g.Expect(fetchedDeploy.Status.ReadyReplicas).To(gomega.BeNumerically(">=", count),
			fmt.Sprintf("Deployment only has %d ready replicas, want at least %d", fetchedDeploy.Status.ReadyReplicas, count))
	}, testConfig.ModelReadyTimeout, testConfig.Interval).Should(gomega.Succeed())
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
func CRDEstablished(testConfig *TestConfig, crd *apiextv1.CustomResourceDefinition) {
	ginkgo.By(fmt.Sprintf("Checking CRD %s status is: %s", crd.Name, apiextv1.Established))
	conditions := []apiextv1.CustomResourceDefinitionCondition{
		{
			Type:   apiextv1.Established,
			Status: apiextv1.ConditionTrue,
		},
	}
	gomega.Eventually(checkCrdStatus, testConfig.ReadyTimeout, testConfig.Interval).
		WithArguments(testConfig.Context, testConfig.K8sClient, crd, conditions).
		Should(gomega.BeTrue())
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
func ExecCommandInPod(testConfig *TestConfig, podName, containerName string, cmd []string) (string, error) {
	parameterCodec := runtime.NewParameterCodec(testConfig.Scheme)

	req := testConfig.KubeCli.CoreV1().RESTClient().
		Post().
		Resource("pods").
		Name(podName).
		Namespace(testConfig.NsName).
		SubResource("exec").
		VersionedParams(&corev1.PodExecOptions{
			Container: containerName,
			Command:   cmd,
			Stdin:     false,
			Stdout:    true,
			Stderr:    true,
			TTY:       false,
		}, parameterCodec)

	exec, err := remotecommand.NewSPDYExecutor(testConfig.RestConfig, "POST", req.URL())
	if err != nil {
		return "", fmt.Errorf("could not initialize executor: %w", err)
	}

	var stdout, stderr bytes.Buffer
	execErr := exec.StreamWithContext(testConfig.Context, remotecommand.StreamOptions{
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
func EventuallyExists(testConfig *TestConfig, getResource func() error) {
	gomega.Eventually(func() error {
		return getResource()
	}, testConfig.ExistsTimeout, testConfig.Interval).Should(gomega.Succeed())
}

// CreateObjsFromYaml creates K8S objects from yaml and waits for them to be instantiated
func CreateObjsFromYaml(testConfig *TestConfig, docs []string) []string {
	objNames := []string{}

	// For each doc, decode and create
	decoder := serializer.NewCodecFactory(testConfig.Scheme).UniversalDeserializer()
	for _, doc := range docs {
		trimmed := strings.TrimSpace(doc)
		if trimmed == "" {
			continue
		}
		// Decode into a runtime.Object
		obj, gvk, decodeErr := decoder.Decode([]byte(trimmed), nil, nil)
		gomega.Expect(decodeErr).NotTo(gomega.HaveOccurred(),
			"Failed to decode YAML document to a Kubernetes object")

		ginkgo.By(fmt.Sprintf("Decoded GVK: %s", gvk))

		unstrObj, ok := obj.(*unstructured.Unstructured)
		if !ok {
			// Fallback if it's a typed object
			unstrObj = &unstructured.Unstructured{}
			// Convert typed to unstructured
			err := testConfig.Scheme.Convert(obj, unstrObj, nil)
			gomega.Expect(err).NotTo(gomega.HaveOccurred())
		}

		unstrObj.SetNamespace(testConfig.NsName)
		kind := unstrObj.GetKind()
		name := unstrObj.GetName()
		objNames = append(objNames, kind+"/"+name)

		// Create the object
		err := testConfig.K8sClient.Create(testConfig.Context, unstrObj, &client.CreateOptions{})
		gomega.Expect(err).NotTo(gomega.HaveOccurred(),
			"Failed to create object from YAML")

		// Wait for the created object to exist.
		clientObj := getClientObject(kind)
		EventuallyExists(testConfig, func() error {
			return testConfig.K8sClient.Get(testConfig.Context,
				types.NamespacedName{Namespace: testConfig.NsName, Name: name}, clientObj)
		})

		switch kind {
		case "CustomResourceDefinition":
			// Wait for the CRD to be established.
			CRDEstablished(testConfig, clientObj.(*apiextv1.CustomResourceDefinition))
		case "Deployment":
			// Wait for the deployment to be available.
			DeploymentAvailable(testConfig, clientObj.(*appsv1.Deployment))
		case "Pod":
			// Wait for the pod to be ready.
			PodReady(testConfig, clientObj.(*corev1.Pod))
		}
	}
	return objNames
}

// DeleteObjects deletes  set of Kubernetes objects in the form of kind/name
func DeleteObjects(testConfig *TestConfig, kindAndNames []string) {
	for _, kindAndName := range kindAndNames {
		split := strings.Split(kindAndName, "/")
		clientObj := getClientObject(split[0])
		err := testConfig.K8sClient.Get(testConfig.Context,
			types.NamespacedName{Namespace: testConfig.NsName, Name: split[1]}, clientObj)
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
		err = testConfig.K8sClient.Delete(testConfig.Context, clientObj)
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
		gomega.Eventually(func() bool {
			clientObj := getClientObject(split[0])
			err := testConfig.K8sClient.Get(testConfig.Context,
				types.NamespacedName{Namespace: testConfig.NsName, Name: split[1]}, clientObj)
			return apierrors.IsNotFound(err)
		}, testConfig.ExistsTimeout, testConfig.Interval).Should(gomega.BeTrue())
	}
}

// applyYAMLFile reads a file containing YAML (possibly multiple docs)
// and applies each object to the cluster.
func ApplyYAMLFile(testConfig *TestConfig, filePath string) {
	// Create the resources from the manifest file
	CreateObjsFromYaml(testConfig, ReadYaml(filePath))
}

// ReadYaml is a helper function to read in K8S YAML files and split by the --- separator
func ReadYaml(filePath string) []string {
	ginkgo.By("Reading YAML file: " + filePath)
	yamlBytes, err := os.ReadFile(filePath)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// Split multiple docs, if needed
	return strings.Split(string(yamlBytes), "\n---")
}

func getClientObject(kind string) client.Object {
	switch strings.ToLower(kind) {
	case "clusterrole":
		return &rbacv1.ClusterRole{}
	case "clusterrolebinding":
		return &rbacv1.ClusterRoleBinding{}
	case "configmap":
		return &corev1.ConfigMap{}
	case "customresourcedefinition":
		return &apiextv1.CustomResourceDefinition{}
	case "deployment":
		return &appsv1.Deployment{}
	case "inferencepool":
		return &v1.InferencePool{}
	case "pod":
		return &corev1.Pod{}
	case "role":
		return &rbacv1.Role{}
	case "rolebinding":
		return &rbacv1.RoleBinding{}
	case "secret":
		return &corev1.Secret{}
	case "service":
		return &corev1.Service{}
	case "serviceaccount":
		return &corev1.ServiceAccount{}
	default:
		ginkgo.Fail("unsupported K8S kind "+kind, 1)
		return nil
	}
}
