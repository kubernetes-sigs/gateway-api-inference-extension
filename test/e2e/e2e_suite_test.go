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

package e2e

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	infextv1a1 "inference.networking.x-k8s.io/gateway-api-inference-extension/api/v1alpha1"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/crd"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/test/consts"
	testutils "inference.networking.x-k8s.io/gateway-api-inference-extension/test/utils"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apiextv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/config"
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
	// modelServerName is the name of the model server test resources.
	modelServerName = consts.TestModelServerName
	// modelName is the tes model name.
	modelName = consts.TestModelName
	// envoyName is the name of the envoy proxy test resources.
	envoyName = consts.TestEnvoyName
	// envoyPort is the listener port number of the test envoy proxy.
	envoyPort = consts.TestEnvoyPort
	// inferExtName is the name of the test inference extension test resources.
	inferExtName = consts.TestInferExtName
)

var (
	ctx context.Context
	cli client.Client
	// Required for exec'ing in curl pod
	kubeCli *kubernetes.Clientset
	scheme  = runtime.NewScheme()
	cfg     = config.GetConfigOrDie()
	// The namespace used for all tests
	nsName = getNamespace()
)

func TestAPIs(t *testing.T) {
	gomega.RegisterFailHandler(ginkgo.Fail)
	ginkgo.RunSpecs(t,
		"End To End Test Suite",
	)
}

var _ = ginkgo.BeforeSuite(func() {
	ginkgo.By("Setting up the test suite")
	setupSuite()

	ginkgo.By("Installing CRDs")
	gomega.Expect(crd.InstallCRDs(ctx, cli, "../../config/crd/bases")).To(gomega.Succeed())

	ginkgo.By("Creating test infrastructure")
	setupInfra()
})

func setupInfra() {
	createNamespace(cli)
	createClient(cli)
	createEnvoy(cli)
	createInferExt(cli)
	// Run this step last, as it requires additional time for the model server to become ready.
	createModel(cli)
}

var _ = ginkgo.AfterSuite(func() {
	ginkgo.By("Performing global cleanup")
	cleanupResources()
})

// setupSuite initializes the test suite by setting up the Kubernetes client,
// loading required API schemes, and validating configuration.
func setupSuite() {
	ctx = context.Background()
	gomega.ExpectWithOffset(1, cfg).NotTo(gomega.BeNil())

	err := clientgoscheme.AddToScheme(scheme)
	gomega.ExpectWithOffset(1, err).NotTo(gomega.HaveOccurred())

	err = apiextv1.AddToScheme(scheme)
	gomega.ExpectWithOffset(1, err).NotTo(gomega.HaveOccurred())

	err = infextv1a1.AddToScheme(scheme)
	gomega.ExpectWithOffset(1, err).NotTo(gomega.HaveOccurred())

	cli, err = client.New(cfg, client.Options{Scheme: scheme})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	gomega.Expect(cli).NotTo(gomega.BeNil())

	kubeCli, err = kubernetes.NewForConfig(cfg)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
}

func cleanupResources() {
	gomega.Expect(testutils.DeleteClusterResources(ctx, cli)).To(gomega.Succeed())
	gomega.Expect(testutils.DeleteNamespacedResources(ctx, cli, nsName)).To(gomega.Succeed())
}

func cleanupInferModelResources() {
	gomega.Expect(testutils.DeleteInferenceModelResources(ctx, cli, nsName)).To(gomega.Succeed())
}

func getTimeout(key string, fallback time.Duration) time.Duration {
	if value, ok := os.LookupEnv(key); ok {
		if parsed, err := time.ParseDuration(value); err == nil {
			return parsed
		}
	}
	return fallback
}

var (
	existsTimeout     = getTimeout("EXISTS_TIMEOUT", defaultExistsTimeout)
	readyTimeout      = getTimeout("READY_TIMEOUT", defaultReadyTimeout)
	modelReadyTimeout = getTimeout("MODEL_READY_TIMEOUT", defaultModelReadyTimeout)
	interval          = defaultInterval
)

// getNamespace retrieves the namespace to be used for testing.
// It first checks the "NAMESPACE" environment variable. If the variable is not set,
// it defaults to "inf-ext-e2e".
func getNamespace() string {
	namespace := os.Getenv("NAMESPACE")
	if namespace == "" {
		namespace = "infer-ext-e2e"
	}
	return namespace
}

// createNamespace creates a Kubernetes namespace if it does not already exist
// and the namespace name returned by `getNamespace()` is not "default".
func createNamespace(k8sClient client.Client) {
	name := getNamespace()
	if name == "default" {
		ginkgo.By("Skipping namespace creation as the target namespace is 'default'")
		return
	}

	ginkgo.By("Creating namespace: " + name)
	ns := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
	}

	err := k8sClient.Create(ctx, ns)
	gomega.Expect(err).NotTo(gomega.HaveOccurred(), "Failed to create namespace: %s", name)
}

// namespaceExists ensures that a specified namespace exists and is ready for use.
func namespaceExists(k8sClient client.Client, ns string) {
	ginkgo.By("Ensuring namespace exists: " + ns)
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Name: ns}, &corev1.Namespace{})
	}, existsTimeout, interval)
}

// createClient creates the client pod used for testing.
func createClient(k8sClient client.Client) {
	// Create the pod
	pod := newClientPod(nsName)
	ginkgo.By("Creating client pod: " + pod.Name)
	gomega.Expect(k8sClient.Create(ctx, pod)).To(gomega.Succeed())

	// Wait for the pod to exist.
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: getNamespace(), Name: pod.Name}, &corev1.Pod{})
	}, existsTimeout, interval)

	// Wait for the pod to be ready.
	testutils.PodReady(ctx, k8sClient, pod, readyTimeout, interval)
}

// createModel creates the model server resources used for testing.
func createModel(k8sClient client.Client) {
	// Create the model server secret. The "HF_TOKEN" environment variable must be set
	// to your Hugging Face access token (with access to Llama2 model).
	secret := newModelSecret(nsName)
	ginkgo.By("Creating model server secret: " + secret.Name)
	gomega.Expect(k8sClient.Create(ctx, secret)).To(gomega.Succeed())

	// Wait for the secret to exist before proceeding with test.
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: getNamespace(), Name: secret.Name}, &corev1.Secret{})
	}, existsTimeout, interval)

	// Create the model server deployment.
	deploy := newModelDeployment(nsName, modelServerName)
	ginkgo.By("Creating model server deployment: " + deploy.Name)
	gomega.Expect(k8sClient.Create(ctx, deploy)).To(gomega.Succeed())

	// Wait for the deployment to exist.
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: getNamespace(), Name: deploy.Name}, &appsv1.Deployment{})
	}, existsTimeout, interval)

	// Wait for the deployment to be available.
	testutils.DeploymentAvailable(ctx, k8sClient, deploy, modelReadyTimeout, interval)

	// Create the model server service.
	svc := newModelService(nsName, modelServerName)
	ginkgo.By("Creating model server service: " + svc.Name)
	gomega.Expect(k8sClient.Create(ctx, svc)).To(gomega.Succeed())

	// Wait for the service to exist.
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: getNamespace(), Name: svc.Name}, &corev1.Service{})
	}, existsTimeout, interval)
}

// createEnvoy creates the envoy proxy resources used for testing.
func createEnvoy(k8sClient client.Client) {
	cm := newEnvoyConfigMap(nsName, envoyName, envoyPort)
	ginkgo.By("Creating envoy proxy configmap: " + cm.Name)
	gomega.Expect(k8sClient.Create(ctx, cm)).To(gomega.Succeed())

	// Wait for the configmap to exist before proceeding with test.
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: getNamespace(), Name: cm.Name}, &corev1.ConfigMap{})
	}, existsTimeout, interval)

	// Create the deployment.
	deploy, err := newEnvoyDeployment(nsName)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	ginkgo.By("Creating envoy proxy deployment: " + deploy.Name)
	gomega.Expect(k8sClient.Create(ctx, deploy)).To(gomega.Succeed())

	// Wait for the deployment to exist.
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: getNamespace(), Name: deploy.Name}, &appsv1.Deployment{})
	}, existsTimeout, interval)

	// Wait for the deployment to be available.
	testutils.DeploymentAvailable(ctx, k8sClient, deploy, readyTimeout, interval)

	// Create the envoy service.
	svc, err := newEnvoyService(nsName, envoyName, envoyPort)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	ginkgo.By("Creating envoy proxy service: " + svc.Name)
	gomega.Expect(k8sClient.Create(ctx, svc)).To(gomega.Succeed())

	// Wait for the service to exist.
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: getNamespace(), Name: svc.Name}, &corev1.Service{})
	}, existsTimeout, interval)
}

// createInferExt creates the inference extension resources used for testing.
func createInferExt(k8sClient client.Client) {
	role := newInfExtClusterRole()
	ginkgo.By("Creating inference extension clusterrole: " + role.Name)
	gomega.Expect(k8sClient.Create(ctx, role)).To(gomega.Succeed())

	// Wait for the clusterrole to exist.
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Name: role.Name}, &rbacv1.ClusterRole{})
	}, existsTimeout, interval)

	binding := newInfExtClusterRoleBinding()
	ginkgo.By("Creating inference extension clusterrolebinding: " + binding.Name)
	gomega.Expect(k8sClient.Create(ctx, binding)).To(gomega.Succeed())

	// Wait for the clusterrolebinding to exist.
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Name: binding.Name}, &rbacv1.ClusterRoleBinding{})
	}, existsTimeout, interval)

	// Create an inferencepool (required by the inference extension readiness endpoint)
	infPool := newInferencePool(nsName)
	ginkgo.By("Creating inferencepool: " + infPool.Name)
	gomega.Expect(k8sClient.Create(ctx, infPool)).To(gomega.Succeed())

	// Wait for the inferencepool to exist.
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: getNamespace(), Name: infPool.Name}, &infextv1a1.InferencePool{})
	}, existsTimeout, interval)

	deploy := newInfExtDeployment(nsName, inferExtName, modelServerName, modelServerName)
	ginkgo.By("Creating inference extension deployment: " + deploy.Name)
	gomega.Expect(k8sClient.Create(ctx, deploy)).To(gomega.Succeed())

	// Wait for the deployment to exist.
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: getNamespace(), Name: deploy.Name}, &appsv1.Deployment{})
	}, existsTimeout, interval)

	// Wait for the deployment to be available.
	testutils.DeploymentAvailable(ctx, k8sClient, deploy, modelReadyTimeout, interval)

	// Create the service.
	svc := newInfExtService(nsName, inferExtName)
	ginkgo.By("Creating inference extension service: " + svc.Name)
	gomega.Expect(k8sClient.Create(ctx, svc)).To(gomega.Succeed())

	// Wait for the service to exist.
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: getNamespace(), Name: svc.Name}, &corev1.Service{})
	}, existsTimeout, interval)
}
