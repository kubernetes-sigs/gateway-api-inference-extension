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

package epp

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apiextv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/config"
	infextv1a2 "sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	testutils "sigs.k8s.io/gateway-api-inference-extension/test/utils"
)

const (
	// defaultExistsTimeout is the default timeout for a resource to exist in the api server.
	defaultExistsTimeout = 30 * time.Second
	// defaultReadyTimeout is the default timeout for a resource to report a ready state.
	defaultReadyTimeout = 3 * time.Minute
	// defaultModelReadyTimeout is the default timeout for the model server deployment to report a ready state.
	defaultModelReadyTimeout = 10 * time.Minute
	// defaultCurlTimeout is the default timeout for the curl command to get a response.
	defaultCurlTimeout = 30 * time.Second
	// defaultInterval is the default interval to check if a resource exists or ready conditions.
	defaultInterval = time.Millisecond * 250
	// defaultCurlInterval is the default interval to run the test curl command.
	defaultCurlInterval = time.Second * 5
	// defaultNsName is the default name of the Namespace used for tests. Can override using the E2E_NS environment variable.
	defaultNsName = "inf-ext-e2e"
	// modelServerName is the name of the model server test resources.
	modelServerName = "vllm-llama3-8b-instruct"
	// modelName is the test model name.
	modelName = "food-review"
	// targetModelName is the target model name of the test model server.
	targetModelName = modelName + "-1"
	// envoyName is the name of the envoy proxy test resources.
	envoyName = "envoy"
	// envoyPort is the listener port number of the test envoy proxy.
	envoyPort = "8081"
	// inferExtName is the name of the inference extension test resources.
	inferExtName = "vllm-llama3-8b-instruct-epp"
	// clientManifest is the manifest for the client test resources.
	clientManifest = "../../testdata/client.yaml"
	// modelServerSecretManifest is the manifest for the model server secret resource.
	modelServerSecretManifest = "../../testdata/model-secret.yaml"
	// inferPoolManifest is the manifest for the inference pool CRD.
	inferPoolManifest = "../../../config/crd/bases/inference.networking.x-k8s.io_inferencepools.yaml"
	// inferModelManifest is the manifest for the inference model CRD.
	inferModelManifest = "../../../config/crd/bases/inference.networking.x-k8s.io_inferencemodels.yaml"
	// inferExtManifest is the manifest for the inference extension test resources.
	inferExtManifest = "../../testdata/inferencepool-e2e.yaml"
	// envoyManifest is the manifest for the envoy proxy test resources.
	envoyManifest = "../../testdata/envoy.yaml"
	// modelServerManifestFilepathEnvVar is the env var that holds absolute path to the manifest for the model server test resource.
	modelServerManifestFilepathEnvVar = "MANIFEST_PATH"
)

var (
	ctx context.Context
	cli client.Client
	// Required for exec'ing in curl pod
	kubeCli *kubernetes.Clientset
	scheme  = runtime.NewScheme()
	cfg     = config.GetConfigOrDie()
	nsName  string
)

func TestAPIs(t *testing.T) {
	gomega.RegisterFailHandler(ginkgo.Fail)
	ginkgo.RunSpecs(t,
		"End To End Test Suite",
	)
}

var _ = ginkgo.BeforeSuite(func() {
	nsName = os.Getenv("E2E_NS")
	if nsName == "" {
		nsName = defaultNsName
	}

	ginkgo.By("Setting up the test suite")
	setupSuite()

	ginkgo.By("Creating test infrastructure")
	setupInfra()
})

func setupInfra() {
	createNamespace(cli, nsName)

	modelServerManifestPath := readModelServerManifestPath()
	modelServerManifestArray := getYamlsFromModelServerManifest(modelServerManifestPath)
	if strings.Contains(modelServerManifestArray[0], "hf-token") {
		createHfSecret(cli, modelServerSecretManifest)
	}
	crds := map[string]string{
		"inferencepools.inference.networking.x-k8s.io":  inferPoolManifest,
		"inferencemodels.inference.networking.x-k8s.io": inferModelManifest,
	}

	createCRDs(cli, crds)
	createInferExt(cli, inferExtManifest)
	createClient(cli, clientManifest)
	createEnvoy(cli, envoyManifest)
	// Run this step last, as it requires additional time for the model server to become ready.
	createModelServer(cli, modelServerManifestArray, modelServerManifestPath)
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

	err = infextv1a2.Install(scheme)
	gomega.ExpectWithOffset(1, err).NotTo(gomega.HaveOccurred())

	cli, err = client.New(cfg, client.Options{Scheme: scheme})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	gomega.Expect(cli).NotTo(gomega.BeNil())

	kubeCli, err = kubernetes.NewForConfig(cfg)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	gomega.Expect(kubeCli).NotTo(gomega.BeNil())
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
	curlTimeout       = getTimeout("CURL_TIMEOUT", defaultCurlTimeout)
	interval          = defaultInterval
	curlInterval      = defaultCurlInterval
)

func createNamespace(k8sClient client.Client, ns string) {
	ginkgo.By("Creating e2e namespace: " + ns)
	obj := &corev1.Namespace{
		ObjectMeta: v1.ObjectMeta{
			Name: ns,
		},
	}
	err := k8sClient.Create(ctx, obj)
	gomega.Expect(err).NotTo(gomega.HaveOccurred(), "Failed to create e2e test namespace")
}

// namespaceExists ensures that a specified namespace exists and is ready for use.
func namespaceExists(k8sClient client.Client, ns string) {
	ginkgo.By("Ensuring namespace exists: " + ns)
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Name: ns}, &corev1.Namespace{})
	}, existsTimeout, interval)
}

// readModelServerManifestPath reads from env var the absolute filepath to model server deployment for testing.
func readModelServerManifestPath() string {
	ginkgo.By(fmt.Sprintf("Ensuring %s environment variable is set", modelServerManifestFilepathEnvVar))
	modelServerManifestFilepath := os.Getenv(modelServerManifestFilepathEnvVar)
	gomega.Expect(modelServerManifestFilepath).NotTo(gomega.BeEmpty(), modelServerManifestFilepathEnvVar+" is not set")
	return modelServerManifestFilepath
}

func getYamlsFromModelServerManifest(modelServerManifestPath string) []string {
	ginkgo.By("Ensuring the model server manifest points to an existing file")
	modelServerManifestArray := readYaml(modelServerManifestPath)
	gomega.Expect(modelServerManifestArray).NotTo(gomega.BeEmpty())
	return modelServerManifestArray
}

// createCRDs creates the Inference Extension CRDs used for testing.
func createCRDs(k8sClient client.Client, crds map[string]string) {
	for name, path := range crds {
		ginkgo.By("Creating CRD resource from manifest: " + path)
		applyYAMLFile(k8sClient, path)

		// Wait for the CRD to exist.
		crd := &apiextv1.CustomResourceDefinition{}
		testutils.EventuallyExists(ctx, func() error {
			return k8sClient.Get(ctx, types.NamespacedName{Name: name}, crd)
		}, existsTimeout, interval)

		// Wait for the CRD to be established.
		testutils.CRDEstablished(ctx, k8sClient, crd, readyTimeout, interval)
	}
}

// createClient creates the client pod used for testing from the given filePath.
func createClient(k8sClient client.Client, filePath string) {
	ginkgo.By("Creating client resources from manifest: " + filePath)
	applyYAMLFile(k8sClient, filePath)

	// Wait for the pod to exist.
	pod := &corev1.Pod{}
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: nsName, Name: "curl"}, pod)
	}, existsTimeout, interval)

	// Wait for the pod to be ready.
	testutils.PodReady(ctx, k8sClient, pod, readyTimeout, interval)
}

// createModelServer creates the model server resources used for testing from the given filePaths.
func createModelServer(k8sClient client.Client, modelServerManifestArray []string, deployPath string) {
	ginkgo.By("Creating model server resources from manifest: " + deployPath)
	createObjsFromYaml(k8sClient, modelServerManifestArray)

	// Wait for the deployment to exist.
	deploy := &appsv1.Deployment{}
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: nsName, Name: modelServerName}, deploy)
	}, existsTimeout, interval)

	// Wait for the deployment to be available.
	testutils.DeploymentAvailable(ctx, k8sClient, deploy, modelReadyTimeout, interval)
}

// createHfSecret read HF_TOKEN from env var and creates a secret that contains the access token.
func createHfSecret(k8sClient client.Client, secretPath string) {
	ginkgo.By("Ensuring the HF_TOKEN environment variable is set")
	token := os.Getenv("HF_TOKEN")
	gomega.Expect(token).NotTo(gomega.BeEmpty(), "HF_TOKEN is not set")

	inManifests := readYaml(secretPath)
	ginkgo.By("Replacing placeholder secret data with HF_TOKEN environment variable")
	outManifests := []string{}
	for _, m := range inManifests {
		outManifests = append(outManifests, strings.Replace(m, "$HF_TOKEN", token, 1))
	}

	ginkgo.By("Creating model server secret resource")
	createObjsFromYaml(k8sClient, outManifests)

	// Wait for the secret to exist before proceeding with test.
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: nsName, Name: "hf-token"}, &corev1.Secret{})
	}, existsTimeout, interval)
}

// createEnvoy creates the envoy proxy resources used for testing from the given filePath.
func createEnvoy(k8sClient client.Client, filePath string) {
	inManifests := readYaml(filePath)
	ginkgo.By("Replacing placeholder namespace with E2E_NS environment variable")
	outManifests := []string{}
	for _, m := range inManifests {
		outManifests = append(outManifests, strings.ReplaceAll(m, "$E2E_NS", nsName))
	}

	ginkgo.By("Creating envoy proxy resources from manifest: " + filePath)
	createObjsFromYaml(k8sClient, outManifests)

	// Wait for the configmap to exist before proceeding with test.
	cfgMap := &corev1.ConfigMap{}
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: nsName, Name: envoyName}, cfgMap)
	}, existsTimeout, interval)

	// Wait for the deployment to exist.
	deploy := &appsv1.Deployment{}
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: nsName, Name: envoyName}, deploy)
	}, existsTimeout, interval)

	// Wait for the deployment to be available.
	testutils.DeploymentAvailable(ctx, k8sClient, deploy, readyTimeout, interval)

	// Wait for the service to exist.
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: nsName, Name: envoyName}, &corev1.Service{})
	}, existsTimeout, interval)
}

// createInferExt creates the inference extension resources used for testing from the given filePath.
func createInferExt(k8sClient client.Client, filePath string) {
	inManifests := readYaml(filePath)
	ginkgo.By("Replacing placeholder namespace with E2E_NS environment variable")
	outManifests := []string{}
	for _, m := range inManifests {
		outManifests = append(outManifests, strings.ReplaceAll(m, "$E2E_NS", nsName))
	}

	ginkgo.By("Creating inference extension resources from manifest: " + filePath)
	createObjsFromYaml(k8sClient, outManifests)

	// Wait for the clusterrole to exist.
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Name: "pod-read"}, &rbacv1.ClusterRole{})
	}, existsTimeout, interval)

	// Wait for the clusterrolebinding to exist.
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Name: "pod-read-binding"}, &rbacv1.ClusterRoleBinding{})
	}, existsTimeout, interval)

	// Wait for the deployment to exist.
	deploy := &appsv1.Deployment{}
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: nsName, Name: inferExtName}, deploy)
	}, existsTimeout, interval)

	// Wait for the deployment to be available.
	testutils.DeploymentAvailable(ctx, k8sClient, deploy, modelReadyTimeout, interval)

	// Wait for the service to exist.
	testutils.EventuallyExists(ctx, func() error {
		return k8sClient.Get(ctx, types.NamespacedName{Namespace: nsName, Name: inferExtName}, &corev1.Service{})
	}, existsTimeout, interval)
}

// applyYAMLFile reads a file containing YAML (possibly multiple docs)
// and applies each object to the cluster.
func applyYAMLFile(k8sClient client.Client, filePath string) {
	// Create the resources from the manifest file
	createObjsFromYaml(k8sClient, readYaml(filePath))
}

func readYaml(filePath string) []string {
	ginkgo.By("Reading YAML file: " + filePath)
	yamlBytes, err := os.ReadFile(filePath)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	// Split multiple docs, if needed
	return strings.Split(string(yamlBytes), "\n---")
}

func createObjsFromYaml(k8sClient client.Client, docs []string) {
	// For each doc, decode and create
	decoder := serializer.NewCodecFactory(scheme).UniversalDeserializer()
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
			err := scheme.Convert(obj, unstrObj, nil)
			gomega.Expect(err).NotTo(gomega.HaveOccurred())
		}

		unstrObj.SetNamespace(nsName)

		// Create the object
		err := k8sClient.Create(ctx, unstrObj)
		gomega.Expect(err).NotTo(gomega.HaveOccurred(),
			"Failed to create object from YAML")
	}
}
