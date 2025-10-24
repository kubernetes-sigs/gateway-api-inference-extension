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
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apiextv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"

	infextv1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	infextv1a2 "sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/env"
	testutils "sigs.k8s.io/gateway-api-inference-extension/test/utils"
)

const (
	// defaultCurlTimeout is the default timeout for the curl command to get a response.
	defaultCurlTimeout = 30 * time.Second
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
	// metricsReaderSecretName is the name of the metrics reader secret which stores sa token to read epp metrics.
	metricsReaderSecretName = "inference-gateway-sa-metrics-reader-secret"
	// clientManifest is the manifest for the client test resources.
	clientManifest = "../../testdata/client.yaml"
	// modelServerSecretManifest is the manifest for the model server secret resource.
	modelServerSecretManifest = "../../testdata/model-secret.yaml"
	// xInferObjectiveManifest is the manifest for the inference model CRD with 'inference.networking.x-k8s.io' group.
	xInferObjectiveManifest = "../../../config/crd/bases/inference.networking.x-k8s.io_inferenceobjectives.yaml"
	// inferPoolManifest is the manifest for the inference pool CRD with 'inference.networking.k8s.io' group.
	inferPoolManifest = "../../../config/crd/bases/inference.networking.k8s.io_inferencepools.yaml"
	// inferExtManifestDefault is the manifest for the default inference extension test resources (single replica).
	inferExtManifestDefault = "../../testdata/inferencepool-e2e.yaml"
	// inferExtManifestLeaderElection is the manifest for the inference extension test resources with leader election enabled (3 replicas).
	inferExtManifestLeaderElection = "../../testdata/inferencepool-leader-election-e2e.yaml"
	// envoyManifest is the manifest for the envoy proxy test resources.
	envoyManifest = "../../testdata/envoy.yaml"
	// metricsRbacManifest is the manifest for the rbac resources for testing metrics.
	metricsRbacManifest = "../../testdata/metrics-rbac.yaml"
	// modelServerManifestFilepathEnvVar is the env var that holds absolute path to the manifest for the model server test resource.
	modelServerManifestFilepathEnvVar = "MANIFEST_PATH"
)

const e2eLeaderElectionEnabledEnvVar = "E2E_LEADER_ELECTION_ENABLED"

var (
	testConfig *testutils.TestConfig
	// Required for exec'ing in curl pod
	e2eImage              string
	leaderElectionEnabled bool
)

func TestAPIs(t *testing.T) {
	gomega.RegisterFailHandler(ginkgo.Fail)
	ginkgo.RunSpecs(t,
		"End To End Test Suite",
	)
}

var _ = ginkgo.BeforeSuite(func() {
	nsName := os.Getenv("E2E_NS")
	if nsName == "" {
		nsName = defaultNsName
	}
	testConfig = testutils.NewTestConfig(nsName)

	e2eImage = os.Getenv("E2E_IMAGE")
	gomega.Expect(e2eImage).NotTo(gomega.BeEmpty(), "E2E_IMAGE environment variable is not set")

	if os.Getenv(e2eLeaderElectionEnabledEnvVar) == "true" {
		leaderElectionEnabled = true
		ginkgo.By("Leader election test mode enabled via " + e2eLeaderElectionEnabledEnvVar)
	}

	ginkgo.By("Setting up the test suite")
	setupSuite()

	ginkgo.By("Creating test infrastructure")
	setupInfra()
})

func setupInfra() {
	// this function ensures ModelServer manifest path exists.
	// run this before createNs to fail fast in case it doesn't.
	modelServerManifestPath := readModelServerManifestPath()

	createNamespace(testConfig)

	modelServerManifestArray := getYamlsFromModelServerManifest(modelServerManifestPath)
	if strings.Contains(modelServerManifestArray[0], "hf-token") {
		createHfSecret(testConfig, modelServerSecretManifest)
	}
	crds := map[string]string{
		"inferenceobjectives.inference.networking.x-k8s.io": xInferObjectiveManifest,
		"inferencepools.inference.networking.k8s.io":        inferPoolManifest,
	}

	createCRDs(testConfig, crds)

	inferExtManifestPath := inferExtManifestDefault
	if leaderElectionEnabled {
		inferExtManifestPath = inferExtManifestLeaderElection
	}
	createInferExt(testConfig, inferExtManifestPath)
	createClient(testConfig, clientManifest)
	createEnvoy(testConfig, envoyManifest)
	createMetricsRbac(testConfig, metricsRbacManifest)
	// Run this step last, as it requires additional time for the model server to become ready.
	ginkgo.By("Creating model server resources from manifest: " + modelServerManifestPath)
	createModelServer(testConfig, modelServerManifestArray)
}

var _ = ginkgo.AfterSuite(func() {
	// If E2E_PAUSE_ON_EXIT is set, pause the test run before cleanup.
	// This is useful for debugging the state of the cluster after the test has run.
	if pauseStr := os.Getenv("E2E_PAUSE_ON_EXIT"); pauseStr != "" {
		ginkgo.By("Pausing before cleanup as requested by E2E_PAUSE_ON_EXIT=" + pauseStr)
		pauseDuration, err := time.ParseDuration(pauseStr)
		if err != nil {
			// If it's not a valid duration (e.g., "true"), just wait indefinitely.
			ginkgo.By("Invalid duration, pausing indefinitely. Press Ctrl+C to stop the test runner when you are done.")
			select {} // Block forever
		}
		ginkgo.By(fmt.Sprintf("Pausing for %v...", pauseDuration))
		time.Sleep(pauseDuration)
	}

	ginkgo.By("Performing global cleanup")
	cleanupResources()
})

// setupSuite initializes the test suite by setting up the Kubernetes client,
// loading required API schemes, and validating configuration.
func setupSuite() {
	err := clientgoscheme.AddToScheme(testConfig.Scheme)
	gomega.ExpectWithOffset(1, err).NotTo(gomega.HaveOccurred())

	err = apiextv1.AddToScheme(testConfig.Scheme)
	gomega.ExpectWithOffset(1, err).NotTo(gomega.HaveOccurred())

	err = infextv1a2.Install(testConfig.Scheme)
	gomega.ExpectWithOffset(1, err).NotTo(gomega.HaveOccurred())

	err = infextv1.Install(testConfig.Scheme)
	gomega.ExpectWithOffset(1, err).NotTo(gomega.HaveOccurred())

	testConfig.CreateCli()
}

func cleanupResources() {
	if testConfig.K8sClient == nil {
		return // could happen if BeforeSuite had an error
	}

	gomega.Expect(testutils.DeleteClusterResources(testConfig)).To(gomega.Succeed())
	gomega.Expect(testutils.DeleteNamespacedResources(testConfig)).To(gomega.Succeed())
}

func cleanupInferModelResources() {
	gomega.Expect(testutils.DeleteInferenceObjectiveResources(testConfig)).To(gomega.Succeed())
}

var (
	curlTimeout  = env.GetEnvDuration("CURL_TIMEOUT", defaultCurlTimeout, ginkgo.GinkgoLogr)
	curlInterval = defaultCurlInterval
)

func createNamespace(testConfig *testutils.TestConfig) {
	ginkgo.By("Creating e2e namespace: " + testConfig.NsName)
	obj := &corev1.Namespace{
		ObjectMeta: v1.ObjectMeta{
			Name: testConfig.NsName,
		},
	}
	err := testConfig.K8sClient.Create(testConfig.Context, obj)
	gomega.Expect(err).NotTo(gomega.HaveOccurred(), "Failed to create e2e test namespace")
}

// namespaceExists ensures that a specified namespace exists and is ready for use.
func namespaceExists(testConfig *testutils.TestConfig) {
	ginkgo.By("Ensuring namespace exists: " + testConfig.NsName)
	testutils.EventuallyExists(testConfig, func() error {
		return testConfig.K8sClient.Get(testConfig.Context,
			types.NamespacedName{Name: testConfig.NsName}, &corev1.Namespace{})
	})
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
	modelServerManifestArray := testutils.ReadYaml(modelServerManifestPath)
	gomega.Expect(modelServerManifestArray).NotTo(gomega.BeEmpty())
	return modelServerManifestArray
}

// createCRDs creates the Inference Extension CRDs used for testing.
func createCRDs(testConfig *testutils.TestConfig, crds map[string]string) {
	for _, path := range crds {
		ginkgo.By("Creating CRD resource from manifest: " + path)
		testutils.ApplyYAMLFile(testConfig, path)
	}
}

// createClient creates the client pod used for testing from the given filePath.
func createClient(testConfig *testutils.TestConfig, filePath string) {
	ginkgo.By("Creating client resources from manifest: " + filePath)
	testutils.ApplyYAMLFile(testConfig, filePath)
}

// createMetricsRbac creates the metrics RBAC resources from the manifest file.
func createMetricsRbac(testConfig *testutils.TestConfig, filePath string) {
	inManifests := testutils.ReadYaml(filePath)
	ginkgo.By("Replacing placeholder namespace with E2E_NS environment variable")
	outManifests := []string{}
	for _, m := range inManifests {
		outManifests = append(outManifests, strings.ReplaceAll(m, "$E2E_NS", testConfig.NsName))
	}
	ginkgo.By("Creating RBAC resources for scraping metrics from manifest: " + filePath)
	testutils.CreateObjsFromYaml(testConfig, outManifests)

	// wait for sa token to exist
	testutils.EventuallyExists(testConfig, func() error {
		token, err := getMetricsReaderToken(testConfig.K8sClient)
		if err != nil {
			return err
		}
		if len(token) == 0 {
			return errors.New("failed to get metrics reader token")
		}
		return nil
	})
}

// createModelServer creates the model server resources used for testing from the given filePaths.
func createModelServer(testConfig *testutils.TestConfig, modelServerManifestArray []string) {
	testutils.CreateObjsFromYaml(testConfig, modelServerManifestArray)
}

// createHfSecret read HF_TOKEN from env var and creates a secret that contains the access token.
func createHfSecret(testConfig *testutils.TestConfig, secretPath string) {
	ginkgo.By("Ensuring the HF_TOKEN environment variable is set")
	token := os.Getenv("HF_TOKEN")
	gomega.Expect(token).NotTo(gomega.BeEmpty(), "HF_TOKEN is not set")

	inManifests := testutils.ReadYaml(secretPath)
	ginkgo.By("Replacing placeholder secret data with HF_TOKEN environment variable")
	outManifests := []string{}
	for _, m := range inManifests {
		outManifests = append(outManifests, strings.Replace(m, "$HF_TOKEN", token, 1))
	}

	ginkgo.By("Creating model server secret resource")
	testutils.CreateObjsFromYaml(testConfig, outManifests)
}

// createEnvoy creates the envoy proxy resources used for testing from the given filePath.
func createEnvoy(testConfig *testutils.TestConfig, filePath string) {
	inManifests := testutils.ReadYaml(filePath)
	ginkgo.By("Replacing placeholder namespace with E2E_NS environment variable")
	outManifests := []string{}
	for _, m := range inManifests {
		outManifests = append(outManifests, strings.ReplaceAll(m, "$E2E_NS", testConfig.NsName))
	}

	ginkgo.By("Creating envoy proxy resources from manifest: " + filePath)
	testutils.CreateObjsFromYaml(testConfig, outManifests)
}

// createInferExt creates the inference extension resources used for testing from the given filePath.
func createInferExt(testConfig *testutils.TestConfig, filePath string) {
	inManifests := testutils.ReadYaml(filePath)
	ginkgo.By("Replacing placeholders with environment variables")
	outManifests := []string{}
	replacer := strings.NewReplacer(
		"$E2E_NS", testConfig.NsName,
		"$E2E_IMAGE", e2eImage,
	)
	for _, manifest := range inManifests {
		outManifests = append(outManifests, replacer.Replace(manifest))
	}

	ginkgo.By("Creating inference extension resources from manifest: " + filePath)
	testutils.CreateObjsFromYaml(testConfig, outManifests)

	// Wait for the deployment to exist.
	deploy := &appsv1.Deployment{
		ObjectMeta: v1.ObjectMeta{
			Name:      inferExtName,
			Namespace: testConfig.NsName,
		},
	}
	if leaderElectionEnabled {
		// With leader election enabled, only 1 replica will be "Ready" at any given time (the leader).
		testutils.DeploymentReadyReplicas(testConfig, deploy, 1)
	} else {
		testutils.DeploymentAvailable(testConfig, deploy)
	}
}
