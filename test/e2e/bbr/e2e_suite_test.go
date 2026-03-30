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

package bbr

import (
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
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client"

	testutils "sigs.k8s.io/gateway-api-inference-extension/test/utils"
)

const (
	defaultNsName       = "bbr-e2e"
	deepseekModelServer = "vllm-deepseek-r1"
	llamaModelServer    = "vllm-llama3-8b-instruct"
	bbrName             = "bbr"
	envoyName           = "envoy"
	envoyPort           = "8081"
	clientManifest      = "../../testdata/client.yaml"
	bbrManifest         = "../../testdata/bbr-deployment-e2e.yaml"
	envoyManifest       = "../../testdata/envoy-bbr.yaml"
	bbrImageEnvVar      = "BBR_E2E_IMAGE"
	manifestPathEnvVar  = "MANIFEST_PATH"
)

var (
	testConfig *testutils.TestConfig
	bbrImage   string
)

func TestBBR(t *testing.T) {
	gomega.RegisterFailHandler(ginkgo.Fail)
	ginkgo.RunSpecs(t,
		"BBR End To End Test Suite",
	)
}

var _ = ginkgo.BeforeSuite(func() {
	nsName := os.Getenv("E2E_NS")
	if nsName == "" {
		nsName = defaultNsName
	}
	testConfig = testutils.NewTestConfig(nsName, "")

	bbrImage = os.Getenv(bbrImageEnvVar)
	gomega.Expect(bbrImage).NotTo(gomega.BeEmpty(), bbrImageEnvVar+" environment variable is not set")

	ginkgo.By("Setting up the test suite")
	setupSuite()

	ginkgo.By("Creating test infrastructure")
	setupInfra()
})

func setupSuite() {
	err := clientgoscheme.AddToScheme(testConfig.Scheme)
	gomega.ExpectWithOffset(1, err).NotTo(gomega.HaveOccurred())

	testConfig.CreateCli()
}

func setupInfra() {
	modelServerManifestPath := readModelServerManifestPath()
	createNamespace(testConfig)

	ginkgo.By("Deploying DeepSeek model server from MANIFEST_PATH")
	modelServerManifestArray := getYamlsFromModelServerManifest(modelServerManifestPath)
	createModelServer(testConfig, modelServerManifestArray)

	ginkgo.By("Deploying BBR, Llama model server, ConfigMaps, and Services")
	createBBR(testConfig, bbrManifest)

	ginkgo.By("Waiting for Llama model server to be available")
	waitForDeployment(testConfig, llamaModelServer)

	ginkgo.By("Waiting for DeepSeek model server to be available")
	waitForDeployment(testConfig, deepseekModelServer)

	createClient(testConfig, clientManifest)
	createEnvoy(testConfig, envoyManifest)

	ginkgo.By("Waiting for Envoy proxy to be available")
	waitForDeployment(testConfig, envoyName)
}

var _ = ginkgo.AfterSuite(func() {
	if pauseStr := os.Getenv("E2E_PAUSE_ON_EXIT"); pauseStr != "" {
		ginkgo.By("Pausing before cleanup as requested by E2E_PAUSE_ON_EXIT=" + pauseStr)
		pauseDuration, err := time.ParseDuration(pauseStr)
		if err != nil {
			ginkgo.By("Invalid duration, pausing indefinitely. Press Ctrl+C to stop.")
			select {}
		}
		ginkgo.By(fmt.Sprintf("Pausing for %v...", pauseDuration))
		time.Sleep(pauseDuration)
	}

	ginkgo.By("Performing global cleanup")
	cleanupResources()
})

func cleanupResources() {
	if testConfig.K8sClient == nil {
		return
	}

	deleteBBRClusterResources()
	deleteNamespace()
}

func deleteBBRClusterResources() {
	ctx := testConfig.Context
	c := testConfig.K8sClient

	for _, obj := range []client.Object{
		&rbacv1.ClusterRoleBinding{ObjectMeta: v1.ObjectMeta{Name: "bbr-auth-reviewer-binding"}},
		&rbacv1.ClusterRole{ObjectMeta: v1.ObjectMeta{Name: "bbr-auth-reviewer"}},
	} {
		_ = c.Delete(ctx, obj)
	}
}

func deleteNamespace() {
	if testConfig.NsName == "default" {
		return
	}
	ns := &corev1.Namespace{ObjectMeta: v1.ObjectMeta{Name: testConfig.NsName}}
	_ = testConfig.K8sClient.Delete(testConfig.Context, ns)
}

func createNamespace(tc *testutils.TestConfig) {
	ginkgo.By("Creating e2e namespace: " + tc.NsName)
	obj := &corev1.Namespace{
		ObjectMeta: v1.ObjectMeta{
			Name: tc.NsName,
		},
	}
	err := tc.K8sClient.Create(tc.Context, obj)
	gomega.Expect(err).NotTo(gomega.HaveOccurred(), "Failed to create e2e test namespace")
}

func namespaceExists(tc *testutils.TestConfig) {
	ginkgo.By("Ensuring namespace exists: " + tc.NsName)
	testutils.EventuallyExists(tc, func() error {
		return tc.K8sClient.Get(tc.Context,
			types.NamespacedName{Name: tc.NsName}, &corev1.Namespace{})
	})
}

func readModelServerManifestPath() string {
	ginkgo.By(fmt.Sprintf("Ensuring %s environment variable is set", manifestPathEnvVar))
	path := os.Getenv(manifestPathEnvVar)
	gomega.Expect(path).NotTo(gomega.BeEmpty(), manifestPathEnvVar+" is not set")
	return path
}

func getYamlsFromModelServerManifest(path string) []string {
	ginkgo.By("Ensuring the model server manifest points to an existing file")
	manifests := testutils.ReadYaml(path)
	gomega.Expect(manifests).NotTo(gomega.BeEmpty())
	return manifests
}

func createClient(tc *testutils.TestConfig, filePath string) {
	ginkgo.By("Creating client resources from manifest: " + filePath)
	testutils.ApplyYAMLFile(tc, filePath)
}

func createModelServer(tc *testutils.TestConfig, manifestArray []string) {
	ginkgo.By("Creating model server resources")
	testutils.CreateObjsFromYaml(tc, manifestArray)
}

func createBBR(tc *testutils.TestConfig, filePath string) {
	inManifests := testutils.ReadYaml(filePath)
	ginkgo.By("Replacing placeholders with environment variables")
	outManifests := make([]string, 0, len(inManifests))
	replacer := strings.NewReplacer(
		"$E2E_NS", tc.NsName,
		"$BBR_E2E_IMAGE", bbrImage,
	)
	for _, manifest := range inManifests {
		outManifests = append(outManifests, replacer.Replace(manifest))
	}

	ginkgo.By("Creating BBR resources from manifest: " + filePath)
	testutils.CreateObjsFromYaml(tc, outManifests)

	deploy := &appsv1.Deployment{
		ObjectMeta: v1.ObjectMeta{
			Name:      bbrName,
			Namespace: tc.NsName,
		},
	}
	testutils.DeploymentAvailable(tc, deploy)
}

func createEnvoy(tc *testutils.TestConfig, filePath string) {
	inManifests := testutils.ReadYaml(filePath)
	ginkgo.By("Replacing placeholder namespace with E2E_NS environment variable")
	outManifests := make([]string, 0, len(inManifests))
	for _, m := range inManifests {
		outManifests = append(outManifests, strings.ReplaceAll(m, "$E2E_NS", tc.NsName))
	}

	ginkgo.By("Creating envoy proxy resources from manifest: " + filePath)
	testutils.CreateObjsFromYaml(tc, outManifests)
}

func waitForDeployment(tc *testutils.TestConfig, name string) {
	deploy := &appsv1.Deployment{
		ObjectMeta: v1.ObjectMeta{
			Name:      name,
			Namespace: tc.NsName,
		},
	}
	testutils.DeploymentAvailable(tc, deploy)
}
