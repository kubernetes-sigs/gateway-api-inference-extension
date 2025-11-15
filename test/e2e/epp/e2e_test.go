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
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"

	appsv1 "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"
	testutils "sigs.k8s.io/gateway-api-inference-extension/test/utils"
)

var _ = ginkgo.Describe("InferencePool", func() {
	var infObjectiveMultiPort *v1alpha2.InferenceObjective

	ginkgo.BeforeEach(func() {
		ginkgo.By("Waiting for the namespace to exist.")
		ginkgo.GinkgoWriter.Printf("Ensuring test namespace '%s' exists.\n", testConfig.NsName)
		namespaceExists(testConfig)

		ginkgo.By("Creating an InferenceObjective resource")
		infObjectiveMultiPort = newInferenceObjective(testConfig.NsName)
		gomega.Expect(testConfig.K8sClient.Create(testConfig.Context, infObjectiveMultiPort)).To(gomega.Succeed())

		ginkgo.By("Ensuring the InferenceObjective resource exists in the namespace")
		gomega.Eventually(func() error {
			return testConfig.K8sClient.Get(testConfig.Context, types.NamespacedName{Namespace: infObjectiveMultiPort.Namespace, Name: infObjectiveMultiPort.Name}, infObjectiveMultiPort)
		}, testConfig.ExistsTimeout, testConfig.Interval).Should(gomega.Succeed())
	})

	ginkgo.AfterEach(func() {
		ginkgo.By("Deleting the InferenceObjective test resource.")
		cleanupInferModelResources()
		gomega.Eventually(func() error {
			err := testConfig.K8sClient.Get(testConfig.Context, types.NamespacedName{Namespace: infObjectiveMultiPort.Namespace, Name: infObjectiveMultiPort.Name}, infObjectiveMultiPort)
			if err == nil {
				return errors.New("InferenceObjective resource still exists")
			}
			if !k8serrors.IsNotFound(err) {
				return nil
			}
			return nil
		}, testConfig.ExistsTimeout, testConfig.Interval).Should(gomega.Succeed())
	})

	ginkgo.When("The Inference Extension is running", func() {
		ginkgo.It("Should route traffic to target model servers", func() {
			verifyTrafficRouting()
		})

		ginkgo.It("Should expose EPP metrics after generating traffic", func() {
			verifyMetrics()
		})
	})

	ginkgo.When("When configured for vLLM Data Parallelism", func() {
		// Define variables here so they are accessible in BeforeEach, It, and AfterEach
		var (
			inferencePoolMultiTarget *v1.InferencePool
			backendMultiPort         *appsv1.Deployment
			backendService           *corev1.Service
		)

		ginkgo.BeforeEach(func() {
			// 1. CLEANUP old objects that may exist from previous runs: objective, deployment, service, pool
			ginkgo.By("Deleting the default/global InferenceObjective")
			// TODO: This is not necessary. Can just make both Objectives siblings rather than deleting the default one.
			// This may be causing the thrashing.
			defaultObj := &v1alpha2.InferenceObjective{
				ObjectMeta: metav1.ObjectMeta{Name: "inferenceobjective-sample", Namespace: testConfig.NsName},
			}
			deleteResourceAndWait(testConfig, defaultObj)

			ginkgo.By("Ensuring vllm-multi-port-backend Deployment is deleted from previous runs")
			cleanupDeploy := &appsv1.Deployment{
				ObjectMeta: metav1.ObjectMeta{Name: "vllm-multi-port-backend", Namespace: testConfig.NsName},
			}
			deleteResourceAndWait(testConfig, cleanupDeploy)

			ginkgo.By("Ensuring vllm-multi-port-service is deleted from previous runs")
			cleanupSvc := &corev1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "vllm-multi-port-service", Namespace: testConfig.NsName},
			}
			deleteResourceAndWait(testConfig, cleanupSvc)

			ginkgo.By("Ensuring pool is deleted from previous runs")
			poolName := "vllm-llama3-8b-instruct"
			prevPool := &v1.InferencePool{
				ObjectMeta: metav1.ObjectMeta{Name: poolName, Namespace: testConfig.NsName},
			}
			deleteResourceAndWait(testConfig, prevPool)

			// 2a. CREATE backend deployment
			ginkgo.By("Creating multi-port backend")
			backendMultiPort = createBackendMultiPortDeployment(testConfig, 8000, 8)
			gomega.Expect(testConfig.K8sClient.Create(testConfig.Context, backendMultiPort)).To(gomega.Succeed())
			testutils.DeploymentReadyReplicas(testConfig, backendMultiPort, 1)

			// 2b. CREATE backend service
			backendService = createBackendMultiPortService(testConfig)
			gomega.Expect(testConfig.K8sClient.Create(testConfig.Context, backendService)).To(gomega.Succeed())

			// 2c. CREATE New pool with the original name from the previous InferencePool
			inferencePoolMultiTarget = &v1.InferencePool{
				ObjectMeta: metav1.ObjectMeta{
					Name:      poolName,
					Namespace: testConfig.NsName,
				},
				Spec: v1.InferencePoolSpec{
					Selector: v1.LabelSelector{
						MatchLabels: map[v1.LabelKey]v1.LabelValue{"app": "vllm-multi-port-backend"},
					},
					EndpointPickerRef: v1.EndpointPickerRef{
						Name: v1.ObjectName(inferExtName),
						Port: &v1.Port{
							Number: v1.PortNumber(9002),
						},
					},
					TargetPorts: []v1.Port{
						{Number: v1.PortNumber(8000)}, {Number: v1.PortNumber(8001)},
						{Number: v1.PortNumber(8002)}, {Number: v1.PortNumber(8003)},
						{Number: v1.PortNumber(8004)}, {Number: v1.PortNumber(8005)},
						{Number: v1.PortNumber(8006)}, {Number: v1.PortNumber(8007)},
					},
				},
			}

			ginkgo.By("Creating the multi-port InferencePool (reusing default name)")
			gomega.Expect(testConfig.K8sClient.Create(testConfig.Context, inferencePoolMultiTarget)).To(gomega.Succeed())

			// 2d. CREATE Inference custom objective that points to the new pool
			ginkgo.By("Creating multi-port InferenceObjective")
			infObjectiveMultiPort = createMultiPortInferenceObjective(testConfig, inferencePoolMultiTarget)
			gomega.Expect(testConfig.K8sClient.Create(testConfig.Context, infObjectiveMultiPort)).To(gomega.Succeed())

			// 3. WAIT for Objective to be Accepted
			ginkgo.By("Waiting for InferenceObjective to be Accepted")
			gomega.Eventually(func() error {
				err := testConfig.K8sClient.Get(testConfig.Context, client.ObjectKeyFromObject(infObjectiveMultiPort), infObjectiveMultiPort)
				if err != nil {
					return err
				}
				ginkgo.GinkgoWriter.Printf("Current Objective Conditions: %+v\n", infObjectiveMultiPort.Status.Conditions)
				time.Sleep(5 * time.Minute)
				for _, cond := range infObjectiveMultiPort.Status.Conditions {
					if cond.Type == "Accepted" && cond.Status == metav1.ConditionTrue {
						return nil
					}
				}
				return fmt.Errorf("objective not accepted")
			}, testConfig.ReadyTimeout, testConfig.Interval).Should(gomega.Succeed())
		})

		ginkgo.AfterEach(func() {
			ginkgo.By("Cleaning up multi-port test resources")
			if infObjectiveMultiPort != nil {
				deleteResourceAndWait(testConfig, infObjectiveMultiPort)
			}
			if inferencePoolMultiTarget != nil {
				deleteResourceAndWait(testConfig, inferencePoolMultiTarget)
			}
			if backendService != nil {
				deleteResourceAndWait(testConfig, backendService)
			}
			if backendMultiPort != nil {
				deleteResourceAndWait(testConfig, backendMultiPort)
			}
		})

		ginkgo.It("Should route traffic to all data parallel ranks within the pod", func() {
			expectedPorts := map[string]bool{
				"8000": false, "8001": false, "8002": false, "8003": false,
				"8004": false, "8005": false, "8006": false, "8007": false,
			}

			gomega.Eventually(func(g gomega.Gomega) {
				curlCmd := []string{
					"curl", "-s", "--max-time", "5",
					fmt.Sprintf("http://%s.%s.svc:%s/v1/completions", envoyName, testConfig.NsName, envoyPort),
					"-H", "Content-Type: application/json",
					"-H", "model: vllm-multi-port-objective",
					"-d", `{"model": "vllm-multi-port-objective", "prompt": "test", "max_tokens": 1}`,
				}

				resp, err := testutils.ExecCommandInPod(testConfig, "curl", "curl", curlCmd)
				g.Expect(err).NotTo(gomega.HaveOccurred())

				// TODO: The backend needs to return JSON (OpenAI) not plain text.
				for port := range expectedPorts {
					if strings.Contains(resp, fmt.Sprintf("Handled by port: %s", port)) {
						if !expectedPorts[port] {
							ginkgo.GinkgoWriter.Printf("Hit port: %s\n", port)
						}
						expectedPorts[port] = true
					}
				}

				for _, hit := range expectedPorts {
					if !hit {
						g.Expect(hit).To(gomega.BeTrue(), "Still waiting to hit all ports. Current state: %v", expectedPorts)
					}
				}
			}, testConfig.ReadyTimeout, testConfig.Interval).Should(gomega.Succeed())
		})
	})
})

func newInferenceObjective(ns string) *v1alpha2.InferenceObjective {
	return testutils.MakeModelWrapper(types.NamespacedName{Name: "inferenceobjective-sample", Namespace: ns}).
		SetPriority(2).
		SetPoolRef(modelServerName).
		Obj()
}

func verifyTrafficRouting() {
	ginkgo.By("Verifying traffic routing")
	for _, t := range []struct {
		api              string
		promptOrMessages any
	}{
		{
			api:              "/completions",
			promptOrMessages: "Write as if you were a critic: San Francisco",
		},
		{
			api: "/chat/completions",
			promptOrMessages: []map[string]any{
				{
					"role":    "user",
					"content": "Write as if you were a critic: San Francisco",
				},
			},
		},
		{
			api: "/chat/completions",
			promptOrMessages: []map[string]any{
				{
					"role":    "user",
					"content": "Write as if you were a critic: San Francisco",
				},
				{"role": "assistant", "content": "Okay, let's see..."},
				{"role": "user", "content": "Now summarize your thoughts."},
			},
		},
	} {
		ginkgo.By(fmt.Sprintf("Verifying connectivity through the inference extension with %s api and prompt/messages: %v", t.api, t.promptOrMessages))

		var expected []string
		expected = append(expected, targetModelName)
		curlCmd := getCurlCommand(envoyName, testConfig.NsName, envoyPort, modelName, curlTimeout, t.api, t.promptOrMessages, false)

		gomega.Eventually(func() error {
			actual := make(map[string]int)
			resp, err := testutils.ExecCommandInPod(testConfig, "curl", "curl", curlCmd)
			if err != nil {
				ginkgo.GinkgoWriter.Printf("Retrying... Exec error: %v\n", err)
				return err
			}
			if !strings.Contains(resp, "200 OK") {
				return fmt.Errorf("did not get 200 OK: %s", resp)
			}
			for _, m := range expected {
				if strings.Contains(resp, m) {
					actual[m] = 0
				}
			}
			var got []string
			for m := range actual {
				got = append(got, m)
			}
			if !cmp.Equal(got, expected, cmpopts.SortSlices(func(a, b string) bool { return a < b })) {
				ginkgo.GinkgoWriter.Printf("Retrying... Mismatch. Got %v, Expected %v\n", got, expected)
				return fmt.Errorf("actual (%v) != expected (%v); resp=%q", got, expected, resp)
			}
			return nil
		}, testConfig.ReadyTimeout, curlInterval).Should(gomega.Succeed())
	}
}

func verifyMetrics() {
	ginkgo.By("Verifying metrics exposure")
	expectedMetrics := []string{
		"inference_objective_request_total",
		"inference_objective_request_error_total",
		"inference_objective_request_duration_seconds",
		"inference_objective_normalized_time_per_output_token_seconds",
		"inference_objective_request_sizes",
		"inference_objective_response_sizes",
		"inference_objective_input_tokens",
		"inference_objective_output_tokens",
		"inference_pool_average_kv_cache_utilization",
		"inference_pool_average_queue_size",
		"inference_pool_per_pod_queue_size",
		"inference_objective_running_requests",
		"inference_pool_ready_pods",
		"inference_extension_info",
	}

	ginkgo.By("Generating traffic through the inference extension")
	curlCmd := getCurlCommand(envoyName, testConfig.NsName, envoyPort, modelName, curlTimeout, "/completions", "Write as if you were a critic: San Francisco", true)

	for i := 0; i < 5; i++ {
		_, err := testutils.ExecCommandInPod(testConfig, "curl", "curl", curlCmd)
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
	}

	curlCmd[len(curlCmd)-1] = "invalid input"
	for i := 0; i < 5; i++ {
		_, err := testutils.ExecCommandInPod(testConfig, "curl", "curl", curlCmd)
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
	}

	ginkgo.By("Scraping metrics from the EPP endpoint")
	podIP := findReadyPod().Status.PodIP

	token := ""
	gomega.Eventually(func(g gomega.Gomega) {
		t, err := getMetricsReaderToken(testConfig.K8sClient)
		g.Expect(err).NotTo(gomega.HaveOccurred())
		g.Expect(t).NotTo(gomega.BeEmpty())
		token = t
	}, testConfig.ExistsTimeout, testConfig.Interval).Should(gomega.Succeed())

	metricScrapeCmd := getMetricsScrapeCommand(podIP, token)

	ginkgo.By("Verifying that all expected metrics are present.")
	gomega.Eventually(func() error {
		resp, err := testutils.ExecCommandInPod(testConfig, "curl", "curl", metricScrapeCmd)
		if err != nil {
			return err
		}
		if !strings.Contains(resp, "200 OK") {
			return fmt.Errorf("did not get 200 OK: %s", resp)
		}
		for _, metric := range expectedMetrics {
			if !strings.Contains(resp, metric) {
				return fmt.Errorf("expected metric %s not found in metrics output", metric)
			}
		}
		return nil
	}, testConfig.ReadyTimeout, curlInterval).Should(gomega.Succeed())
}

func getMetricsReaderToken(k8sClient client.Client) (string, error) {
	secret := &corev1.Secret{}
	err := k8sClient.Get(testConfig.Context, types.NamespacedName{Namespace: testConfig.NsName, Name: metricsReaderSecretName}, secret)
	if err != nil {
		return "", err
	}
	return string(secret.Data["token"]), nil
}

func findReadyPod() *corev1.Pod {
	var readyPod *corev1.Pod
	gomega.Eventually(func(g gomega.Gomega) {
		podList := &corev1.PodList{}
		err := testConfig.K8sClient.List(testConfig.Context, podList, client.InNamespace(testConfig.NsName), client.MatchingLabels{"app": inferExtName})
		g.Expect(err).NotTo(gomega.HaveOccurred())

		foundReadyPod := false
		for i := range podList.Items {
			pod := &podList.Items[i]
			for _, cond := range pod.Status.Conditions {
				if cond.Type == corev1.PodReady && cond.Status == corev1.ConditionTrue {
					g.Expect(pod.Status.PodIP).NotTo(gomega.BeEmpty(), "Ready pod must have an IP")
					readyPod = pod
					foundReadyPod = true
					break
				}
			}
			if foundReadyPod {
				break
			}
		}
		g.Expect(foundReadyPod).To(gomega.BeTrue(), "No ready EPP pod found")
	}, testConfig.ReadyTimeout, testConfig.Interval).Should(gomega.Succeed())
	return readyPod
}

func getMetricsScrapeCommand(podIP, token string) []string {
	return []string{
		"curl", "-i", "--max-time", strconv.Itoa((int)(curlTimeout.Seconds())),
		"-H", "Authorization: Bearer " + token, fmt.Sprintf("http://%s:%d/metrics", podIP, 9090),
	}
}

func getCurlCommand(name, ns, port, model string, timeout time.Duration, api string, promptOrMessages any, streaming bool) []string {
	body := map[string]any{
		"model":       model,
		"max_tokens":  100,
		"temperature": 0,
	}
	body["model"] = model
	switch api {
	case "/completions":
		body["prompt"] = promptOrMessages
	case "/chat/completions":
		body["messages"] = promptOrMessages
	}
	if streaming {
		body["stream"] = true
		body["stream_options"] = map[string]any{
			"include_usage": true,
		}
	}
	b, err := json.Marshal(body)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	return []string{
		"curl",
		"-i",
		"--max-time",
		strconv.Itoa((int)(timeout.Seconds())),
		fmt.Sprintf("%s.%s.svc:%s/v1%s", name, ns, port, api),
		"-H",
		"Content-Type: application/json",
		"-H",
		fmt.Sprintf("%v: inferenceobjective-sample", metadata.ObjectiveKey),
		"-H",
		fmt.Sprintf("%v: %s", metadata.ModelNameRewriteKey, targetModelName),
		"-d",
		string(b),
	}
}

// createMultiPortInferenceObjective creates an InferenceObjective for the multi-port InferencePool.
func createMultiPortInferenceObjective(testConfig *testutils.TestConfig, pool *v1.InferencePool) *v1alpha2.InferenceObjective {
	return &v1alpha2.InferenceObjective{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "vllm-multi-port-objective",
			Namespace: testConfig.NsName,
		},
		Spec: v1alpha2.InferenceObjectiveSpec{
			PoolRef: v1alpha2.PoolObjectReference{
				// Ensure this uses the correct Group constant from the API package
				Group: v1alpha2.Group(v1.GroupVersion.Group),
				Kind:  "InferencePool",
				Name:  v1alpha2.ObjectName(pool.Name),
			},
		},
	}
}

func createBackendMultiPortDeployment(testConfig *testutils.TestConfig, startPort int32, portCount int) *appsv1.Deployment {
	backendLabels := map[string]string{"app": "vllm-multi-port-backend"}

	containerPorts := make([]corev1.ContainerPort, portCount)
	for i := 0; i < portCount; i++ {
		containerPorts[i] = corev1.ContainerPort{
			Name:          fmt.Sprintf("http-%d", 8000+i),
			ContainerPort: int32(8000 + i),
		}
	}

	args := []string{
		strconv.Itoa(int(startPort)),
		strconv.Itoa(portCount),
	}

	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "vllm-multi-port-backend",
			Namespace: testConfig.NsName,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: ptr.To(int32(1)),
			Selector: &metav1.LabelSelector{MatchLabels: backendLabels},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: backendLabels},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:            "multi-port-server",
							Image:           "vllm-dynamic-backend:local",
							ImagePullPolicy: corev1.PullIfNotPresent,
							Args:            args,
							Ports:           containerPorts,
						},
					},
				},
			},
		},
	}
}

// TODO: The image above is not generalizable and will not pass CI/CD.

func createBackendMultiPortService(testConfig *testutils.TestConfig) *corev1.Service {
	backendLabels := map[string]string{"app": "vllm-multi-port-backend"}
	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "vllm-multi-port-service",
			Namespace: testConfig.NsName,
			Labels:    backendLabels,
		},
		Spec: corev1.ServiceSpec{
			Selector: backendLabels,
			Ports: []corev1.ServicePort{
				{Name: "http-8000", Port: 8000, TargetPort: intstr.FromInt(8000)},
				{Name: "http-8001", Port: 8001, TargetPort: intstr.FromInt(8001)},
				{Name: "http-8002", Port: 8002, TargetPort: intstr.FromInt(8002)},
				{Name: "http-8003", Port: 8003, TargetPort: intstr.FromInt(8003)},
				{Name: "http-8004", Port: 8004, TargetPort: intstr.FromInt(8004)},
				{Name: "http-8005", Port: 8005, TargetPort: intstr.FromInt(8005)},
				{Name: "http-8006", Port: 8006, TargetPort: intstr.FromInt(8006)},
				{Name: "http-8007", Port: 8007, TargetPort: intstr.FromInt(8007)},
			},
		},
	}
}

func deleteResourceAndWait(tc *testutils.TestConfig, obj client.Object) {
	if obj == nil {
		return
	}
	name := client.ObjectKeyFromObject(obj)
	kind := obj.GetObjectKind().GroupVersionKind().Kind
	ginkgo.GinkgoWriter.Printf("Attempting to delete %s %s/%s\n", kind, name.Namespace, name.Name)

	propagationPolicy := metav1.DeletePropagationForeground
	err := tc.K8sClient.Delete(tc.Context, obj,
		client.GracePeriodSeconds(0),
		client.PropagationPolicy(propagationPolicy),
	)
	if err != nil {
		if k8serrors.IsNotFound(err) {
			ginkgo.GinkgoWriter.Printf("%s %s/%s not found during cleanup, may have been deleted already.\n", kind, name.Namespace, name.Name)
			return
		}
		gomega.Expect(err).To(gomega.Succeed(), "Failed to initiate deletion of %s %s/%s", kind, name.Namespace, name.Name)
	} else {
		ginkgo.GinkgoWriter.Printf("Deletion initiated for %s %s/%s. Waiting for it to be fully removed...\n", kind, name.Namespace, name.Name)
		gomega.Eventually(func() error {
			key := client.ObjectKeyFromObject(obj)
			tempObj := obj.DeepCopyObject().(client.Object)
			err := tc.K8sClient.Get(tc.Context, key, tempObj)
			if k8serrors.IsNotFound(err) {
				ginkgo.GinkgoWriter.Printf("%s %s/%s is now fully deleted.\n", kind, name.Namespace, name.Name)
				return nil
			}
			if err != nil {
				ginkgo.GinkgoWriter.Printf("Error while waiting for deletion of %s %s/%s: %v\n", kind, name.Namespace, name.Name, err)
				return err
			}
			return fmt.Errorf("%s %s/%s still exists (deletionTimestamp=%v, finalizers=%v)",
				kind, name.Namespace, name.Name, tempObj.GetDeletionTimestamp(), tempObj.GetFinalizers())
		}, tc.ExistsTimeout, tc.Interval).Should(gomega.Succeed(), "Timed out waiting for %s %s/%s to be fully deleted", kind, name.Namespace, name.Name)
	}
}
