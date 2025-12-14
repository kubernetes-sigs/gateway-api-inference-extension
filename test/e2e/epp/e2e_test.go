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
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"
	testutils "sigs.k8s.io/gateway-api-inference-extension/test/utils"
)

const (
	firstPort             = 8000
	numPorts              = 2
	maxConcurrentRequests = 2 // prevent hammering Envoy and backend
	maxRetries            = 5
	backoff               = 5 * time.Second
	batches               = 20
)

var _ = ginkgo.Describe("InferencePool", func() {
	var infObjective *v1alpha2.InferenceObjective
	ginkgo.BeforeEach(func() {
		ginkgo.By("Waiting for the namespace to exist.")
		namespaceExists(testConfig)

		ginkgo.By("Modifying deployment using local image for testing (temporary).")
		deploy := &appsv1.Deployment{}
		key := types.NamespacedName{Name: modelServerName, Namespace: testConfig.NsName}

		gomega.Eventually(func() error {
			err := testConfig.K8sClient.Get(testConfig.Context, key, deploy)
			if err != nil {
				return err
			}

			// Instead of hardcoding arguments, we can instead replace the arguments that need
			// to be changed, preserving any others that may exist.
			var newArgs []string
			skipNext := false
			for _, arg := range deploy.Spec.Template.Spec.Containers[0].Args {
				if skipNext {
					skipNext = false
					continue
				}
				// If this is one of the arguments we are updating, skip it AND its value
				if arg == "--port" || arg == "--data-parallel-size" {
					skipNext = true
					continue
				}
				newArgs = append(newArgs, arg)
			} // contains only the args we want to keep

			// add new arguments to open proper ports
			newArgs = append(newArgs, "--port", strconv.Itoa(firstPort))
			newArgs = append(newArgs, "--data-parallel-size", strconv.Itoa(numPorts))
			deploy.Spec.Template.Spec.Containers[0].Args = newArgs
			deploy.Spec.Template.Spec.Containers[0].Ports = buildContainerPorts(firstPort, numPorts)

			return testConfig.K8sClient.Update(testConfig.Context, deploy)

		}, testConfig.ExistsTimeout, testConfig.Interval).Should(gomega.Succeed())

		waitForDeploymentRollout(testConfig, deploy)

		pool := &v1.InferencePool{}
		gomega.Eventually(func() error {
			err := testConfig.K8sClient.Get(testConfig.Context, key, pool)
			if err != nil {
				return err
			}

			pool.Spec.TargetPorts = buildTargetPorts(firstPort, numPorts)

			return testConfig.K8sClient.Update(testConfig.Context, pool)
		}, testConfig.ExistsTimeout, testConfig.Interval).Should(gomega.Succeed())

		ginkgo.By("Restarting EPP to force configuration reload")
		// We delete the EPP *POD*, not the deployment. The Deployment will recreate it immediately.
		// This forces the new EPP process to read the Multi-Port InferencePool from scratch.
		eppLabels := client.MatchingLabels{"app": inferExtName}
		gomega.Expect(testConfig.K8sClient.DeleteAllOf(testConfig.Context, &corev1.Pod{}, client.InNamespace(testConfig.NsName), eppLabels)).To(gomega.Succeed())

		// Wait for the new EPP to be ready
		eppDeploy := &appsv1.Deployment{ObjectMeta: metav1.ObjectMeta{Name: inferExtName, Namespace: testConfig.NsName}}
		waitForDeploymentReady(testConfig, eppDeploy)

		ginkgo.By("Creating an InferenceObjective resource")
		infObjective = newInferenceObjective(testConfig.NsName)
		gomega.Expect(testConfig.K8sClient.Create(testConfig.Context, infObjective)).To(gomega.Succeed())

		ginkgo.By("Ensuring the InferenceObjective resource exists in the namespace")
		gomega.Eventually(func() error {
			return testConfig.K8sClient.Get(testConfig.Context, types.NamespacedName{Namespace: infObjective.Namespace, Name: infObjective.Name}, infObjective)
		}, testConfig.ExistsTimeout, testConfig.Interval).Should(gomega.Succeed())
	})

	ginkgo.AfterEach(func() {
		ginkgo.By("Deleting the InferenceObjective test resource.")
		cleanupInferModelResources()

		// Wait for InferenceObjective to be fully deleted.
		gomega.Eventually(func() error {
			err := testConfig.K8sClient.Get(testConfig.Context, types.NamespacedName{Namespace: infObjective.Namespace, Name: infObjective.Name}, infObjective)
			if err == nil {
				return errors.New("InferenceObjective resource still exists")
			}
			if !k8serrors.IsNotFound(err) {
				return nil
			}
			return nil
		}, testConfig.ExistsTimeout, testConfig.Interval).Should(gomega.Succeed())

		ginkgo.By("Restoring vLLM Deployment and InferencePool.")
		key := types.NamespacedName{Name: modelServerName, Namespace: testConfig.NsName}

		// Restore InferencePool
		pool := &v1.InferencePool{}
		gomega.Eventually(func() error {
			if err := testConfig.K8sClient.Get(testConfig.Context, key, pool); err != nil {
				return err
			}
			pool.Spec.TargetPorts = []v1.Port{{Number: 8000}}
			return testConfig.K8sClient.Update(testConfig.Context, pool)
		}, testConfig.ExistsTimeout, testConfig.Interval).Should(gomega.Succeed())

		// Restore Deployment Args
		deploy := &appsv1.Deployment{}
		gomega.Eventually(func() error {
			if err := testConfig.K8sClient.Get(testConfig.Context, key, deploy); err != nil {
				return err
			}

			// Filter out the custom args we added in BeforeEach
			var originalArgs []string
			skipNext := false
			for _, arg := range deploy.Spec.Template.Spec.Containers[0].Args {
				if skipNext {
					skipNext = false
					continue
				}
				if arg == "--port" || arg == "--data-parallel-size" {
					skipNext = true
					continue
				}
				originalArgs = append(originalArgs, arg)
			}
			deploy.Spec.Template.Spec.Containers[0].Args = originalArgs

			// Restore container ports to just 8000
			deploy.Spec.Template.Spec.Containers[0].Ports = []corev1.ContainerPort{
				{Name: "http-8000", ContainerPort: 8000, Protocol: corev1.ProtocolTCP},
			}

			return testConfig.K8sClient.Update(testConfig.Context, deploy)
		}, testConfig.ExistsTimeout, testConfig.Interval).Should(gomega.Succeed())

		// Wait for rollback to finish.
		waitForDeploymentRollout(testConfig, deploy)
	})

	ginkgo.When("The Inference Extension is running", func() {
		ginkgo.It("Should route traffic to target model servers", func() {
			verifyTrafficRouting()
		})

		ginkgo.It("Should expose EPP metrics after generating traffic", func() {
			verifyMetrics()
		})
	})

	ginkgo.When("Leader election is enabled", func() {
		ginkgo.It("Should elect one leader and have other pods as not ready", func() {
			if !leaderElectionEnabled {
				ginkgo.Skip("Leader election is not enabled for this test run, skipping.")
			}

			ginkgo.By("Verifying that exactly one EPP pod is ready")
			gomega.Eventually(func(g gomega.Gomega) {
				podList := &corev1.PodList{}
				err := testConfig.K8sClient.List(testConfig.Context, podList, client.InNamespace(testConfig.NsName), client.MatchingLabels{"app": inferExtName})
				g.Expect(err).NotTo(gomega.HaveOccurred())

				// The deployment should have 3 replicas for leader election.
				g.Expect(podList.Items).To(gomega.HaveLen(3))

				readyPods := 0
				for _, pod := range podList.Items {
					for _, cond := range pod.Status.Conditions {
						if cond.Type == corev1.PodReady && cond.Status == corev1.ConditionTrue {
							readyPods++
						}
					}
				}
				g.Expect(readyPods).To(gomega.Equal(1), "Expected exactly one pod to be ready")
			}, testConfig.ReadyTimeout, testConfig.Interval).Should(gomega.Succeed())
		})

		ginkgo.It("Should successfully failover and serve traffic after the leader pod is deleted", func() {
			if !leaderElectionEnabled {
				ginkgo.Skip("Leader election is not enabled for this test run, skipping.")
			}

			ginkgo.By("STEP 1: Verifying initial leader is working correctly before failover")
			verifyTrafficRouting()
			verifyMetrics()

			ginkgo.By("STEP 2: Finding and deleting the current leader pod")
			oldLeaderPod := findReadyPod()
			ginkgo.By("Found initial leader pod: " + oldLeaderPod.Name)

			ginkgo.By(fmt.Sprintf("Deleting leader pod %s to trigger failover", oldLeaderPod.Name))
			gomega.Expect(testConfig.K8sClient.Delete(testConfig.Context, oldLeaderPod)).To(gomega.Succeed())

			ginkgo.By("STEP 3: Waiting for a new leader to be elected")
			// The deployment controller will create a new pod. We need to wait for the total number of pods
			// to be back to 3, and for one of the other pods to become the new leader.
			deploy := &appsv1.Deployment{}
			gomega.Eventually(func() error {
				return testConfig.K8sClient.Get(testConfig.Context, types.NamespacedName{Namespace: testConfig.NsName, Name: inferExtName}, deploy)
			}, testConfig.ExistsTimeout, testConfig.Interval).Should(gomega.Succeed())

			// Wait for one replica to become ready again.
			testutils.DeploymentReadyReplicas(testConfig, deploy, 1)

			// Also wait for the total number of replicas to be back to 3.
			gomega.Eventually(func(g gomega.Gomega) {
				d := &appsv1.Deployment{}
				err := testConfig.K8sClient.Get(testConfig.Context, types.NamespacedName{Namespace: testConfig.NsName, Name: inferExtName}, d)
				g.Expect(err).NotTo(gomega.HaveOccurred())
				g.Expect(d.Status.Replicas).To(gomega.Equal(int32(3)), "Deployment should have 3 replicas")
			}, testConfig.ReadyTimeout, testConfig.Interval).Should(gomega.Succeed())

			ginkgo.By("STEP 4: Verifying a new, different leader is elected")
			var newLeaderPod *corev1.Pod
			gomega.Eventually(func(g gomega.Gomega) {
				// Find the current ready pod.
				newLeaderPod = findReadyPod()

				// Ensure the new leader is not the same as the one we just deleted.
				// This guards against a race condition where we might find the old leader
				// before its status is updated to NotReady.
				g.Expect(newLeaderPod.Name).NotTo(gomega.Equal(oldLeaderPod.Name), "The new leader should not be the same as the old deleted leader")
			}, testConfig.ReadyTimeout, testConfig.Interval).Should(gomega.Succeed())
			ginkgo.By("Found new leader pod: " + newLeaderPod.Name)

			ginkgo.By("STEP 5: Verifying the new leader is working correctly after failover")
			verifyTrafficRouting()
			verifyMetrics()
		})
	})
})

// newInferenceObjective creates an InferenceObjective in the given namespace for testutils.
func newInferenceObjective(ns string) *v1alpha2.InferenceObjective {
	return testutils.MakeModelWrapper(types.NamespacedName{Name: "inferenceobjective-sample", Namespace: ns}).
		SetPriority(2).
		SetPoolRef(modelServerName).
		Obj()
}

// verifyTrafficRouting contains the logic for the "Should route traffic to target model servers" test.
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

		// Expected ports and InferenceObjective target models
		expectedPort := generateSequence(firstPort, numPorts)
		expectedModel := []string{targetModelName}

		// Observed ports and InferenceObjective target models
		actualModel := make(map[string]int)
		actualPort := make(map[int]int)

		// Send curl requests to verify routing to all target ports in the InferencePool.
		// Run a small batch per retry (e.g., 5) to keep the test active
		for i := range batches {
			uniqueID := time.Now().UnixNano()
			dynamicHashValue := fmt.Sprintf("Nonce-%d", uniqueID)
			currentPromptOrMessages := t.promptOrMessages // Start with the original

			// Check if the payload is a slice of maps (e.g., for /chat/completions)
			if originalMessages, ok := currentPromptOrMessages.([]map[string]any); ok {
				nonceMsg := map[string]any{
					"role":    "system",
					"content": fmt.Sprintf("TestNonce: %s-%d", dynamicHashValue, i),
				}

				currentPromptOrMessages = append([]map[string]any{nonceMsg}, originalMessages...)
			} else if originalString, ok := t.promptOrMessages.(string); ok {
				currentPromptOrMessages = fmt.Sprintf("[TestNonce: %s-%d] %s", dynamicHashValue, i, originalString)
			} else {
				currentPromptOrMessages = t.promptOrMessages
			}

			curlCmd := getCurlCommand(envoyName, testConfig.NsName, envoyPort, modelName, curlTimeout, t.api, currentPromptOrMessages, false)

			var resp string
			var err error
			// Repeatedly send a message until we get a successful response.
			for attempt := 0; attempt <= maxRetries; attempt++ {
				resp, err = testutils.ExecCommandInPod(testConfig, "curl", "curl", curlCmd)
				if err == nil && strings.Contains(resp, "200 OK") {
					break // Success!
				}

				if attempt < maxRetries {
					time.Sleep(backoff)
				}
			}

			gomega.Expect(err).ToNot(gomega.HaveOccurred(), "Expected curl command to succeed")
			gomega.Expect(resp).To(gomega.ContainSubstring("200 OK"), "Expected to receive a 200 OK response...")

			for _, m := range expectedModel {
				if strings.Contains(resp, m) {
					actualModel[m] = 0
				}
			}
			for _, p := range expectedPort {
				if strings.Contains(resp, fmt.Sprintf("x-inference-port: %d", p)) {
					actualPort[p] = 0
				}
			}
		}

		var gotModel []string
		for m := range actualModel {
			gotModel = append(gotModel, m)
		}
		var gotPort []int
		for p := range actualPort {
			gotPort = append(gotPort, p)
		}

		ginkgo.GinkgoWriter.Printf("Port distribution: %v\n", actualPort)
		ginkgo.GinkgoWriter.Printf("Model distribution: %v\n", actualModel)

		gomega.Expect(gotModel).To(gomega.BeComparableTo(expectedModel, cmpopts.SortSlices(func(a, b string) bool { return a < b })))
		gomega.Expect(gotPort).To(gomega.BeComparableTo(expectedPort, cmpopts.SortSlices(func(a, b int) bool { return a < b })))
	}
}

// verifyMetrics contains the logic for the "Should expose EPP metrics after generating traffic" test.
func verifyMetrics() {
	ginkgo.By("Verifying metrics exposure")
	// Define the metrics we expect to see
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

	// Generate traffic by sending requests through the inference extension.
	ginkgo.By("Generating traffic through the inference extension")
	curlCmd := getCurlCommand(envoyName, testConfig.NsName, envoyPort, modelName, curlTimeout, "/completions", "Write as if you were a critic: San Francisco", true)

	// Run the curl command multiple times to generate some metrics data.

	semaphore := make(chan struct{}, maxConcurrentRequests)

	errorGood := generateTraffic(curlCmd, batches, semaphore)
	gomega.Expect(errorGood).NotTo(gomega.HaveOccurred(), "Expected good traffic generation to succeed")

	// Modify the curl command to generate some error metrics.
	curlCmd[len(curlCmd)-1] = "invalid input"
	errorBad := generateTraffic(curlCmd, batches, semaphore)
	gomega.Expect(errorBad).NotTo(gomega.HaveOccurred(), "Expected bad traffic generation to succeed")

	// Now scrape metrics from the EPP endpoint via the curl pod.
	ginkgo.By("Scraping metrics from the EPP endpoint and verifying all backends were hit")
	podIP := findReadyPod().Status.PodIP

	// Get the authorization token for reading metrics.
	token := ""
	gomega.Eventually(func(g gomega.Gomega) {
		t, err := getMetricsReaderToken(testConfig.K8sClient)
		g.Expect(err).NotTo(gomega.HaveOccurred())
		g.Expect(t).NotTo(gomega.BeEmpty())
		token = t
	}, testConfig.ExistsTimeout, testConfig.Interval).Should(gomega.Succeed())

	// Construct the metric scraping curl command using Pod IP.
	metricScrapeCmd := getMetricsScrapeCommand(podIP, token)

	modelServerPods, err := getPodsByLabel(testConfig.K8sClient, testConfig.Context, testConfig.NsName, "app", modelServerName)
	gomega.Expect(err).NotTo(gomega.HaveOccurred(), "Expected to find model server pods")

	for _, modelServerPod := range modelServerPods {
		for rank := range numPorts {
			metricQueueSize := fmt.Sprintf(
				"inference_pool_per_pod_queue_size{model_server_pod=\"%s-rank-%d\",name=\"%s\"}",
				modelServerPod.Name,
				rank,
				modelServerName,
			)
			expectedMetrics = append(expectedMetrics, metricQueueSize)
		}
	}

	ginkgo.By("Verifying that all expected metrics are present.")
	gomega.Eventually(func() error {
		// Execute the metrics scrape command inside the curl pod.
		resp, err := testutils.ExecCommandInPod(testConfig, "curl", "curl", metricScrapeCmd)
		if err != nil {
			return err
		}
		// Verify that we got a 200 OK response.
		if !strings.Contains(resp, "200 OK") {
			return fmt.Errorf("did not get 200 OK: %s", resp)
		}
		// Check if all expected metrics are present in the metrics output.
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

// findReadyPod finds the first EPP pod that has a "Ready" status condition.
// It's used to target the leader pod in an HA setup.
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
					break // break inner loop
				}
			}
			if foundReadyPod {
				break // break outer loop
			}
		}
		g.Expect(foundReadyPod).To(gomega.BeTrue(), "No ready EPP pod found")
	}, testConfig.ReadyTimeout, testConfig.Interval).Should(gomega.Succeed())
	return readyPod
}

// getMetricsScrapeCommand returns the command to scrape the /metrics endpoint.
func getMetricsScrapeCommand(podIP, token string) []string {
	return []string{
		"curl", "-i", "--max-time", strconv.Itoa((int)(4 * curlTimeout.Seconds())),
		"-H", "Authorization: Bearer " + token, fmt.Sprintf("http://%s:%d/metrics", podIP, 9090),
	}
}

// getCurlCommand returns the command, as a slice of strings, for curl'ing
// the test model server at the given name, namespace, port, and model name.
// This command gets executed by a dummy pod that communicates with Envoy
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
		"Cache-Control: no-cache",
		"-H",
		fmt.Sprintf("%v: inferenceobjective-sample", metadata.ObjectiveKey),
		"-H",
		fmt.Sprintf("%v: %s", metadata.ModelNameRewriteKey, targetModelName),
		"-H",
		"Connection: close",
		"-d",
		string(b),
	}
}

// buildContainerPorts constructs a slice of corev1.ContainerPort starting from 'start' with 'count' ports.
func buildContainerPorts(start int, count int) []corev1.ContainerPort {
	ports := make([]corev1.ContainerPort, count)
	for i := range count {
		portNum := int32(start + i)
		ports[i] = corev1.ContainerPort{
			Name:          fmt.Sprintf("http-%d", portNum),
			ContainerPort: portNum,
			Protocol:      corev1.ProtocolTCP,
		}
	}
	return ports
}

// buildTargetPorts constructs a slice of v1.Port starting from 'start' with 'count' ports.
func buildTargetPorts(start int, count int) []v1.Port {
	ports := make([]v1.Port, count)
	for i := range count {
		ports[i] = v1.Port{
			Number: v1.PortNumber(start + i),
		}
	}
	return ports
}

// waitForDeploymentRollout waits until the Deployment has completed its update.
// It ensures that the new version is fully rolled out and available.
func waitForDeploymentRollout(tc *testutils.TestConfig, deploy *appsv1.Deployment) {
	ginkgo.By(fmt.Sprintf("Waiting for Deployment %s/%s to complete rollout", deploy.Namespace, deploy.Name))

	key := types.NamespacedName{Name: deploy.Name, Namespace: deploy.Namespace}

	gomega.Eventually(func() error {
		currentDeploy := &appsv1.Deployment{}
		if err := tc.K8sClient.Get(tc.Context, key, currentDeploy); err != nil {
			return err
		}

		if currentDeploy.Generation > currentDeploy.Status.ObservedGeneration {
			return errors.New("deployment generation not observed yet")
		}

		desiredReplicas := *currentDeploy.Spec.Replicas

		if currentDeploy.Status.UpdatedReplicas < desiredReplicas {
			return fmt.Errorf("waiting for updated replicas: %d/%d", currentDeploy.Status.UpdatedReplicas, desiredReplicas)
		}

		if currentDeploy.Status.AvailableReplicas < desiredReplicas {
			return fmt.Errorf("waiting for available replicas: %d/%d", currentDeploy.Status.AvailableReplicas, desiredReplicas)
		}

		if currentDeploy.Status.Replicas > desiredReplicas {
			return fmt.Errorf("waiting for old replicas to terminate: %d > %d", currentDeploy.Status.Replicas, desiredReplicas)
		}

		return nil
	}, testConfig.ReadyTimeout, testConfig.Interval).Should(gomega.Succeed(), "Deployment failed to roll out within timeout")

	ginkgo.By("Deployment rollout complete")
}

// waitForDeploymentReady waits for the Deployment to have all replicas ready.
func waitForDeploymentReady(tc *testutils.TestConfig, deploy *appsv1.Deployment) {
	ginkgo.By(fmt.Sprintf("waiting for Deployment %s/%s to be ready", deploy.Namespace, deploy.Name))

	key := types.NamespacedName{Name: deploy.Name, Namespace: deploy.Namespace}

	gomega.Eventually(func() error {
		current := &appsv1.Deployment{}
		if err := tc.K8sClient.Get(tc.Context, key, current); err != nil {
			return err
		}

		if current.Status.Replicas != current.Status.ReadyReplicas {
			return fmt.Errorf("replicas mismatch: expected %d, got %d ready",
				current.Status.Replicas, current.Status.ReadyReplicas)
		}

		if current.Status.ReadyReplicas == 0 {
			return errors.New("no replicas are ready yet")
		}

		return nil
	}, testConfig.ReadyTimeout, testConfig.Interval).Should(gomega.Succeed())
}

// generateTraffic sends multiple concurrent requests using the provided curl command.
func generateTraffic(
	curlCmd []string,
	batches int,
	semaphore chan struct{},
) error {
	var wg sync.WaitGroup
	errorCh := make(chan error, batches)

	for i := 0; i < batches; i++ {
		wg.Add(1)
		semaphore <- struct{}{}

		go func(requestNum int) {
			defer wg.Done()
			defer func() { <-semaphore }()

			var err error
			// RETRY LOOP
			for attempt := 0; attempt <= maxRetries; attempt++ {
				_, err = testutils.ExecCommandInPod(testConfig, "curl", "curl", curlCmd)
				if err == nil {
					return // Success!
				}

				time.Sleep(backoff)
			}

			// If we get here, we failed all retries
			errorCh <- fmt.Errorf("request %d failed: %w", requestNum, err)
		}(i)
	}

	wg.Wait()
	close(errorCh)

	// Collect any errors that occurred
	failures := make([]error, 0, batches)
	for err := range errorCh {
		failures = append(failures, err)
	}

	if len(failures) > 0 {
		return fmt.Errorf("found %d failed requests: %v", len(failures), failures)
	}

	return nil
}

// getPodsByLabel lists pods in a given namespace that have a specific label key-value pair.
func getPodsByLabel(k8sClient client.Client, ctx context.Context, namespace, labelKey, labelValue string) ([]corev1.Pod, error) {
	podList := &corev1.PodList{}
	labels := map[string]string{labelKey: labelValue}

	listOptions := []client.ListOption{
		client.InNamespace(namespace),
		client.MatchingLabels(labels),
	}

	if err := k8sClient.List(ctx, podList, listOptions...); err != nil {
		return nil, fmt.Errorf("failed to list pods with label %s=%s in namespace %s: %w", labelKey, labelValue, namespace, err)
	}
	return podList.Items, nil
}

// generateSequence generates a sequence of integers starting from 'start' with 'count' numbers.
func generateSequence(start int, count int) []int {
	nums := make([]int, count)
	for i := range count {
		nums[i] = start + i
	}
	return nums
}
