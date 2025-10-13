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
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"
	testutils "sigs.k8s.io/gateway-api-inference-extension/test/utils"
)

var _ = ginkgo.Describe("InferencePool", func() {
	var infObjective *v1alpha2.InferenceObjective
	ginkgo.BeforeEach(func() {
		ginkgo.By("Waiting for the namespace to exist.")
		namespaceExists(testConfig)

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

		// Ensure the expected responses include the InferenceObjective target model names.
		var expected []string
		expected = append(expected, targetModelName)
		curlCmd := getCurlCommand(envoyName, testConfig.NsName, envoyPort, modelName, curlTimeout, t.api, t.promptOrMessages, false)

		actual := make(map[string]int)
		gomega.Eventually(func() error {
			resp, err := testutils.ExecCommandInPod(testConfig, "curl", "curl", curlCmd)
			if err != nil {
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
			// Compare ignoring order
			if !cmp.Equal(got, expected, cmpopts.SortSlices(func(a, b string) bool { return a < b })) {
				return fmt.Errorf("actual (%v) != expected (%v); resp=%q", got, expected, resp)
			}
			return nil
		}, testConfig.ReadyTimeout, curlInterval).Should(gomega.Succeed())
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

	// Generate traffic by sending requests through the inference extension
	ginkgo.By("Generating traffic through the inference extension")
	curlCmd := getCurlCommand(envoyName, testConfig.NsName, envoyPort, modelName, curlTimeout, "/completions", "Write as if you were a critic: San Francisco", true)

	// Run the curl command multiple times to generate some metrics data
	for i := 0; i < 5; i++ {
		_, err := testutils.ExecCommandInPod(testConfig, "curl", "curl", curlCmd)
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
	}

	// modify the curl command to generate some error metrics
	curlCmd[len(curlCmd)-1] = "invalid input"
	for i := 0; i < 5; i++ {
		_, err := testutils.ExecCommandInPod(testConfig, "curl", "curl", curlCmd)
		gomega.Expect(err).NotTo(gomega.HaveOccurred())
	}

	// Now scrape metrics from the EPP endpoint via the curl pod
	ginkgo.By("Scraping metrics from the EPP endpoint")
	podIP := findReadyPod().Status.PodIP

	// Get the authorization token for reading metrics
	token := ""
	gomega.Eventually(func(g gomega.Gomega) {
		t, err := getMetricsReaderToken(testConfig.K8sClient)
		g.Expect(err).NotTo(gomega.HaveOccurred())
		g.Expect(t).NotTo(gomega.BeEmpty())
		token = t
	}, testConfig.ExistsTimeout, testConfig.Interval).Should(gomega.Succeed())

	// Construct the metric scraping curl command using Pod IP
	metricScrapeCmd := getMetricsScrapeCommand(podIP, token)

	ginkgo.By("Verifying that all expected metrics are present.")
	gomega.Eventually(func() error {
		// Execute the metrics scrape command inside the curl pod
		resp, err := testutils.ExecCommandInPod(testConfig, "curl", "curl", metricScrapeCmd)
		if err != nil {
			return err
		}
		// Verify that we got a 200 OK responsecurl
		if !strings.Contains(resp, "200 OK") {
			return fmt.Errorf("did not get 200 OK: %s", resp)
		}
		// Check if all expected metrics are present in the metrics output
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
		"curl", "-i", "--max-time", strconv.Itoa((int)(curlTimeout.Seconds())),
		"-H", "Authorization: Bearer " + token, fmt.Sprintf("http://%s:%d/metrics", podIP, 9090),
	}
}

// getCurlCommand returns the command, as a slice of strings, for curl'ing
// the test model server at the given name, namespace, port, and model name.
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
