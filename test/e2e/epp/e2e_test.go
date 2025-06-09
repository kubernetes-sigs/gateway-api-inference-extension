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
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	client "sigs.k8s.io/controller-runtime/pkg/client"
	v1alpha2 "sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
	testutils "sigs.k8s.io/gateway-api-inference-extension/test/utils"
)

var _ = ginkgo.Describe("InferencePool", func() {
	var infModel *v1alpha2.InferenceModel
	ginkgo.BeforeEach(func() {
		ginkgo.By("Waiting for the namespace to exist.")
		namespaceExists(cli, nsName)

		ginkgo.By("Creating an InferenceModel resource")
		infModel = newInferenceModel(nsName)
		gomega.Expect(cli.Create(ctx, infModel)).To(gomega.Succeed())

		ginkgo.By("Ensuring the InferenceModel resource exists in the namespace")
		gomega.Eventually(func() error {
			return cli.Get(ctx, types.NamespacedName{Namespace: infModel.Namespace, Name: infModel.Name}, infModel)
		}, existsTimeout, interval).Should(gomega.Succeed())
	})

	ginkgo.AfterEach(func() {
		ginkgo.By("Deleting the InferenceModel test resource.")
		cleanupInferModelResources()
		gomega.Eventually(func() error {
			err := cli.Get(ctx, types.NamespacedName{Namespace: infModel.Namespace, Name: infModel.Name}, infModel)
			if err == nil {
				return fmt.Errorf("InferenceModel resource still exists")
			}
			if !errors.IsNotFound(err) {
				return nil
			}
			return nil
		}, existsTimeout, interval).Should(gomega.Succeed())
	})

	ginkgo.When("The Inference Extension is running", func() {
		ginkgo.It("Should route traffic to target model servers", func() {
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

				// Ensure the expected responses include the inferencemodel target model names.
				var expected []string
				for _, m := range infModel.Spec.TargetModels {
					expected = append(expected, m.Name)
				}
				curlCmd := getCurlCommand(envoyName, nsName, envoyPort, modelName, curlTimeout, t.api, t.promptOrMessages, false)

				actual := make(map[string]int)
				gomega.Eventually(func() error {
					resp, err := testutils.ExecCommandInPod(ctx, cfg, scheme, kubeCli, nsName, "curl", "curl", curlCmd)
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
				}, readyTimeout, curlInterval).Should(gomega.Succeed())
			}
		})

		ginkgo.It("Should expose EPP metrics after generating traffic", func() {
			// Define the metrics we expect to see
			expectedMetrics := []string{
				"inference_model_request_total",
				"inference_model_request_error_total",
				"inference_model_request_duration_seconds",
				// TODO: normalized_time_per_output_token_seconds is not actually recorded yet
				// "normalized_time_per_output_token_seconds",
				"inference_model_request_sizes",
				"inference_model_response_sizes",
				"inference_model_input_tokens",
				"inference_model_output_tokens",
				"inference_pool_average_kv_cache_utilization",
				"inference_pool_average_queue_size",
				"inference_pool_per_pod_queue_size",
				"inference_model_running_requests",
				"inference_pool_ready_pods",
				"inference_extension_info",
			}

			// Generate traffic by sending requests through the inference extension
			ginkgo.By("Generating traffic through the inference extension")
			curlCmd := getCurlCommand(envoyName, nsName, envoyPort, modelName, curlTimeout, "/completions", "Write as if you were a critic: San Francisco", true)

			// Run the curl command multiple times to generate some metrics data
			for i := 0; i < 5; i++ {
				_, err := testutils.ExecCommandInPod(ctx, cfg, scheme, kubeCli, nsName, "curl", "curl", curlCmd)
				gomega.Expect(err).NotTo(gomega.HaveOccurred())
			}

			// modify the curl command to generate some error metrics
			curlCmd[len(curlCmd)-1] = "invalid input"
			for i := 0; i < 5; i++ {
				_, err := testutils.ExecCommandInPod(ctx, cfg, scheme, kubeCli, nsName, "curl", "curl", curlCmd)
				gomega.Expect(err).NotTo(gomega.HaveOccurred())
			}

			// Now scrape metrics from the EPP endpoint via the curl pod
			ginkgo.By("Scraping metrics from the EPP endpoint")

			// Get Pod IP instead of Service
			podList := &corev1.PodList{}
			err := cli.List(ctx, podList, client.InNamespace(nsName), client.MatchingLabels{"app": inferExtName})
			gomega.Expect(err).NotTo(gomega.HaveOccurred())
			gomega.Expect(podList.Items).NotTo(gomega.BeEmpty())
			podIP := podList.Items[0].Status.PodIP
			gomega.Expect(podIP).NotTo(gomega.BeEmpty())

			// Get the authorization token for reading metrics
			token := ""
			gomega.Eventually(func() error {
				token, err = getMetricsReaderToken(cli)
				if err != nil {
					return err
				}
				if token == "" {
					return fmt.Errorf("token not found")
				}
				return nil
			}, existsTimeout, interval).Should(gomega.Succeed())

			// Construct the metric scraping curl command using Pod IP
			metricScrapeCmd := []string{
				"curl",
				"-i",
				"--max-time",
				strconv.Itoa((int)(curlTimeout.Seconds())),
				"-H",
				"Authorization: Bearer " + token,
				fmt.Sprintf("http://%s:%d/metrics", podIP, 9090),
			}

			ginkgo.By("Verifying that all expected metrics are present.")
			gomega.Eventually(func() error {
				// Execute the metrics scrape command inside the curl pod
				resp, err := testutils.ExecCommandInPod(ctx, cfg, scheme, kubeCli, nsName, "curl", "curl", metricScrapeCmd)
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
			}, readyTimeout, curlInterval).Should(gomega.Succeed())
		})
	})
})

// newInferenceModel creates an InferenceModel in the given namespace for testutils.
func newInferenceModel(ns string) *v1alpha2.InferenceModel {
	targets := []v1alpha2.TargetModel{
		{
			Name:   targetModelName,
			Weight: ptr.To(int32(100)),
		},
	}
	return testutils.MakeModelWrapper(types.NamespacedName{Name: "inferencemodel-sample", Namespace: ns}).
		SetCriticality(v1alpha2.Critical).
		SetModelName(modelName).
		SetPoolRef(modelServerName).
		SetTargetModels(targets).
		Obj()
}

func getMetricsReaderToken(k8sClient client.Client) (string, error) {
	secret := &corev1.Secret{}
	err := k8sClient.Get(ctx, types.NamespacedName{Namespace: nsName, Name: metricsReaderSecretName}, secret)
	if err != nil {
		return "", err
	}
	return string(secret.Data["token"]), nil
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
		"-d",
		string(b),
	}
}
