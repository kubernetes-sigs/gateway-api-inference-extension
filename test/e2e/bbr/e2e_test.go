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
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"

	testutils "sigs.k8s.io/gateway-api-inference-extension/test/utils"
)

const (
	defaultCurlTimeout  = 30 * time.Second
	defaultCurlInterval = 5 * time.Second
	maxRetries          = 5
	backoff             = 5 * time.Second
)

var (
	curlTimeout  = getEnvDuration("CURL_TIMEOUT", defaultCurlTimeout)
	curlInterval = defaultCurlInterval
)

func getEnvDuration(key string, defaultVal time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return defaultVal
}

var _ = ginkgo.Describe("BBR", func() {
	ginkgo.BeforeEach(func() {
		ginkgo.By("Waiting for the namespace to exist.")
		namespaceExists(testConfig)
	})

	ginkgo.When("The BBR extension is running with multi-pool routing", func() {
		ginkgo.It("Should route base models to the correct pool", func() {
			verifyBaseModelRouting()
		})

		ginkgo.It("Should route LoRA adapters to the correct pool via base model lookup", func() {
			verifyLoRARouting()
		})

		ginkgo.It("Should handle streaming requests with correct pool routing", func() {
			verifyStreamingRouting()
		})

		ginkgo.It("Should expose BBR metrics after processing traffic", func() {
			verifyBBRMetrics()
		})
	})
})

func verifyBaseModelRouting() {
	ginkgo.By("Verifying base model requests route to the correct pool")

	for _, tc := range []struct {
		api          string
		model        string
		expectedPool string
		prompt       any
	}{
		{
			api:          "/chat/completions",
			model:        "meta-llama/Llama-3.1-8B-Instruct",
			expectedPool: "llama",
			prompt: []map[string]any{
				{"role": "user", "content": "Hello from Llama"},
			},
		},
		{
			api:          "/completions",
			model:        "deepseek/vllm-deepseek-r1",
			expectedPool: "deepseek",
			prompt:       "Hello from DeepSeek",
		},
	} {
		ginkgo.By(fmt.Sprintf("Sending %s request with base model %q, expecting pool %q", tc.api, tc.model, tc.expectedPool))
		resp := sendRequestWithRetry(tc.api, tc.model, tc.prompt, false)

		gomega.Expect(resp).To(gomega.ContainSubstring("200 OK"),
			fmt.Sprintf("Expected 200 OK for base model %s", tc.model))
		gomega.Expect(resp).To(gomega.ContainSubstring("x-bbr-routed-pool: "+tc.expectedPool),
			fmt.Sprintf("Expected routing to pool %q for base model %s, got response:\n%s", tc.expectedPool, tc.model, resp))
	}
}

func verifyLoRARouting() {
	ginkgo.By("Verifying LoRA adapter requests route via base model lookup")

	for _, tc := range []struct {
		api          string
		model        string
		expectedPool string
		prompt       any
	}{
		{
			api:          "/chat/completions",
			model:        "food-review-1",
			expectedPool: "llama",
			prompt: []map[string]any{
				{"role": "user", "content": "Review this pasta dish"},
			},
		},
		{
			api:          "/completions",
			model:        "ski-resorts",
			expectedPool: "deepseek",
			prompt:       "Tell me about the best ski resorts",
		},
		{
			api:          "/chat/completions",
			model:        "movie-critique",
			expectedPool: "deepseek",
			prompt: []map[string]any{
				{"role": "user", "content": "Review the latest sci-fi movie"},
			},
		},
	} {
		ginkgo.By(fmt.Sprintf("Sending %s request with LoRA %q, expecting pool %q", tc.api, tc.model, tc.expectedPool))
		resp := sendRequestWithRetry(tc.api, tc.model, tc.prompt, false)

		gomega.Expect(resp).To(gomega.ContainSubstring("200 OK"),
			fmt.Sprintf("Expected 200 OK for LoRA %s", tc.model))
		gomega.Expect(resp).To(gomega.ContainSubstring("x-bbr-routed-pool: "+tc.expectedPool),
			fmt.Sprintf("Expected routing to pool %q for LoRA %s, got response:\n%s", tc.expectedPool, tc.model, resp))
	}
}

func verifyStreamingRouting() {
	ginkgo.By("Verifying streaming requests route correctly and produce SSE output")

	for _, tc := range []struct {
		model        string
		expectedPool string
	}{
		{
			model:        "food-review-1",
			expectedPool: "llama",
		},
		{
			model:        "ski-resorts",
			expectedPool: "deepseek",
		},
	} {
		ginkgo.By(fmt.Sprintf("Sending streaming request with model %q, expecting pool %q", tc.model, tc.expectedPool))
		resp := sendRequestWithRetry("/chat/completions", tc.model, []map[string]any{
			{"role": "user", "content": "Streaming test"},
		}, true)

		gomega.Expect(resp).To(gomega.ContainSubstring("200 OK"),
			fmt.Sprintf("Expected 200 OK for streaming model %s", tc.model))
		gomega.Expect(resp).To(gomega.ContainSubstring("x-bbr-routed-pool: "+tc.expectedPool),
			fmt.Sprintf("Expected routing to pool %q for streaming model %s", tc.expectedPool, tc.model))
		gomega.Expect(resp).To(gomega.ContainSubstring("data: "),
			fmt.Sprintf("Expected SSE data chunks in streaming response for model %s", tc.model))
	}
}

func verifyBBRMetrics() {
	ginkgo.By("Generating traffic through BBR to populate metrics")
	for _, model := range []string{"food-review-1", "ski-resorts", "movie-critique"} {
		_ = sendRequestWithRetry("/completions", model, "Metrics test prompt", false)
	}

	ginkgo.By("Scraping metrics from the BBR endpoint")
	bbrPodIP := findBBRPodIP()
	metricScrapeCmd := []string{
		"curl", "-i", "--max-time", strconv.Itoa(int(6 * curlTimeout.Seconds())),
		fmt.Sprintf("http://%s:%d/metrics", bbrPodIP, 9090),
	}

	expectedMetrics := []string{
		"bbr_success_total",
		"bbr_info",
	}

	ginkgo.By("Verifying that all expected BBR metrics are present")
	gomega.Eventually(func() error {
		resp, err := testutils.ExecCommandInPod(testConfig, "curl", "curl", metricScrapeCmd)
		if err != nil {
			return err
		}
		if !strings.Contains(resp, "200 OK") {
			return fmt.Errorf("did not get 200 OK from metrics endpoint: %s", resp)
		}
		for _, metric := range expectedMetrics {
			if !strings.Contains(resp, metric) {
				return fmt.Errorf("expected metric %s not found in metrics output", metric)
			}
		}
		return nil
	}, testConfig.ReadyTimeout, curlInterval).Should(gomega.Succeed())
}

func sendRequestWithRetry(api, model string, prompt any, streaming bool) string {
	curlCmd := buildCurlCommand(envoyName, testConfig.NsName, envoyPort, model, curlTimeout, api, prompt, streaming)

	var resp string
	var err error
	for attempt := 0; attempt <= maxRetries; attempt++ {
		resp, err = testutils.ExecCommandInPod(testConfig, "curl", "curl", curlCmd)
		if err == nil && strings.Contains(resp, "200 OK") {
			return resp
		}
		if attempt < maxRetries {
			time.Sleep(backoff)
		}
	}
	gomega.Expect(err).ToNot(gomega.HaveOccurred(), "Expected curl command to succeed")
	gomega.Expect(resp).To(gomega.ContainSubstring("200 OK"),
		fmt.Sprintf("Expected 200 OK for model %s via %s after %d retries, got:\n%s", model, api, maxRetries, resp))
	return resp
}

func findBBRPodIP() string {
	var podIP string
	gomega.Eventually(func(g gomega.Gomega) {
		podList := &corev1.PodList{}
		err := testConfig.K8sClient.List(testConfig.Context, podList,
			client.InNamespace(testConfig.NsName),
			client.MatchingLabels{"app": bbrName})
		g.Expect(err).NotTo(gomega.HaveOccurred())

		foundReady := false
		for i := range podList.Items {
			pod := &podList.Items[i]
			for _, cond := range pod.Status.Conditions {
				if cond.Type == corev1.PodReady && cond.Status == corev1.ConditionTrue {
					g.Expect(pod.Status.PodIP).NotTo(gomega.BeEmpty(), "Ready BBR pod must have an IP")
					podIP = pod.Status.PodIP
					foundReady = true
					break
				}
			}
			if foundReady {
				break
			}
		}
		g.Expect(foundReady).To(gomega.BeTrue(), "No ready BBR pod found")
	}, testConfig.ReadyTimeout, testConfig.Interval).Should(gomega.Succeed())
	return podIP
}

func buildCurlCommand(name, ns, port, model string, timeout time.Duration, api string, promptOrMessages any, streaming ...bool) []string {
	body := map[string]any{
		"model":       model,
		"max_tokens":  100,
		"temperature": 0,
	}
	switch api {
	case "/completions":
		body["prompt"] = promptOrMessages
	case "/chat/completions":
		body["messages"] = promptOrMessages
	}
	if len(streaming) > 0 && streaming[0] {
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
		strconv.Itoa(int(timeout.Seconds())),
		fmt.Sprintf("%s.%s.svc:%s/v1%s", name, ns, port, api),
		"-H",
		"Content-Type: application/json",
		"-H",
		"Cache-Control: no-cache",
		"-H",
		"Connection: close",
		"-d",
		string(b),
	}
}
