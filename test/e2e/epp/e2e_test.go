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
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
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
	})

	ginkgo.When("The Inference Extension is running", func() {
		ginkgo.It("Should route traffic to target model servers", func() {
			for _, t := range []struct {
				api              string
				promptOrMessages string
			}{
				{
					api:              "/completions",
					promptOrMessages: "Write as if you were a critic: San Francisco",
				},
				{
					api:              "/chat/completions",
					promptOrMessages: `[{"role": "user", "content": "Write as if you were a critic: San Francisco"}]`,
				},
				{
					api: "/chat/completions",
					promptOrMessages: `[{"role": "user", "content": "Write as if you were a critic: San Francisco"},` +
						`{"role": "assistant", "content": "Okay, let's see..."},` +
						`{"role": "user", "content": "Now summarize your thoughts."}]`,
				},
			} {
				ginkgo.By("Verifying connectivity through the inference extension with " +
					t.api + " api and prompt/messages: " + t.promptOrMessages)

				// Ensure the expected responses include the inferencemodel target model names.
				var expected []string
				for _, m := range infModel.Spec.TargetModels {
					expected = append(expected, m.Name)
				}
				curlCmd := getCurlCommand(envoyName, nsName, envoyPort, modelName, curlTimeout, t.api, t.promptOrMessages)

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

// getCurlCommand returns the command, as a slice of strings, for curl'ing
// the test model server at the given name, namespace, port, and model name.
func getCurlCommand(name, ns, port, model string, timeout time.Duration, api string, promptOrMessages string) []string {
	var body string
	switch api {
	case "/completions":
		body = fmt.Sprintf(`{"model": "%s", "prompt": "%s", "max_tokens": 100, "temperature": 0}`, model, promptOrMessages)
	case "/chat/completions":
		body = fmt.Sprintf(`{"model": "%s", "messages": %s, "max_tokens": 100, "temperature": 0}`, model, promptOrMessages)
	}
	return []string{
		"curl",
		"-i",
		"--max-time",
		strconv.Itoa((int)(timeout.Seconds())),
		fmt.Sprintf("%s.%s.svc:%s/v1%s", name, ns, port, api),
		"-H",
		"Content-Type: application/json",
		"-d",
		body,
	}
}
