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
	"bytes"
	"fmt"
	"os"
	"strings"
	"text/template"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	infextv1a1 "inference.networking.x-k8s.io/gateway-api-inference-extension/api/v1alpha1"
	testutils "inference.networking.x-k8s.io/gateway-api-inference-extension/test/utils"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
)

var _ = ginkgo.Describe("InferencePool", func() {
	ginkgo.BeforeEach(func() {
		ginkgo.By("Waiting for the namespace to exist.")
		namespaceExists(cli, nsName)
	})

	ginkgo.AfterEach(func() {
		ginkgo.By("Deleting the InferenceModel test resource.")
		cleanupInferModelResources()
	})

	ginkgo.When("The Inference Extension is running", func() {
		ginkgo.It("Should route traffic to target model servers", func() {
			ginkgo.By("Creating an InferenceModel resource")
			infModel := newInferenceModel(nsName)
			gomega.Expect(cli.Create(ctx, infModel)).To(gomega.Succeed())

			ginkgo.By("Ensuring the InferenceModel resource exists in the namespace")
			gomega.Eventually(func() error {
				err := cli.Get(ctx, types.NamespacedName{Namespace: infModel.Namespace, Name: infModel.Name}, infModel)
				if err != nil {
					return err
				}
				return nil
			}, existsTimeout, interval).Should(gomega.Succeed())

			ginkgo.By("Verifying connectivity through the inference extension")
			curlCmd := getCurlCommand(envoyName, nsName, envoyPort, modelName)

			// Ensure the expected responses include the inferencemodel target model names.
			var expected []string
			for _, m := range infModel.Spec.TargetModels {
				expected = append(expected, m.Name)
			}
			actual := []string{}
			gomega.Eventually(func() error {
				resp, err := testutils.ExecCommandInPod(ctx, cfg, scheme, kubeCli, nsName, "curl", "curl", curlCmd)
				if err != nil || !strings.Contains(resp, "200 OK") {
					return err
				}
				for _, m := range expected {
					if strings.Contains(resp, m) {
						actual = append(actual, m)
					}
				}
				// Compare expected and actual models in responses, ignoring order.
				if !cmp.Equal(actual, expected, cmpopts.SortSlices(func(a, b string) bool { return a < b })) {
					return err
				}
				return nil
			}, existsTimeout, interval).Should(gomega.Succeed())
		})
	})
})

// newInferenceModel creates an InferenceModel in the given namespace for testutils.
func newInferenceModel(ns string) *infextv1a1.InferenceModel {
	targets := []infextv1a1.TargetModel{
		{
			Name:   modelName + "%-0",
			Weight: ptr.To(int32(50)),
		},
		{
			Name:   modelName + "-1",
			Weight: ptr.To(int32(50)),
		},
	}
	return testutils.MakeModelWrapper("inferencemodel-sample", ns).
		SetCriticality(infextv1a1.Critical).
		SetModelName(modelName).
		SetPoolRef(modelServerName).
		SetTargetModels(targets).
		Obj()
}

// newInferencePool creates an InferencePool in the given namespace for testutils.
func newInferencePool(ns string) *infextv1a1.InferencePool {
	return testutils.MakePoolWrapper(modelServerName, ns).
		SetLabels(map[string]string{"app": "vllm"}).
		SetSelector(map[string]string{"app": modelServerName}).
		SetTargetPort(8000).
		Obj()
}

// newClientPod creates a Pod in the given namespace for the curl client.
func newClientPod(ns string) *corev1.Pod {
	name := "curl"
	return testutils.MakePodWrapper(name, ns).
		SetLabels(map[string]string{"app": name}).
		SetSpec(testutils.NewTestClientPodSpec()).
		Obj()
}

// newEnvoyDeployment creates a Deployment in the given namespace for the envoy proxy.
func newEnvoyDeployment(ns string) (*appsv1.Deployment, error) {
	deploySpec, err := testutils.NewTestEnvoyDeploySpec(envoyName, envoyPort)
	if err != nil {
		return nil, err
	}
	return testutils.MakeDeploymentWrapper(envoyName, ns).
		SetLabels(map[string]string{"app": envoyName}).
		SetSpec(deploySpec).
		Obj(), nil
}

// newEnvoyService creates a Service in the given namespace for the envoy proxy.
func newEnvoyService(ns, name, port string) (*corev1.Service, error) {
	spec, err := testutils.NewTestEnvoyServiceSpec(name, port)
	if err != nil {
		return nil, err
	}
	return testutils.MakeServiceWrapper(name, ns).
		SetLabels(map[string]string{"app": name}).
		SetSpec(spec).
		Obj(), nil
}

// newEnvoyConfigMap creates a ConfigMap in the given namespace containing
// the Envoy static configuration.
func newEnvoyConfigMap(ns, name, port string) *corev1.ConfigMap {
	envoyCfg := `
admin:
  address:
    socket_address:
      address: 127.0.0.1
      port_value: 19000
  access_log:
    - name: envoy.access_loggers.file
      typed_config:
        "@type": type.googleapis.com/envoy.extensions.access_loggers.file.v3.FileAccessLog
        path: /dev/null
static_resources:
  listeners:
    - name: envoy-proxy-ready-0.0.0.0-19001
      address:
        socket_address:
          address: 0.0.0.0
          port_value: 19001
      filter_chains:
        - filters:
            - name: envoy.filters.network.http_connection_manager
              typed_config:
                "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
                stat_prefix: envoy-ready-http
                route_config:
                  name: local_route
                  virtual_hosts:
                    - name: prometheus_stats
                      domains: ["*"]
                      routes:
                        - match:
                            prefix: "/stats/prometheus"
                          route:
                            cluster: "prometheus_stats"
                http_filters:
                  - name: envoy.filters.http.health_check
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.health_check.v3.HealthCheck
                      pass_through_mode: false
                      headers:
                        - name: ":path"
                          string_match:
                            exact: "/ready"
                  - name: envoy.filters.http.router
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router
    - name: vllm
      address:
        socket_address:
          address: 0.0.0.0
          port_value: {{ .Port }}
      per_connection_buffer_limit_bytes: 32768
      access_log:
        - name: envoy.access_loggers.file
          filter:
            response_flag_filter:
              flags: ["NR"]
          typed_config:
            "@type": type.googleapis.com/envoy.extensions.access_loggers.file.v3.FileAccessLog
            path: /dev/stdout
            log_format:
              text_format_source:
                inline_string: "{\"start_time\":\"%START_TIME%\",\"method\":\"%REQ(:METHOD)%\",...}\n"
      filter_chains:
        - name: vllm
          filters:
            - name: envoy.filters.network.http_connection_manager
              typed_config:
                "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
                stat_prefix: http-{{ .Port }}
                route_config:
                  name: vllm
                  virtual_hosts:
                    - name: vllm-default
                      domains: ["*"]
                      routes:
                        - match:
                            prefix: "/"
                          route:
                            cluster: original_destination_cluster
                            timeout: 86400s
                            idle_timeout: 86400s
                            upgrade_configs:
                              - upgrade_type: websocket
                          typed_per_filter_config:
                            envoy.filters.http.ext_proc:
                              "@type": type.googleapis.com/envoy.config.route.v3.FilterConfig
                              config: {}
                http_filters:
                  - name: envoy.filters.http.ext_proc
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.ext_proc.v3.ExternalProcessor
                      grpc_service:
                        envoy_grpc:
                          cluster_name: ext_proc
                          authority: inference-gateway-ext-proc.{{ .Namespace }}:9002
                        timeout: 10s
                      processing_mode:
                        request_header_mode: SEND
                        response_header_mode: SKIP
                        request_body_mode: BUFFERED
                        response_body_mode: BUFFERED
                        request_trailer_mode: SKIP
                        response_trailer_mode: SKIP
                      message_timeout: 1000s
                  - name: envoy.filters.http.router
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router
                      suppress_envoy_headers: true
  clusters:
    - name: prometheus_stats
      type: STATIC
      connect_timeout: 0.250s
      load_assignment:
        cluster_name: prometheus_stats
        endpoints:
          - lb_endpoints:
              - endpoint:
                  address:
                    socket_address:
                      address: 127.0.0.1
                      port_value: 19000
    - name: original_destination_cluster
      type: ORIGINAL_DST
      connect_timeout: 1000s
      lb_policy: CLUSTER_PROVIDED
      circuit_breakers:
        thresholds:
          - max_connections: 40000
            max_pending_requests: 40000
            max_requests: 40000
      original_dst_lb_config:
        use_http_header: true
        http_header_name: target-pod
    - name: ext_proc
      type: STRICT_DNS
      connect_timeout: 86400s
      lb_policy: LEAST_REQUEST
      circuit_breakers:
        thresholds:
          - max_connections: 40000
            max_pending_requests: 40000
            max_requests: 40000
            max_retries: 1024
      typed_extension_protocol_options:
        envoy.extensions.upstreams.http.v3.HttpProtocolOptions:
          "@type": type.googleapis.com/envoy.extensions.upstreams.http.v3.HttpProtocolOptions
          explicit_http_config:
            http2_protocol_options:
              initial_stream_window_size: 65536
              initial_connection_window_size: 1048576
      load_assignment:
        cluster_name: ext_proc
        endpoints:
          - locality:
              region: ext_proc/e2e/0
            lb_endpoints:
              - endpoint:
                  address:
                    socket_address:
                      address: inference-gateway-ext-proc.{{ .Namespace }}
                      port_value: 9002
                health_status: HEALTHY
                load_balancing_weight: 1
`

	// Use the template to generate the static configuration
	tmpl, err := template.New("envoyConfig").Parse(envoyCfg)
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	var renderedCfg bytes.Buffer
	err = tmpl.Execute(&renderedCfg, map[string]string{
		"Namespace": ns,
		"Port":      port,
	})
	gomega.Expect(err).NotTo(gomega.HaveOccurred())

	return testutils.MakeConfigMapWrapper(name, ns).
		SetLabels(map[string]string{"app": name}).
		SetData(map[string]string{"envoy.yaml": renderedCfg.String()}).
		Obj()
}

// newModelDeployment creates a Deployment in the given namespace for the model server.
func newModelDeployment(ns, name string) *appsv1.Deployment {
	return testutils.MakeDeploymentWrapper(modelServerName, ns).
		SetLabels(map[string]string{"app": "vllm"}).
		SetSpec(testutils.NewTestModelDeploySpec(name)).
		Obj()
}

// newModelService creates a Service in the given namespace for the model server.
func newModelService(ns, name string) *corev1.Service {
	return testutils.MakeServiceWrapper(modelServerName, ns).
		SetLabels(map[string]string{"app": "vllm"}).
		SetSpec(testutils.NewTestModelServiceSpec(name)).
		Obj()
}

// newModelSecret creates a secret in the given namespace to store the Hugging Face Hub token.
// The token is required by the model server to download the "meta-llama/Llama-2-7b-hf" model used
// for testutils.
func newModelSecret(ns string) *corev1.Secret {
	// Get the Hugging Face Hub token
	token := os.Getenv("HF_TOKEN")
	gomega.Expect(token).ShouldNot(gomega.BeEmpty(), "HF_TOKEN is not set")

	// Store the token raw. K8s will handle base64 encoding in the Secret’s .data field.
	return testutils.MakeSecretWrapper("hf-token", ns).
		SetLabels(map[string]string{"app": "vllm"}).
		SetData(map[string][]byte{"token": []byte(token)}).
		Obj()
}

// newInfExtClusterRole creates a ClusterRole for the inference extension.
func newInfExtClusterRole() *rbacv1.ClusterRole {
	name := "pod-read"
	return testutils.MakeClusterRoleWrapper(name).
		SetLabels(map[string]string{"app": inferExtName}).
		SetRules(testutils.NewTestInferExtRules()).
		Obj()
}

// newInfExtClusterRoleBinding creates a ClusterRoleBinding for the inference extension.
func newInfExtClusterRoleBinding() *rbacv1.ClusterRoleBinding {
	subjects := []rbacv1.Subject{
		{
			Name:      "default",
			Namespace: getNamespace(),
			Kind:      "ServiceAccount",
		},
	}
	return testutils.MakeClusterRoleBindingWrapper("pod-read-binding").
		SetLabels(map[string]string{"app": inferExtName}).
		SetSubjects(subjects).
		SetRoleRef("pod-read").
		Obj()
}

// newInfExtDeployment creates a Deployment in the given namespace for the inference extension.
func newInfExtDeployment(ns, name, pool, svc string) *appsv1.Deployment {
	return testutils.MakeDeploymentWrapper(name, ns).
		SetLabels(map[string]string{"app": name}).
		SetSpec(testutils.NewTestInferExtDeploySpec(name, pool, svc)).
		Obj()
}

// newInfExtService creates a Service in the given namespace for the inference extension.
func newInfExtService(ns, name string) *corev1.Service {
	return testutils.MakeServiceWrapper(name, ns).
		SetLabels(map[string]string{"app": name}).
		SetSpec(testutils.NewTestInferExtServiceSpec(name)).
		Obj()
}

// getCurlCommand returns the command, as a slice of strings, for curl'ing
// the test model server at the given name, namespace, port, and model name.
func getCurlCommand(name, ns, port, model string) []string {
	return []string{
		"curl",
		"-i",
		fmt.Sprintf("%s.%s.svc:%s/v1/completions", name, ns, port),
		"-H",
		"Content-Type: application/json",
		"-d",
		fmt.Sprintf(`{"model": "%s", "prompt": "Write as if you were a critic: San Francisco", "max_tokens": 100, "temperature": 0}`, model),
	}
}
