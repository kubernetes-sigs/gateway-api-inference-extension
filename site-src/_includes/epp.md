=== "GKE"

      ```bash
      export GATEWAY_PROVIDER=gke
      helm install vllm-llama3-8b-instruct \
      --set inferencePool.modelServers.matchLabels.app=vllm-llama3-8b-instruct \
      --set provider.name=$GATEWAY_PROVIDER \
      --version $IGW_CHART_VERSION \
      oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool
      ```

=== "Istio"

      ```bash
      export GATEWAY_PROVIDER=istio
      helm install vllm-llama3-8b-instruct \
      --set inferencePool.modelServers.matchLabels.app=vllm-llama3-8b-instruct \
      --set provider.name=$GATEWAY_PROVIDER \
      --version $IGW_CHART_VERSION \
      oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool
      ```

=== "Kgateway"

      ```bash
      export GATEWAY_PROVIDER=none
      helm install vllm-llama3-8b-instruct \
      --set inferencePool.modelServers.matchLabels.app=vllm-llama3-8b-instruct \
      --set provider.name=$GATEWAY_PROVIDER \
      --version $IGW_CHART_VERSION \
      oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool
      ```

=== "NGINX Gateway Fabric"

      ```bash
      export GATEWAY_PROVIDER=none
      helm install vllm-llama3-8b-instruct \
      --set inferencePool.modelServers.matchLabels.app=vllm-llama3-8b-instruct \
      --set provider.name=$GATEWAY_PROVIDER \
      --version $IGW_CHART_VERSION \
      oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool
      ```
