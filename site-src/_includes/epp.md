=== "GKE"

      ```bash
      export GATEWAY_PROVIDER=gke
      helm install ${MODEl_SERVER}-vllm-qwen3-32b \
      --dependency-update \
      --set inferencePool.modelServers.matchLabels.app=vllm-qwen3-32b \
      --set provider.name=$GATEWAY_PROVIDER \
      --set experimentalHttpRoute.enabled=true \
      --version $IGW_CHART_VERSION \
      oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool
      ```

=== "Istio"

      ```bash
      export GATEWAY_PROVIDER=istio
      helm install vllm-qwen3-32b \
      --dependency-update \
      --set inferencePool.modelServers.matchLabels.app=vllm-qwen3-32b \
      --set provider.name=$GATEWAY_PROVIDER \
      --set experimentalHttpRoute.enabled=true \
      --version $IGW_CHART_VERSION \
      oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool
      ```

=== "Kgateway"

      ```bash
      export GATEWAY_PROVIDER=none
      helm install vllm-qwen3-32b \
      --dependency-update \
      --set inferencePool.modelServers.matchLabels.app=vllm-qwen3-32b \
      --set provider.name=$GATEWAY_PROVIDER \
      --set experimentalHttpRoute.enabled=true \
      --version $IGW_CHART_VERSION \
      oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool
      ```

=== "NGINX Gateway Fabric"

      ```bash
      export GATEWAY_PROVIDER=none
      helm install vllm-qwen3-32b \
      --dependency-update \
      --set inferencePool.modelServers.matchLabels.app=vllm-qwen3-32b \
      --set provider.name=$GATEWAY_PROVIDER \
      --set experimentalHttpRoute.enabled=true \
      --version $IGW_CHART_VERSION \
      oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool
      ```