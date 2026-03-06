=== "GKE"

      ```bash
      export GATEWAY_PROVIDER=gke
      helm install ${INFERENCE_POOL_NAME} \
      --dependency-update \
      --set inferencePool.modelServers.matchLabels.app=${MODEL_SERVER_APP_LABEL} \
      --set provider.name=$GATEWAY_PROVIDER \
      --set experimentalHttpRoute.enabled=true \
      --version $IGW_CHART_VERSION \
      oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool
      ```

=== "Istio"

      ```bash
      export GATEWAY_PROVIDER=istio
      helm install ${INFERENCE_POOL_NAME} \
      --dependency-update \
      --set inferencePool.modelServers.matchLabels.app=${MODEL_SERVER_APP_LABEL} \
      --set provider.name=$GATEWAY_PROVIDER \
      --set experimentalHttpRoute.enabled=true \
      --version $IGW_CHART_VERSION \
      oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool
      ```

=== "Agentgateway"

      ```bash
      export GATEWAY_PROVIDER=none
      helm install ${INFERENCE_POOL_NAME} \
      --dependency-update \
      --set inferencePool.modelServers.matchLabels.app=${MODEL_SERVER_APP_LABEL} \
      --set provider.name=$GATEWAY_PROVIDER \
      --set experimentalHttpRoute.enabled=true \
      --version $IGW_CHART_VERSION \
      oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool
      ```

=== "NGINX Gateway Fabric"

      ```bash
      export GATEWAY_PROVIDER=none
      helm install ${INFERENCE_POOL_NAME} \
      --dependency-update \
      --set inferencePool.modelServers.matchLabels.app=${MODEL_SERVER_APP_LABEL} \
      --set provider.name=$GATEWAY_PROVIDER \
      --set experimentalHttpRoute.enabled=true \
      --version $IGW_CHART_VERSION \
      oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool
      ```
