=== "GKE"

      ```bash
      export GATEWAY_PROVIDER=gke
      helm install ${MODEL_SERVER}-qwen3-32b \
      --dependency-update \
      --set inferencePool.modelServers.matchLabels.app=${MODEL_SERVER}-qwen3-32b \
      --set provider.name=$GATEWAY_PROVIDER \
      --set inferencePool.modelServerType=${MODEL_SERVER} \
      --set experimentalHttpRoute.enabled=true \
      --version $IGW_CHART_VERSION \
      oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/inferencepool
      ```

=== "Istio"

      ```bash
      export GATEWAY_PROVIDER=istio
      helm install ${MODEL_SERVER}-qwen3-32b \
      --dependency-update \
      --set inferencePool.modelServers.matchLabels.app=${MODEL_SERVER}-qwen3-32b \
      --set provider.name=$GATEWAY_PROVIDER \
      --set inferencePool.modelServerType=${MODEL_SERVER} \
      --set experimentalHttpRoute.enabled=true \
      --version $IGW_CHART_VERSION \
      oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/inferencepool
      ```

=== "Kgateway"

      ```bash
      export GATEWAY_PROVIDER=none
      helm install ${MODEL_SERVER}-qwen3-32b \
      --dependency-update \
      --set inferencePool.modelServers.matchLabels.app=${MODEL_SERVER}-qwen3-32b \
      --set provider.name=$GATEWAY_PROVIDER \
      --set inferencePool.modelServerType=${MODEL_SERVER} \
      --set experimentalHttpRoute.enabled=true \
      --version $IGW_CHART_VERSION \
      oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/inferencepool
      ```

=== "NGINX Gateway Fabric"

      ```bash
      export GATEWAY_PROVIDER=none
      helm install ${MODEL_SERVER}-qwen3-32b \
      --dependency-update \
      --set inferencePool.modelServers.matchLabels.app=${MODEL_SERVER}-qwen3-32b \
      --set provider.name=$GATEWAY_PROVIDER \
      --set inferencePool.modelServerType=${MODEL_SERVER} \
      --set experimentalHttpRoute.enabled=true \
      --version $IGW_CHART_VERSION \
      oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/inferencepool
      ```