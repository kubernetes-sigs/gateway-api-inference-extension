=== "GKE"

      ```bash
      export GATEWAY_PROVIDER=gke
      helm install ${MODEl_SERVER}-llama3-8b-instruct \
      --dependency-update \
      --set inferencePool.modelServers.matchLabels.app=${MODEl_SERVER}-llama3-8b-instruct \
      --set provider.name=$GATEWAY_PROVIDER \
      --set inferencePool.modelServerType=${MODEL_SERVER} \
      --set experimentalHttpRoute.enabled=true \
      --version $IGW_CHART_VERSION \
      oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/inferencepool
      ```

=== "Istio"

      ```bash
      export GATEWAY_PROVIDER=istio
      helm install ${MODEl_SERVER}-llama3-8b-instruct \
      --dependency-update \
      --set inferencePool.modelServers.matchLabels.app=${MODEl_SERVER}-llama3-8b-instruct \
      --set provider.name=$GATEWAY_PROVIDER \
      --set inferencePool.modelServerType=${MODEL_SERVER} \
      --set experimentalHttpRoute.enabled=true \
      --version $IGW_CHART_VERSION \
      oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/inferencepool
      ```

=== "Kgateway"

      ```bash
      export GATEWAY_PROVIDER=none
      helm install ${MODEl_SERVER}-llama3-8b-instruct \
      --dependency-update \
      --set inferencePool.modelServers.matchLabels.app=${MODEl_SERVER}-llama3-8b-instruct \
      --set provider.name=$GATEWAY_PROVIDER \
      --set inferencePool.modelServerType=${MODEL_SERVER} \
      --set experimentalHttpRoute.enabled=true \
      --version $IGW_CHART_VERSION \
      oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/inferencepool
      ```

=== "NGINX Gateway Fabric"

      ```bash
      export GATEWAY_PROVIDER=none
      helm install ${MODEl_SERVER}-llama3-8b-instruct \
      --dependency-update \
      --set inferencePool.modelServers.matchLabels.app=${MODEl_SERVER}-llama3-8b-instruct \
      --set provider.name=$GATEWAY_PROVIDER \
      --set inferencePool.modelServerType=${MODEL_SERVER} \
      --set experimentalHttpRoute.enabled=true \
      --version $IGW_CHART_VERSION \
      oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/inferencepool
      ```