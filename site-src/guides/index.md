<!-- If you are updating this index.md guide, please make sure to update the getting-started-latest.md as well -->

# Getting started with an Inference Gateway

--8<-- "site-src/_includes/intro.md"

## **Prerequisites**

--8<-- "site-src/_includes/prereqs.md"

## **Steps**

### Set Latest Release Variable

```bash
IGW_LATEST_RELEASE=$(curl -s https://api.github.com/repos/kubernetes-sigs/gateway-api-inference-extension/releases \
  | jq -r '.[] | select(.prerelease == false) | .tag_name' \
  | sort -V \
  | tail -n1)
```

### Deploy Sample Model Server

--8<-- "site-src/_includes/vllm-gpu.md"

    ```bash
    export INFERENCE_POOL_NAME=vllm-qwen3-32b
    export MODEL_NAME=Qwen/Qwen3-32B
    kubectl create secret generic hf-token --from-literal=token=$HF_TOKEN # Your Hugging Face Token with access to the set of Llama models
    kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/${IGW_LATEST_RELEASE}/config/manifests/vllm/gpu-deployment.yaml
    ```

--8<-- "site-src/_includes/model-server-cpu.md"

    ```bash
    export INFERENCE_POOL_NAME=vllm-qwen3-32b
    export MODEL_NAME=Qwen/Qwen3-32B
    kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/${IGW_LATEST_RELEASE}/config/manifests/vllm/cpu-deployment.yaml
    ```

--8<-- "site-src/_includes/model-server-sim.md"

    ```bash
    export INFERENCE_POOL_NAME=vllm-llama3-8b-instruct
    export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
    kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/${IGW_LATEST_RELEASE}/config/manifests/vllm/sim-deployment.yaml
    ```

### Install the Inference Extension CRDs

```bash
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${IGW_LATEST_RELEASE}/manifests.yaml
```

### Install the Gateway

   Choose one of the following options to install Gateway.

=== "GKE"

      Nothing to install here, you can move to the next [section](#deploy-the-inferencepool-and-endpoint-picker-extension)

=== "Istio"

      1. Requirements

         - Gateway API [CRDs](https://gateway-api.sigs.k8s.io/guides/#installing-gateway-api) installed.

      1. Install Istio:

         On Linux or MacOS
         ```
         ISTIO_VERSION=1.28.0
         curl -L https://istio.io/downloadIstio | ISTIO_VERSION=${ISTIO_VERSION} sh -
         ./istio-$ISTIO_VERSION/bin/istioctl install \
            --set values.pilot.env.ENABLE_GATEWAY_API_INFERENCE_EXTENSION=true
         ```
         On Windows
         ```
         ISTIO_VERSION=1.28.0
         wget https://storage.googleapis.com/istio-release/releases/$ISTIO_VERSION/istio-$ISTIO_VERSION-win.zip
         unzip istioctl-$ISTIO_VERSION-win.zip
         ./istio-$ISTIO_VERSION/bin/istioctl.exe install \
            --set values.pilot.env.ENABLE_GATEWAY_API_INFERENCE_EXTENSION=true
         ```
         > **Note**
         >
         > Istio v1.28.0 includes full support for InferencePool v1. This guide assumes you are using Istio v1.28.0 or later to ensure compatibility with the InferencePool API.

=== "Agentgateway"

      1. Requirements

         - Gateway API [CRDs](https://gateway-api.sigs.k8s.io/guides/#installing-gateway-api) installed.

      1. Set the Agentgateway version and install the Agentgateway CRDs:

         ```bash
         AGW_VERSION=v1.0.0
         helm upgrade -i --create-namespace --namespace agentgateway-system --version $AGW_VERSION agentgateway-crds oci://cr.agentgateway.dev/charts/agentgateway-crds
         ```

      1. Install Agentgateway:

         ```bash
         helm upgrade -i --namespace agentgateway-system --version $AGW_VERSION agentgateway oci://cr.agentgateway.dev/charts/agentgateway --set inferenceExtension.enabled=true
         ```

=== "NGINX Gateway Fabric"

      1. Requirements

         - Gateway API [CRDs](https://gateway-api.sigs.k8s.io/guides/#installing-gateway-api) installed.

      1. Install NGINX Gateway Fabric with the Inference Extension enabled by setting the `nginxGateway.gwAPIInferenceExtension.enable=true` Helm value

         ```bash
         helm install ngf oci://ghcr.io/nginx/charts/nginx-gateway-fabric --create-namespace -n nginx-gateway --set nginxGateway.gwAPIInferenceExtension.enable=true
         ```

### Deploy an Inference Gateway

   Choose one of the following options to deploy an Inference Gateway.

=== "GKE"

      1. Enable the Google Kubernetes Engine API, Compute Engine API, the Network Services API and configure proxy-only subnets when necessary. 
         See [Deploy Inference Gateways](https://cloud.google.com/kubernetes-engine/docs/how-to/deploy-gke-inference-gateway)
         for detailed instructions.

      1. Deploy the Inference Gateway:

         ```bash
         kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/${IGW_LATEST_RELEASE}/config/manifests/gateway/gke/gateway.yaml
         ```

      1. Confirm that the Gateway was assigned an IP address and reports a `Programmed=True` status:

         ```bash
         $ kubectl get gateway inference-gateway
         NAME                CLASS               ADDRESS         PROGRAMMED   AGE
         inference-gateway   inference-gateway   <MY_ADDRESS>    True         22s
         ```

=== "Istio"

      Please note that this feature is currently in an experimental phase and is not intended for production use.
      The implementation and user experience are subject to changes as we continue to iterate on this project.

      1. Deploy the Inference Gateway:

         ```bash
         kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/${IGW_LATEST_RELEASE}/config/manifests/gateway/istio/gateway.yaml
         ```

      1. Confirm that the Gateway was assigned an IP address and reports a `Programmed=True` status:

         ```bash
         $ kubectl get gateway inference-gateway
         NAME                CLASS               ADDRESS         PROGRAMMED   AGE
         inference-gateway   inference-gateway   <MY_ADDRESS>    True         22s
         ```

=== "Agentgateway"

      [Agentgateway](https://agentgateway.dev/) is a Gateway API and Inference Gateway implementation. Follow these steps
      to run Agentgateway as an Inference Gateway:

      1. Deploy the Inference Gateway:

         ```bash
         cat <<'EOF' | kubectl apply -f -
         apiVersion: gateway.networking.k8s.io/v1
         kind: Gateway
         metadata:
           name: inference-gateway
         spec:
           gatewayClassName: agentgateway
           listeners:
           - name: http
             port: 80
             protocol: HTTP
         EOF
         ```

      1. Confirm that the Gateway was assigned an IP address and reports a `Programmed=True` status:

         ```bash
         $ kubectl get gateway inference-gateway
         NAME                CLASS               ADDRESS         PROGRAMMED   AGE
         inference-gateway   inference-gateway   <MY_ADDRESS>    True         22s
         ```

=== "NGINX Gateway Fabric"

      1. Deploy the Gateway

         ```bash
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/nginxgatewayfabric/gateway.yaml
         ```

      1. Confirm that the Gateway was assigned an IP address and reports a `Programmed=True` status:

         ```bash
         $ kubectl get gateway inference-gateway
         NAME                CLASS               ADDRESS         PROGRAMMED   AGE
         inference-gateway   inference-gateway   <MY_ADDRESS>    True         22s
         ```
      
       For more information, see the [NGINX Gateway Fabric - Inference Gateway Setup guide](https://docs.nginx.com/nginx-gateway-fabric/how-to/gateway-api-inference-extension/)

### Deploy the InferencePool and Endpoint Picker Extension

   Install an InferencePool that selects the endpoints from the sample model server you deployed and listens on port 8000. The Helm install command automatically installs the EPP, InferencePool along with provider specific resources.

   Set the chart version and then select a tab to follow the provider-specific instructions.

   ```bash
   export IGW_CHART_VERSION=${IGW_LATEST_RELEASE}
   ```

--8<-- "site-src/_includes/epp.md"

### Verify HttpRoute and InferencePool Status

--8<-- "site-src/_includes/verify-status.md"

### Deploy InferenceObjective (Optional)

Deploy the sample InferenceObjective which allows you to specify priority of requests.

   ```bash
   kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/${IGW_LATEST_RELEASE}/config/manifests/inferenceobjective.yaml
   ```

--8<-- "site-src/_includes/test.md"

--8<-- "site-src/_includes/bbr.md"

### Next Steps: Advanced Features

You have now deployed a basic Inference Gateway with a simple routing strategy. To explore more advanced features, such as latency-based routing, please refer to the following guide:

*   [Latency-Based Routing](https://gateway-api-inference-extension.sigs.k8s.io/guides/latency-based-predictor.md)

### Cleanup

   The following instructions assume you would like to cleanup ALL resources that were created in this quickstart guide.
   Please be careful not to delete resources you'd like to keep.

   1. Uninstall the InferencePool, InferenceObjective and model server resources:

      ```bash
      helm uninstall ${INFERENCE_POOL_NAME}
      kubectl delete -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/${IGW_LATEST_RELEASE}/config/manifests/inferenceobjective.yaml --ignore-not-found
      kubectl delete -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/${IGW_LATEST_RELEASE}/config/manifests/vllm/cpu-deployment.yaml --ignore-not-found
      kubectl delete -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/${IGW_LATEST_RELEASE}/config/manifests/vllm/gpu-deployment.yaml --ignore-not-found
      kubectl delete -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/${IGW_LATEST_RELEASE}/config/manifests/vllm/sim-deployment.yaml --ignore-not-found
      kubectl delete secret hf-token --ignore-not-found
      ```

   1. Uninstall the Gateway API Inference Extension CRDs:

      ```bash
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/${IGW_LATEST_RELEASE}/manifests.yaml --ignore-not-found
      ```
      
   1. Choose one of the following options to cleanup the Inference Gateway.

=== "GKE"

      ```bash
      kubectl delete -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/${IGW_LATEST_RELEASE}/config/manifests/gateway/gke/gateway.yaml --ignore-not-found
      kubectl delete -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/${IGW_LATEST_RELEASE}/config/manifests/gateway/gke/healthcheck.yaml --ignore-not-found
      kubectl delete -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/${IGW_LATEST_RELEASE}/config/manifests/gateway/gke/gcp-backend-policy.yaml --ignore-not-found
      ```

=== "Istio"

      ```bash
      kubectl delete -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/${IGW_LATEST_RELEASE}/config/manifests/gateway/istio/gateway.yaml --ignore-not-found
      ```

      The following steps assume you would like to clean up ALL Istio resources that were created in this quickstart guide.

      1. Uninstall All Istio resources:

         ```bash
         istioctl uninstall -y --purge
         ```

      1. Remove the Istio namespace:

         ```bash
         kubectl delete ns istio-system
         ```

=== "Agentgateway"

      ```bash
      kubectl delete gateway inference-gateway --ignore-not-found
      ```

      The following steps assume you would like to cleanup ALL Agentgateway resources that were created in this quickstart guide.

      1. Uninstall Agentgateway:

         ```bash
         helm uninstall agentgateway -n agentgateway-system
         ```

      1. Uninstall the Agentgateway CRDs:

         ```bash
         helm uninstall agentgateway-crds -n agentgateway-system
         ```

      1. Remove the Agentgateway namespace:

         ```bash
         kubectl delete ns agentgateway-system
         ```

=== "NGINX Gateway Fabric"

      Follow these steps to remove the NGINX Gateway Fabric Inference Gateway and all related resources.

      1. Remove Inference Gateway and HTTPRoute:

         ```bash
         kubectl delete -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/${IGW_LATEST_RELEASE}/config/manifests/gateway/nginxgatewayfabric/gateway.yaml --ignore-not-found
         ```

      1. Uninstall NGINX Gateway Fabric:

         ```bash
         helm uninstall ngf -n nginx-gateway
         ```

      1. Clean up namespace:
   
         ```bash
         kubectl delete ns nginx-gateway
         ```
