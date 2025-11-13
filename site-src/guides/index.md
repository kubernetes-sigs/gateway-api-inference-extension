<!-- If you are updating this index.md guide, please make sure to update the getting-started-latest.md as well -->

# Getting started with an Inference Gateway

--8<-- "site-src/_includes/intro.md"

## **Prerequisites**

--8<-- "site-src/_includes/prereqs.md"

## **Steps**

### Deploy Sample Model Server

--8<-- "site-src/_includes/model-server-intro.md"

--8<-- "site-src/_includes/model-server-gpu.md"

    ```bash
    kubectl create secret generic hf-token --from-literal=token=$HF_TOKEN # Your Hugging Face Token with access to the set of Llama models
    kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/v1.1.0/config/manifests/vllm/gpu-deployment.yaml
    ```

--8<-- "site-src/_includes/model-server-cpu.md"

    ```bash
    kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/v1.1.0/config/manifests/vllm/cpu-deployment.yaml
    ```

--8<-- "site-src/_includes/model-server-sim.md"

    ```bash
    kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/v1.1.0/config/manifests/vllm/sim-deployment.yaml
    ```

### Install the Inference Extension CRDs

```bash
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v1.1.0/manifests.yaml
```

### Install the Gateway

   Choose one of the following options to install Gateway.

=== "GKE"

      Nothing to install here, you can move to the next [section](#deploy-the-inferencepool-and-endpoint-picker-extension)

=== "Istio"

      1. Requirements
         - Gateway API [CRDs](https://gateway-api.sigs.k8s.io/guides/#installing-gateway-api) installed.

      1. Install Istio:

         ```
         TAG=$(curl https://storage.googleapis.com/istio-build/dev/1.28-dev)
         # on Linux
         wget https://storage.googleapis.com/istio-build/dev/$TAG/istioctl-$TAG-linux-amd64.tar.gz
         tar -xvf istioctl-$TAG-linux-amd64.tar.gz
         # on macOS
         wget https://storage.googleapis.com/istio-build/dev/$TAG/istioctl-$TAG-osx.tar.gz
         tar -xvf istioctl-$TAG-osx.tar.gz
         # on Windows
         wget https://storage.googleapis.com/istio-build/dev/$TAG/istioctl-$TAG-win.zip
         unzip istioctl-$TAG-win.zip

         ./istioctl install --set tag=$TAG --set hub=gcr.io/istio-testing --set values.pilot.env.ENABLE_GATEWAY_API_INFERENCE_EXTENSION=true
         ```

=== "Kgateway"

      1. Requirements

         - Gateway API [CRDs](https://gateway-api.sigs.k8s.io/guides/#installing-gateway-api) installed.

      1. Set the Kgateway version and install the Kgateway CRDs:

         ```bash
         KGTW_VERSION=v2.2.0-main
         helm upgrade -i --create-namespace --namespace kgateway-system --version $KGTW_VERSION kgateway-crds oci://cr.kgateway.dev/kgateway-dev/charts/kgateway-crds
         ```

      1. Install Kgateway:

         ```bash
         helm upgrade -i --namespace kgateway-system --version $KGTW_VERSION kgateway oci://cr.kgateway.dev/kgateway-dev/charts/kgateway --set inferenceExtension.enabled=true
         ```

### Deploy the InferencePool and Endpoint Picker Extension

   Install an InferencePool named `vllm-llama3-8b-instruct` that selects from endpoints with label `app: vllm-llama3-8b-instruct` and listening on port 8000. The Helm install command automatically installs the endpoint-picker, InferencePool along with provider specific resources.

   Set the chart version and then select a tab to follow the provider-specific instructions.

   ```bash
   export IGW_CHART_VERSION=v1.1.0
   ```

--8<-- "site-src/_includes/epp.md"

### Deploy an Inference Gateway

   Choose one of the following options to deploy an Inference Gateway.

=== "GKE"

      1. Enable the Google Kubernetes Engine API, Compute Engine API, the Network Services API and configure proxy-only subnets when necessary. 
         See [Deploy Inference Gateways](https://cloud.google.com/kubernetes-engine/docs/how-to/deploy-gke-inference-gateway)
         for detailed instructions.

      1. Deploy the Inference Gateway:

         ```bash
         kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/v1.1.0/config/manifests/gateway/gke/gateway.yaml
         ```

         Confirm that the Gateway was assigned an IP address and reports a `Programmed=True` status:

         ```bash
         $ kubectl get gateway inference-gateway
         NAME                CLASS               ADDRESS         PROGRAMMED   AGE
         inference-gateway   inference-gateway   <MY_ADDRESS>    True         22s
         ```
      1. Deploy the HTTPRoute:

         ```bash
         kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/v1.1.0/config/manifests/gateway/gke/httproute.yaml
         ```

      1. Confirm that the HTTPRoute status conditions include `Accepted=True` and `ResolvedRefs=True`:

         ```bash
         kubectl get httproute llm-route -o yaml
         ```

=== "Istio"

      Please note that this feature is currently in an experimental phase and is not intended for production use.
      The implementation and user experience are subject to changes as we continue to iterate on this project.

      1. Deploy the Inference Gateway:

         ```bash
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/istio/gateway.yaml
         ```

         Confirm that the Gateway was assigned an IP address and reports a `Programmed=True` status:
         ```bash
         $ kubectl get gateway inference-gateway
         NAME                CLASS               ADDRESS         PROGRAMMED   AGE
         inference-gateway   inference-gateway   <MY_ADDRESS>    True         22s
         ```

      1. Deploy the HTTPRoute:

         ```bash
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/istio/httproute.yaml
         ```

      1. Confirm that the HTTPRoute status conditions include `Accepted=True` and `ResolvedRefs=True`:

         ```bash
         kubectl get httproute llm-route -o yaml
         ```

=== "Kgateway"

      [Kgateway](https://kgateway.dev/) is a Gateway API and Inference Gateway
      [conformant](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/conformance/reports/v1.0.0/gateway/kgateway)
      implementation. Kgateway supports Inference Gateway with the [agentgateway](https://agentgateway.dev/) data plane. Follow these steps
      to run Kgateway as an Inference Gateway:

      1. Deploy the Inference Gateway:

         ```bash
         kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/v1.1.0/config/manifests/gateway/agentgateway/gateway.yaml
         ```

         Confirm that the Gateway was assigned an IP address and reports a `Programmed=True` status:
         ```bash
         kubectl get gateway inference-gateway
         ```

      1. Deploy the HTTPRoute:

         ```bash
         kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/v1.1.0/config/manifests/gateway/agentgateway/httproute.yaml
         ```

      1. Confirm that the HTTPRoute status conditions include `Accepted=True` and `ResolvedRefs=True`:

         ```bash
         kubectl get httproute llm-route -o yaml
         ```

### Deploy InferenceObjective (Optional)

Deploy the sample InferenceObjective which allows you to specify priority of requests.

   ```bash
   kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/v1.1.0/config/manifests/inferenceobjective.yaml
   ```

--8<-- "site-src/_includes/test.md"

--8<-- "site-src/_includes/bbr.md"

### Cleanup

   The following instructions assume you would like to cleanup ALL resources that were created in this quickstart guide.
   Please be careful not to delete resources you'd like to keep.

   1. Uninstall the InferencePool, InferenceObjective and model server resources:

      ```bash
      helm uninstall vllm-llama3-8b-instruct
      kubectl delete -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/v1.1.0/config/manifests/inferenceobjective.yaml --ignore-not-found
      kubectl delete -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/v1.1.0/config/manifests/vllm/cpu-deployment.yaml --ignore-not-found
      kubectl delete -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/v1.1.0/config/manifests/vllm/gpu-deployment.yaml --ignore-not-found
      kubectl delete -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/v1.1.0/config/manifests/vllm/sim-deployment.yaml --ignore-not-found
      kubectl delete secret hf-token --ignore-not-found
      ```

   1. Uninstall the Gateway API Inference Extension CRDs:

      ```bash
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v1.1.0/manifests.yaml --ignore-not-found
      ```
      
   1. Choose one of the following options to cleanup the Inference Gateway.

=== "GKE"

      ```bash
      kubectl delete -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/v1.1.0/config/manifests/gateway/gke/gateway.yaml --ignore-not-found
      kubectl delete -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/v1.1.0/config/manifests/gateway/gke/healthcheck.yaml --ignore-not-found
      kubectl delete -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/v1.1.0/config/manifests/gateway/gke/gcp-backend-policy.yaml --ignore-not-found
      kubectl delete -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/v1.1.0/config/manifests/gateway/gke/httproute.yaml --ignore-not-found
      ```

=== "Istio"

      ```bash
      kubectl delete -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/v1.1.0/config/manifests/gateway/istio/gateway.yaml --ignore-not-found
      kubectl delete -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/v1.1.0/config/manifests/gateway/istio/httproute.yaml --ignore-not-found
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

=== "Kgateway"

      ```bash
      kubectl delete -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/v1.1.0/config/manifests/gateway/agentgateway/gateway.yaml --ignore-not-found
      kubectl delete -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/v1.1.0/config/manifests/gateway/agentgateway/httproute.yaml --ignore-not-found
      ```

      The following steps assume you would like to cleanup ALL Kgateway resources that were created in this quickstart guide.

      1. Uninstall Kgateway:

         ```bash
         helm uninstall kgateway -n kgateway-system
         ```

      1. Uninstall the Kgateway CRDs:

         ```bash
         helm uninstall kgateway-crds -n kgateway-system
         ```

      1. Remove the Kgateway namespace:

         ```bash
         kubectl delete ns kgateway-system
         ```
