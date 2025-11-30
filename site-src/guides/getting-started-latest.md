<!-- If you are updating this getting-started-latest.md guide, please make sure to update the index.md as well -->

# Getting started with an Inference Gateway

!!! warning "Unreleased/main branch"
    This guide tracks **main**. It is intended for users who want the very latest features and fixes and are comfortable with potential breakage.
    For the stable, tagged experience, see **Getting started (Released)**.

--8<-- "site-src/_includes/intro.md"

## **Prerequisites**

--8<-- "site-src/_includes/prereqs.md"

## **Steps**

### Deploy Sample Model Server

--8<-- "site-src/_includes/model-server-intro.md"

--8<-- "site-src/_includes/model-server-gpu.md"

    ```bash
    kubectl create secret generic hf-token --from-literal=token=$HF_TOKEN # Your Hugging Face Token with access to the set of Llama models
    kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/gpu-deployment.yaml
    ```

--8<-- "site-src/_includes/model-server-cpu.md"

    ```bash
    kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/cpu-deployment.yaml
    ```

--8<-- "site-src/_includes/model-server-sim.md"

    ```bash
    kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/sim-deployment.yaml
    ```

### Install the Inference Extension CRDs

```bash
kubectl apply -k https://github.com/kubernetes-sigs/gateway-api-inference-extension/config/crd
```

### Install the Gateway

   Choose one of the following options to install Gateway.

=== "GKE"

      Nothing to install here, you can move to the next [section](#deploy-the-inferencepool-and-endpoint-picker-extension)

=== "Istio"

      1. Requirements
         - Gateway API [CRDs](https://gateway-api.sigs.k8s.io/guides/#installing-gateway-api) installed.

      2. Install Istio:
   
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

=== "Kgateway"

      1. Requirements

         - Gateway API [CRDs](https://gateway-api.sigs.k8s.io/guides/#installing-gateway-api) installed.

      2. Set the Kgateway version and install the Kgateway CRDs:

         ```bash
         KGTW_VERSION=v2.1.0
         helm upgrade -i --create-namespace --namespace kgateway-system --version $KGTW_VERSION kgateway-crds oci://cr.kgateway.dev/kgateway-dev/charts/kgateway-crds
         ```

      3. Install Kgateway:

         ```bash
         helm upgrade -i --namespace kgateway-system --version $KGTW_VERSION kgateway oci://cr.kgateway.dev/kgateway-dev/charts/kgateway --set inferenceExtension.enabled=true
         ```

### Deploy the InferencePool and Endpoint Picker Extension

   Install an InferencePool named `vllm-llama3-8b-instruct` that selects from endpoints with label `app: vllm-llama3-8b-instruct` and listening on port 8000. The Helm install command automatically installs the endpoint-picker, InferencePool along with provider specific resources.

   Set the chart version and then select a tab to follow the provider-specific instructions.

   ```bash
   export IGW_CHART_VERSION=v0
   ```

--8<-- "site-src/_includes/epp-latest.md"

### Deploy an Inference Gateway

   Choose one of the following options to deploy an Inference Gateway.

=== "GKE"

      1. Enable the Google Kubernetes Engine API, Compute Engine API, the Network Services API and configure proxy-only subnets when necessary. 
         See [Deploy Inference Gateways](https://cloud.google.com/kubernetes-engine/docs/how-to/deploy-gke-inference-gateway)
         for detailed instructions.

      1. Deploy the Inference Gateway:

         ```bash
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/gke/gateway.yaml
         ```

         Confirm that the Gateway was assigned an IP address and reports a `Programmed=True` status:

         ```bash
         $ kubectl get gateway inference-gateway
         NAME                CLASS               ADDRESS         PROGRAMMED   AGE
         inference-gateway   inference-gateway   <MY_ADDRESS>    True         22s
         ```
      1. Deploy the HTTPRoute:

         ```bash
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/gke/httproute.yaml
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
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/kgateway/gateway.yaml
         ```

         Confirm that the Gateway was assigned an IP address and reports a `Programmed=True` status:
         ```bash
         kubectl get gateway inference-gateway
         ```

      1. Deploy the HTTPRoute:

         ```bash
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/kgateway/httproute.yaml
         ```

      1. Confirm that the HTTPRoute status conditions include `Accepted=True` and `ResolvedRefs=True`:

         ```bash
         kubectl get httproute llm-route -o yaml
         ```

=== "NGINX Gateway Fabric"

      NGINX Gateway Fabric is an implementation of the Gateway API that supports the Inference Extension. Follow these steps to deploy an Inference Gateway using NGINX Gateway Fabric.

      1. Requirements

         - Gateway API [CRDs](https://gateway-api.sigs.k8s.io/guides/#installing-gateway-api) installed (Standard or Experimental channel).
         - [Helm](https://helm.sh/docs/intro/install/) installed.
         - A Kubernetes cluster with LoadBalancer or NodePort access.

      2. Install NGINX Gateway Fabric with the Inference Extension enabled by setting the `nginxGateway.gwAPIInferenceExtension.enable=true` Helm value

         ```bash 
         helm install ngf oci://ghcr.io/nginx/charts/nginx-gateway-fabric --create-namespace -n nginx-gateway --set nginxGateway.gwAPIInferenceExtension.enable=true
         ```
         This enables NGINX Gateway Fabric to watch and manage Inference Extension resources such as InferencePool and InferenceObjective.

      3. Deploy the Gateway

         ```bash
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/nginxgatewayfabric/gateway.yaml
         ```

      4. Verify the Gateway status
         
         Ensure that the Gateway is running and has been assigned an address:

         ```bash
         kubectl get gateway inference-gateway
         ```

         Check that the Gateway has been successfully provisioned and that its status shows Programmed=True
      
      5. Deploy the HTTPRoute
         
         Create the HTTPRoute resource to route traffic to your InferencePool:

         ```bash
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/nginxgatewayfabric/httproute.yaml
         ```

      6. Verify the route status

         Check that the HTTPRoute was successfully configured and references were resolved:

         ```bash
         kubectl get httproute llm-route -o yaml
         ```
         
         The route status should include Accepted=True and ResolvedRefs=True.

      7. Verify the InferencePool Status

         Make sure the InferencePool is active before sending traffic.

         ```bash
         kubectl describe inferencepools.inference.networking.k8s.io vllm-llama3-8b-instruct
         ```

         Check that the status shows Accepted=True and ResolvedRefs=True. This confirms the InferencePool is ready to handle traffic.
      
       For more information, see the [NGINX Gateway Fabric - Inference Gateway Setup guide](https://docs.nginx.com/nginx-gateway-fabric/how-to/gateway-api-inference-extension/#overview)

### Deploy InferenceObjective (Optional)

Deploy the sample InferenceObjective which allows you to specify priority of requests.

   ```bash
   kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/inferenceobjective.yaml
   ```

--8<-- "site-src/_includes/test.md"

--8<-- "site-src/_includes/bbr.md"

### Cleanup

   The following instructions assume you would like to cleanup ALL resources that were created in this quickstart guide.
   Please be careful not to delete resources you'd like to keep.

   1. Uninstall the InferencePool, InferenceObjective and model server resources:

      ```bash
      helm uninstall vllm-llama3-8b-instruct
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/inferenceobjective.yaml --ignore-not-found
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/cpu-deployment.yaml --ignore-not-found
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/gpu-deployment.yaml --ignore-not-found
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/sim-deployment.yaml --ignore-not-found
      kubectl delete secret hf-token --ignore-not-found
      ```

   1. Uninstall the Gateway API Inference Extension CRDs:

      ```bash
      kubectl delete -k https://github.com/kubernetes-sigs/gateway-api-inference-extension/config/crd --ignore-not-found
      ```
      
   1. Choose one of the following options to cleanup the Inference Gateway.

=== "GKE"

      ```bash
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/gke/gateway.yaml --ignore-not-found
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/gke/httproute.yaml --ignore-not-found
      ```

=== "Istio"

      ```bash
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/istio/gateway.yaml --ignore-not-found
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/istio/httproute.yaml --ignore-not-found

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
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/kgateway/gateway.yaml --ignore-not-found
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/kgateway/httproute.yaml --ignore-not-found
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

=== "NGINX Gateway Fabric"

      Follow these steps to remove the NGINX Gateway Fabric Inference Gateway and all related resources.


      1. Remove Inference Gateway and HTTPRoute:

         ```bash
         kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/nginxgatewayfabric/gateway.yaml --ignore-not-found
         kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/nginxgatewayfabric/httproute.yaml --ignore-not-found
         ```

      2. Uninstall NGINX Gateway Fabric:

         ```bash
         helm uninstall ngf -n nginx-gateway
         ```

      3. Clean up namespace:
   
         ```bash
         kubectl delete ns nginx-gateway
         ```