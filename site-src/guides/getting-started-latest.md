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

      2. Deploy Inference Gateway:

         ```bash
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/gke/gateway.yaml
         ```

         Confirm that the Gateway was assigned an IP address and reports a `Programmed=True` status:

         ```bash
         $ kubectl get gateway inference-gateway
         NAME                CLASS               ADDRESS         PROGRAMMED   AGE
         inference-gateway   inference-gateway   <MY_ADDRESS>    True         22s
         ```
      3. Deploy the HTTPRoute

         ```bash
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/gke/httproute.yaml
         ```

      4. Confirm that the HTTPRoute status conditions include `Accepted=True` and `ResolvedRefs=True`:

         ```bash
         kubectl get httproute llm-route -o yaml
         ```

=== "Istio"

      Please note that this feature is currently in an experimental phase and is not intended for production use.
      The implementation and user experience are subject to changes as we continue to iterate on this project.

      1.  Requirements

         - Gateway API [CRDs](https://gateway-api.sigs.k8s.io/guides/#installing-gateway-api) installed.

      2. Install Istio

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

      3. If your EPP uses secure serving with self-signed certs (default), temporarily bypass TLS verification:

         ```bash
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/istio/destination-rule.yaml
         ```
 
      4. Deploy Gateway

         ```bash
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/istio/gateway.yaml
         ```

         Confirm that the Gateway was assigned an IP address and reports a `Programmed=True` status:
         ```bash
         $ kubectl get gateway inference-gateway
         NAME                CLASS               ADDRESS         PROGRAMMED   AGE
         inference-gateway   inference-gateway   <MY_ADDRESS>    True         22s
         ```

      5. Deploy the HTTPRoute

         ```bash
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/istio/httproute.yaml
         ```

      6. Confirm that the HTTPRoute status conditions include `Accepted=True` and `ResolvedRefs=True`:

         ```bash
         kubectl get httproute llm-route -o yaml
         ```

=== "Kgateway"

      [Kgateway](https://kgateway.dev/) is a Gateway API and Inference Gateway
      [conformant](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/conformance/reports/v1.0.0/gateway/kgateway)
      gateway. Follow these steps to run Kgateway:

      1. Requirements

         - [Helm](https://helm.sh/docs/intro/install/) installed.
         - Gateway API [CRDs](https://gateway-api.sigs.k8s.io/guides/#installing-gateway-api) installed.

      2. Set the Kgateway version and install the Kgateway CRDs.

         ```bash
         KGTW_VERSION=v2.2.0-main
         helm upgrade -i --create-namespace --namespace kgateway-system --version $KGTW_VERSION kgateway-crds oci://cr.kgateway.dev/kgateway-dev/charts/kgateway-crds
         ```

      3. Install Kgateway

         ```bash
         helm upgrade -i --namespace kgateway-system --version $KGTW_VERSION kgateway oci://cr.kgateway.dev/kgateway-dev/charts/kgateway --set inferenceExtension.enabled=true
         ```

      4. Deploy the Gateway

         ```bash
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/kgateway/gateway.yaml
         ```

         Confirm that the Gateway was assigned an IP address and reports a `Programmed=True` status:
         ```bash
         $ kubectl get gateway inference-gateway
         NAME                CLASS               ADDRESS         PROGRAMMED   AGE
         inference-gateway   kgateway            <MY_ADDRESS>    True         22s
         ```

      5. Deploy the HTTPRoute

         ```bash
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/kgateway/httproute.yaml
         ```

      6. Confirm that the HTTPRoute status conditions include `Accepted=True` and `ResolvedRefs=True`:

         ```bash
         kubectl get httproute llm-route -o yaml
         ```

=== "Agentgateway"

      [Agentgateway](https://agentgateway.dev/) is a purpose-built proxy designed for AI workloads, and comes with native support for Inference Gateway.
      Agentgateway integrates with [Kgateway](https://kgateway.dev/) as it's control plane. Follow these steps to run Kgateway with the agentgateway
      data plane:

      1. Requirements

         - [Helm](https://helm.sh/docs/intro/install/) installed.
         - Gateway API [CRDs](https://gateway-api.sigs.k8s.io/guides/#installing-gateway-api) installed.

      2. Set the Kgateway version and install the Kgateway CRDs.

         ```bash
         KGTW_VERSION=v2.2.0-main
         helm upgrade -i --create-namespace --namespace kgateway-system --version $KGTW_VERSION kgateway-crds oci://cr.kgateway.dev/kgateway-dev/charts/kgateway-crds
         ```

      3. Install Kgateway

         ```bash
         helm upgrade -i --namespace kgateway-system --version $KGTW_VERSION kgateway oci://cr.kgateway.dev/kgateway-dev/charts/kgateway --set inferenceExtension.enabled=true --set agentgateway.enabled=true
         ```

      4. Deploy the Gateway

         ```bash
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/agentgateway/gateway.yaml
         ```

         Confirm that the Gateway was assigned an IP address and reports a `Programmed=True` status:
         ```bash
         $ kubectl get gateway inference-gateway
         NAME                CLASS               ADDRESS         PROGRAMMED   AGE
         inference-gateway   agentgateway        <MY_ADDRESS>    True         22s
         ```

      5. Deploy the HTTPRoute

         ```bash
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/agentgateway/httproute.yaml
         ```

      6. Confirm that the HTTPRoute status conditions include `Accepted=True` and `ResolvedRefs=True`:

         ```bash
         kubectl get httproute llm-route -o yaml
         ```

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

   1. Uninstall the InferencePool, InferenceObjective and model server resources

      ```bash
      helm uninstall vllm-llama3-8b-instruct
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/inferenceobjective.yaml --ignore-not-found
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/cpu-deployment.yaml --ignore-not-found
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/gpu-deployment.yaml --ignore-not-found
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/sim-deployment.yaml --ignore-not-found
      kubectl delete secret hf-token --ignore-not-found
      ```

   1. Uninstall the Gateway API Inference Extension CRDs

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
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/istio/destination-rule.yaml --ignore-not-found
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/istio/httproute.yaml --ignore-not-found

      ```

      The following steps assume you would like to clean up ALL Istio resources that were created in this quickstart guide.

      1. Uninstall All Istio resources

         ```bash
         istioctl uninstall -y --purge
         ```

      2. Remove the Istio namespace

         ```bash
         kubectl delete ns istio-system
         ```

=== "Kgateway"

      ```bash
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/kgateway/gateway.yaml --ignore-not-found
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/kgateway/httproute.yaml --ignore-not-found
      ```

      The following steps assume you would like to cleanup ALL Kgateway resources that were created in this quickstart guide.

      1. Uninstall Kgateway

         ```bash
         helm uninstall kgateway -n kgateway-system
         ```

      2. Uninstall the Kgateway CRDs.

         ```bash
         helm uninstall kgateway-crds -n kgateway-system
         ```

      3. Remove the Kgateway namespace.

         ```bash
         kubectl delete ns kgateway-system
         ```

=== "Agentgateway"

      ```bash
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/agentgateway/gateway.yaml --ignore-not-found
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/agentgateway/httproute.yaml --ignore-not-found
      ```

      The following steps assume you would like to cleanup ALL Kgateway resources that were created in this quickstart guide.

      1. Uninstall Kgateway

         ```bash
         helm uninstall kgateway -n kgateway-system
         ```

      2. Uninstall the Kgateway CRDs.

         ```bash
         helm uninstall kgateway-crds -n kgateway-system
         ```

      3. Remove the Kgateway namespace.

         ```bash
         kubectl delete ns kgateway-system
         ```
