# Getting started with an Inference Gateway

??? example "Experimental"

    This project is still in an alpha state and breaking changes may occur in the future.


This quickstart guide is intended for engineers familiar with k8s and model servers (vLLM in this instance). The goal of this guide is to get an Inference Gateway up and running!

## **Prerequisites**

A cluster with:
  - Support for services of type `LoadBalancer`. For kind clusters, follow [this guide](https://kind.sigs.k8s.io/docs/user/loadbalancer)
  to get services of type LoadBalancer working.
  - Support for [sidecar containers](https://kubernetes.io/docs/concepts/workloads/pods/sidecar-containers/) (enabled by default since Kubernetes v1.29)
  to run the model server deployment.

Tooling:
  - [Helm](https://helm.sh/docs/intro/install/) installed

## **Steps**

### Deploy Sample Model Server

   Three options are supported for running the model server:

   1. GPU-based model server.
      Requirements: a Hugging Face access token that grants access to the model [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

   1. CPU-based model server (not using GPUs).
      The sample uses the model [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).

   1. [vLLM Simulator](https://github.com/llm-d/llm-d-inference-sim/tree/main) model server (not using GPUs).
      The sample is configured to simulate the [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) model.

   Choose one of these options and follow the steps below. Please do not deploy more than one, as the deployments have the same name and will override each other.

=== "GPU-Based Model Server"

    For this setup, you will need 3 GPUs to run the sample model server. Adjust the number of replicas in `./config/manifests/vllm/gpu-deployment.yaml` as needed.
    Create a Hugging Face secret to download the model [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct). Ensure that the token grants access to this model.

    Deploy a sample vLLM deployment with the proper protocol to work with the LLM Instance Gateway.

    ```bash
    kubectl create secret generic hf-token --from-literal=token=$HF_TOKEN # Your Hugging Face Token with access to the set of Llama models
    kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/gpu-deployment.yaml
    ```

=== "CPU-Based Model Server"

    ???+ warning

        CPU deployment can be unreliable i.e. the pods may crash/restart because of resource contraints.

    This setup is using the formal `vllm-cpu` image, which according to the documentation can run vLLM on x86 CPU platform.
    For this setup, we use approximately 9.5GB of memory and 12 CPUs for each replica.

    While it is possible to deploy the model server with less resources, this is not recommended. For example, in our tests, loading the model using 8GB of memory and 1 CPU was possible but took almost 3.5 minutes and inference requests took unreasonable time. In general, there is a tradeoff between the memory and CPU we allocate to our pods and the performance. The more memory and CPU we allocate the better performance we can get.

    After running multiple configurations of these values we decided in this sample to use 9.5GB of memory and 12 CPUs for each replica, which gives reasonable response times. You can increase those numbers and potentially may even get better response times. For modifying the allocated resources, adjust the numbers in [cpu-deployment.yaml](https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/cpu-deployment.yaml) as needed.

    Deploy a sample vLLM deployment with the proper protocol to work with the LLM Instance Gateway.

    ```bash
    kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/cpu-deployment.yaml
    ```

=== "vLLM Simulator Model Server"

    This option uses the [vLLM simulator](https://github.com/llm-d/llm-d-inference-sim/tree/main) to simulate a backend model server.
    This setup uses the least amount of compute resources, does not require GPU's, and is ideal for test/dev environments.

    To deploy the vLLM simulator, run the following command.

    ```bash
    kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/sim-deployment.yaml
    ```

### Install the Inference Extension CRDs

   ```bash
   kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/latest/download/manifests.yaml
   ```

### Deploy the InferencePool and Endpoint Picker Extension

   Install an InferencePool named `vllm-llama3-8b-instruct` that selects from endpoints with label `app: vllm-llama3-8b-instruct` and listening on port 8000. The Helm install command automatically installs the endpoint-picker, inferencepool along with provider specific resources.

=== "GKE"

      ```bash
      export GATEWAY_PROVIDER=gke
      export IGW_CHART_VERSION=v1.0.1
      helm install vllm-llama3-8b-instruct \
      --set inferencePool.modelServers.matchLabels.app=vllm-llama3-8b-instruct \
      --set provider.name=$GATEWAY_PROVIDER \
      --version $IGW_CHART_VERSION \
      oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool
      ```

=== "Istio"

      ```bash
      export GATEWAY_PROVIDER=istio
      export IGW_CHART_VERSION=v1.0.1
      helm install vllm-llama3-8b-instruct \
      --set inferencePool.modelServers.matchLabels.app=vllm-llama3-8b-instruct \
      --set provider.name=$GATEWAY_PROVIDER \
      --version $IGW_CHART_VERSION \
      oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool
      ```

=== "Kgateway"

      ```bash
      export GATEWAY_PROVIDER=none
      export IGW_CHART_VERSION=v1.0.1
      helm install vllm-llama3-8b-instruct \
      --set inferencePool.modelServers.matchLabels.app=vllm-llama3-8b-instruct \
      --set provider.name=$GATEWAY_PROVIDER \
      --version $IGW_CHART_VERSION \
      oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool
      ```

=== "Agentgateway"

      ```bash
      export GATEWAY_PROVIDER=none
      export IGW_CHART_VERSION=v1.0.1
      helm install vllm-llama3-8b-instruct \
      --set inferencePool.modelServers.matchLabels.app=vllm-llama3-8b-instruct \
      --set provider.name=$GATEWAY_PROVIDER \
      --version $IGW_CHART_VERSION \
      oci://registry.k8s.io/gateway-api-inference-extension/charts/inferencepool
      ```

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

      3. Deploy Gateway

         ```bash
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/istio/gateway.yaml
         ```

         Confirm that the Gateway was assigned an IP address and reports a `Programmed=True` status:
         ```bash
         $ kubectl get gateway inference-gateway
         NAME                CLASS               ADDRESS         PROGRAMMED   AGE
         inference-gateway   inference-gateway   <MY_ADDRESS>    True         22s
         ```

      4. Deploy the HTTPRoute

         ```bash
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/istio/httproute.yaml
         ```

      5. Confirm that the HTTPRoute status conditions include `Accepted=True` and `ResolvedRefs=True`:

         ```bash
         kubectl get httproute llm-route -o yaml
         ```

=== "Kgateway"

      [Kgateway](https://kgateway.dev/) added Inference Gateway support as a **technical preview** in the
      [v2.0.0 release](https://github.com/kgateway-dev/kgateway/releases/tag/v2.0.0). InferencePool v1.0.0 is currently supported in the latest [rolling release](https://github.com/kgateway-dev/kgateway/releases/tag/v2.1.0-main), which includes the latest changes but may be unstable until the [v2.1.0 release](https://github.com/kgateway-dev/kgateway/milestone/58) is published.

      1. Requirements

         - [Helm](https://helm.sh/docs/intro/install/) installed.
         - Gateway API [CRDs](https://gateway-api.sigs.k8s.io/guides/#installing-gateway-api) installed.

      2. Set the Kgateway version and install the Kgateway CRDs.

         ```bash
         KGTW_VERSION=v2.1.0-main
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

      [Agentgateway](https://agentgateway.dev/) is a purpose-built proxy designed for AI workloads, and comes with native support for Inference Gateway. Agentgateway integrates with [Kgateway](https://kgateway.dev/) as it's control plane. InferencePool v1.0.0 is currently supported in the latest [rolling release](https://github.com/kgateway-dev/kgateway/releases/tag/v2.1.0-main), which includes the latest changes but may be unstable until the [v2.1.0 release](https://github.com/kgateway-dev/kgateway/milestone/58) is published.

      1. Requirements

         - [Helm](https://helm.sh/docs/intro/install/) installed.
         - Gateway API [CRDs](https://gateway-api.sigs.k8s.io/guides/#installing-gateway-api) installed.

      2. Set the Kgateway version and install the Kgateway CRDs.

         ```bash
         KGTW_VERSION=v2.1.0-main
         helm upgrade -i --create-namespace --namespace kgateway-system --version $KGTW_VERSION kgateway-crds oci://cr.kgateway.dev/kgateway-dev/charts/kgateway-crds
         ```

      3. Install Kgateway

         ```bash
         helm upgrade -i --namespace kgateway-system --version $KGTW_VERSION kgateway oci://cr.kgateway.dev/kgateway-dev/charts/kgateway --set inferenceExtension.enabled=true --set agentGateway.enabled=true
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

### Try it out

   Wait until the gateway is ready.

   ```bash
   IP=$(kubectl get gateway/inference-gateway -o jsonpath='{.status.addresses[0].value}')
   PORT=80

   curl -i ${IP}:${PORT}/v1/completions -H 'Content-Type: application/json' -d '{
   "model": "food-review-1",
   "prompt": "Write as if you were a critic: San Francisco",
   "max_tokens": 100,
   "temperature": 0
   }'
   ```

### Deploy the Body Based Router Extension (Optional)

This guide has shown how to get started with serving a single base model type per L7 URL path. If after this exercise, you wish to continue on to exercise model-aware routing such that more than 1 base model is served at the same L7 url path, that requires use of the (optional) Body Based Routing (BBR) extension which is described in a separate section of the documentation, namely the [`Serving Multiple GenAI Models`](serve-multiple-genai-models.md) section. If you wish to exercise that function, then retain the setup you have deployed so far from this guide and move on to the additional steps described in [that guide](serve-multiple-genai-models.md) or else move on to the following section to cleanup your setup. 

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
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/gke/healthcheck.yaml --ignore-not-found
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/gke/gcp-backend-policy.yaml --ignore-not-found
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

      1. Remove the Istio namespace

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

      1. Uninstall the Kgateway CRDs.

         ```bash
         helm uninstall kgateway-crds -n kgateway-system
         ```

      1. Remove the Kgateway namespace.

         ```bash
         kubectl delete ns kgateway-system
         ```

=== "Agentgateway"

      ```bash
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/kgateway/gateway.yaml --ignore-not-found
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/kgateway/httproute.yaml --ignore-not-found
      ```

      The following steps assume you would like to cleanup ALL Kgateway resources that were created in this quickstart guide.

      1. Uninstall Kgateway

         ```bash
         helm uninstall kgateway -n kgateway-system
         ```

      1. Uninstall the Kgateway CRDs.

         ```bash
         helm uninstall kgateway-crds -n kgateway-system
         ```

      1. Remove the Kgateway namespace.

         ```bash
         kubectl delete ns kgateway-system
         ```
