# Getting started with Gateway API Inference Extension

??? example "Experimental"

    This project is still in an alpha state and breaking changes may occur in the future.

This quickstart guide is intended for engineers familiar with k8s and model servers (vLLM in this instance). The goal of this guide is to get an Inference Gateway up and running! 

## **Prerequisites**

- A cluster with:
  - Support for services of type `LoadBalancer`. For kind clusters, follow [this guide](https://kind.sigs.k8s.io/docs/user/loadbalancer)
  to get services of type LoadBalancer working.
  - Support for [sidecar containers](https://kubernetes.io/docs/concepts/workloads/pods/sidecar-containers/) (enabled by default since Kubernetes v1.29)
  to run the model server deployment.

## **Steps**

### Deploy Sample Model Server

   Two options are supported for running the model server:

   1. GPU-based model server.  
      Requirements: a Hugging Face access token that grants access to the model [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

   1. CPU-based model server (not using GPUs).  
      The sample uses the model [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).  

   Choose one of these options and follow the steps below. Please do not deploy both, as the deployments have the same name and will override each other.

=== "GPU-Based Model Server"

      For this setup, you will need 3 GPUs to run the sample model server. Adjust the number of replicas in `./config/manifests/vllm/gpu-deployment.yaml` as needed.
      Create a Hugging Face secret to download the model [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct). Ensure that the token grants access to this model.
      
      Deploy a sample vLLM deployment with the proper protocol to work with the LLM Instance Gateway.
      ```bash
      kubectl create secret generic hf-token --from-literal=token=$HF_TOKEN # Your Hugging Face Token with access to the set of Llama models
      kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/gpu-deployment.yaml
      ```

=== "CPU-Based Model Server"

      This setup is using the formal `vllm-cpu` image, which according to the documentation can run vLLM on x86 CPU platform.
      For this setup, we use approximately 9.5GB of memory and 12 CPUs for each replica.  
      
      While it is possible to deploy the model server with less resources, this is not recommended. For example, in our tests, loading the model using 8GB of memory and 1 CPU was possible but took almost 3.5 minutes and inference requests took unreasonable time. In general, there is a tradeoff between the memory and CPU we allocate to our pods and the performance. The more memory and CPU we allocate the better performance we can get.
      
      After running multiple configurations of these values we decided in this sample to use 9.5GB of memory and 12 CPUs for each replica, which gives reasonable response times. You can increase those numbers and potentially may even get better response times. For modifying the allocated resources, adjust the numbers in [cpu-deployment.yaml](https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/cpu-deployment.yaml) as needed.  

      Deploy a sample vLLM deployment with the proper protocol to work with the LLM Instance Gateway.
      ```bash
      kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/cpu-deployment.yaml
      ```

### Install the Inference Extension CRDs

=== "Latest Release"

      ```bash
      VERSION=v0.3.0
      kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/$VERSION/manifests.yaml
      ```

=== "Dev Version"

      ```bash
      kubectl apply -k https://github.com/kubernetes-sigs/gateway-api-inference-extension/config/crd
      ```

### Deploy InferenceModel

   Deploy the sample InferenceModel which is configured to forward traffic to the `food-review-1` [LoRA adapter](https://docs.vllm.ai/en/latest/features/lora.html) of the sample model server.

   ```bash
   kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/inferencemodel.yaml
   ```

### Deploy the InferencePool and Extension

   ```bash
   kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/inferencepool-resources.yaml
   ```

### Deploy Inference Gateway

   Choose one of the following options to deploy an Inference Gateway.

=== "GKE"

      1. Enable the Gateway API and configure proxy-only subnets when necessary. See [Deploy Gateways](https://cloud.google.com/kubernetes-engine/docs/how-to/deploying-gateways) 
      for detailed instructions.

      1. Deploy Gateway and HealthCheckPolicy resources

         ```bash
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/gke/gateway.yaml
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/gke/healthcheck.yaml
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

      5. Given that the default connection timeout may be insufficient for most inference workloads, it is recommended to configure a timeout appropriate for your intended use case.

         ```bash
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/gke/gcp-backend-policy.yaml
         ```

=== "Istio"

      Please note that this feature is currently in an experimental phase and is not intended for production use. 
      The implementation and user experience are subject to changes as we continue to iterate on this project.

      1.  Requirements

         - Gateway API [CRDs](https://gateway-api.sigs.k8s.io/guides/#installing-gateway-api) installed.

      2. Install Istio
      
         ```
         TAG=1.26-alpha.80c74f7f43482c226f4f4b10b4dda6261b67a71f
         # on Linux
         wget https://storage.googleapis.com/istio-build/dev/$TAG/istioctl-$TAG-linux-amd64.tar.gz
         tar -xvf istioctl-$TAG-linux-amd64.tar.gz
         # on macOS
         wget https://storage.googleapis.com/istio-build/dev/$TAG/istioctl-$TAG-osx.tar.gz
         tar -xvf istioctl-$TAG-osx.tar.gz
         # on Windows
         wget https://storage.googleapis.com/istio-build/dev/$TAG/istioctl-$TAG-win.zip
         unzip istioctl-$TAG-win.zip

         ./istioctl install --set tag=$TAG --set hub=gcr.io/istio-testing
         ```

      3. If you run the Endpoint Picker (EPP) with the `--secureServing` flag set to `true` (the default mode), it is currently using a self-signed certificate. As a security measure, Istio does not trust self-signed certificates by default. As a temporary workaround, you can apply the destination rule to bypass TLS verification for EPP. A more secure TLS implementation in EPP is being discussed in [Issue 582](https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/582).

         ```bash
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/istio/destination-rule.yaml
         ```

      4. Deploy Gateway

         ```bash
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/istio/gateway.yaml
         ```

      5. Label the gateway

         ```bash
         kubectl label gateway llm-gateway istio.io/enable-inference-extproc=true
         ```

         Confirm that the Gateway was assigned an IP address and reports a `Programmed=True` status:
         ```bash
         $ kubectl get gateway inference-gateway
         NAME                CLASS               ADDRESS         PROGRAMMED   AGE
         inference-gateway   inference-gateway   <MY_ADDRESS>    True         22s
         ```

      6. Deploy the HTTPRoute

         ```bash
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/istio/httproute.yaml
         ```

      7. Confirm that the HTTPRoute status conditions include `Accepted=True` and `ResolvedRefs=True`:

         ```bash
         kubectl get httproute llm-route -o yaml
         ```

=== "Kgateway"

      [Kgateway](https://kgateway.dev/) recently added support for inference extension as a **technical preview**. This means do not
      run Kgateway with inference extension in production environments. Refer to [Issue 10411](https://github.com/kgateway-dev/kgateway/issues/10411)
      for the list of caveats, supported features, etc.

      1. Requirements

         - [Helm](https://helm.sh/docs/intro/install/) installed.
         - Gateway API [CRDs](https://gateway-api.sigs.k8s.io/guides/#installing-gateway-api) installed.

      2. Set the Kgateway version and install the Kgateway CRDs.

         ```bash
         KGTW_VERSION=v2.0.0
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

### Try it out

   Wait until the gateway is ready.

=== "GPU-Based Model Server"

      ```bash
      IP=$(kubectl get gateway/inference-gateway -o jsonpath='{.status.addresses[0].value}')
      PORT=80

      curl -i ${IP}:${PORT}/v1/completions -H 'Content-Type: application/json' -d '{
      "model": "food-review",
      "prompt": "Write as if you were a critic: San Francisco",
      "max_tokens": 100,
      "temperature": 0
      }'
      ```

=== "CPU-Based Model Server"

      ```bash
      IP=$(kubectl get gateway/inference-gateway -o jsonpath='{.status.addresses[0].value}')
      PORT=80

      curl -i ${IP}:${PORT}/v1/completions -H 'Content-Type: application/json' -d '{
      "model": "Qwen/Qwen2.5-1.5B-Instruct",
      "prompt": "Write as if you were a critic: San Francisco",
      "max_tokens": 100,
      "temperature": 0
      }'
      ```

### Cleanup

   The following instructions assume you would like to cleanup ALL resources that were created in this quickstart guide.
   Please be careful not to delete resources you'd like to keep.

   1. Uninstall the InferencePool, InferenceModel, and model server resources

      ```bash
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/inferencepool-resources.yaml --ignore-not-found
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/inferencemodel.yaml --ignore-not-found
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/cpu-deployment.yaml --ignore-not-found
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/gpu-deployment.yaml --ignore-not-found
      kubectl delete secret hf-token --ignore-not-found
      ```

   1. Uninstall the Gateway API resources

      ```bash
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/gke/gateway.yaml --ignore-not-found
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/gke/healthcheck.yaml --ignore-not-found
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/gke/gcp-backend-policy.yaml --ignore-not-found
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/gke/httproute.yaml --ignore-not-found
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/istio/gateway.yaml --ignore-not-found
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/istio/destination-rule.yaml --ignore-not-found
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/istio/httproute.yaml --ignore-not-found
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/kgateway/gateway.yaml --ignore-not-found
      kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/kgateway/httproute.yaml --ignore-not-found
      ```

   1. Uninstall the Gateway API Inference Extension CRDs

      ```bash
      kubectl delete -k https://github.com/kubernetes-sigs/gateway-api-inference-extension/config/crd --ignore-not-found
      ```

   1. Choose one of the following options to cleanup the Inference Gateway.

=== "GKE"

      **TODO**

=== "Istio"

      **TODO**

=== "Kgateway"

      The following instructions assume you would like to cleanup ALL Kgateway resources that were created in this quickstart guide.

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
