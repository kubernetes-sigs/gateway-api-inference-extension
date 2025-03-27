# Getting started with Gateway API Inference Extension

This quickstart guide is intended for engineers familiar with k8s and model servers (vLLM in this instance). The goal of this guide is to get a first, single InferencePool up and running! 

## **Prerequisites**
 - A cluster with:
    - Support for services of type `LoadBalancer`. (This can be validated by ensuring your Envoy Gateway is up and running).
   For example, with Kind, you can follow [these steps](https://kind.sigs.k8s.io/docs/user/loadbalancer).
    - Support for [sidecar containers](https://kubernetes.io/docs/concepts/workloads/pods/sidecar-containers/) (enabled by default since Kubernetes v1.29)
   to run the model server deployment.

## **Steps**

### Deploy Sample Model Server

   Two options are supported for running the model server:

   1. GPU-based model server.  
      Requirements: a Hugging Face access token that grants access to the model [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf).

   1. CPU-based model server (not using GPUs).  
      The sample uses the model [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).  

   Choose one of these options and follow the steps below. Please do not deploy both, as the deployments have the same name and will override each other.

=== "GPU-Based Model Server"

      For this setup, you will need 3 GPUs to run the sample model server. Adjust the number of replicas in `./config/manifests/vllm/gpu-deployment.yaml` as needed.
      Create a Hugging Face secret to download the model [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf). Ensure that the token grants access to this model.
      
      Deploy a sample vLLM deployment with the proper protocol to work with the LLM Instance Gateway.
      ```bash
      kubectl create secret generic hf-token --from-literal=token=$HF_TOKEN # Your Hugging Face Token with access to Llama2
      kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/gpu-deployment.yaml
      ```

=== "CPU-Based Model Server"

      This setup is using the formal `vllm-cpu` image, which according to the documentation can run vLLM on x86 CPU platform.
      For this setup, we use approximately 9.5GB of memory and 12 CPUs for each replica.  
      While it is possible to deploy the model server with less resources, this is not recommended.  
      For example, in our tests, loading the model using 8GB of memory and 1 CPU was possible but took almost 3.5 minutes and inference requests took unreasonable time.  
      In general, there is a tradeoff between the memory and CPU we allocate to our pods and the performance. The more memory and CPU we allocate the better performance we can get.  
      After running multiple configurations of these values we decided in this sample to use 9.5GB of memory and 12 CPUs for each replica, which gives reasonable response times. You can increase those numbers and potentially may even get better response times.
      For modifying the allocated resources, adjust the numbers in `./config/manifests/vllm/cpu-deployment.yaml` as needed.  

      Deploy a sample vLLM deployment with the proper protocol to work with the LLM Instance Gateway.
      ```bash
      kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/cpu-deployment.yaml
      ```

### Install the Inference Extension CRDs

   ```bash
   kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v0.2.0/manifests.yaml
   ```

### Deploy InferenceModel

   Deploy the sample InferenceModel which is configured to load balance traffic between the `tweet-summary-0` and `tweet-summary-1`
   [LoRA adapters](https://docs.vllm.ai/en/latest/features/lora.html) of the sample model server.

   ```bash
   kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/inferencemodel.yaml
   ```

### Deploy the InferencePool and Extension

   ```bash
   kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/inferencepool.yaml
   ```

### Deploy Inference Gateway

   Choose one of the following options to deploy an Inference Gateway.

=== "GKE"

      1. Enable the Gateway API

         ```bash
         gcloud container clusters update <CLUSTER_NAME> \
             --location=<CLUSTER_LOCATION> \
             --gateway-api=standard
         ```

      1. Create the proxy-only subnet
      
         A proxy-only subnet provides a set of IP addresses that Google uses to run Envoy proxies on your behalf. 
         ```
         gcloud compute networks subnets create proxy-only-subnet \
             --purpose=REGIONAL_MANAGED_PROXY \
             --role=ACTIVE \
             --region=<REGION> \
             --network=<VPC_NETWORK_NAME> \
             --range=<CIDR_RANGE>
         ```

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

=== "Istio"

      Please note that this feature is currently in an experimental phase and is not intended for production use. 
      The implementation and user experience are subject to changes as we continue to iterate on this project.

      1. Install Istio
      
         Please follow the [Istio installation guide](https://istio.io/latest/docs/setup/install/).

      1. If you run the Endpoint Picker (EPP) with TLS (with `--secureServing=true`), it is currently using a self-signed certificate 
      and the gateway cannot successfully validate the CA signature and the SAN. Apply the destination rule to bypass verification as 
      a temporary workaround. A better TLS implementation is being discussed in https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/582.

         ```bash
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/istio/destination-rule.yaml
         ```

      1. Deploy Gateway

         ```bash
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/istio/gateway.yaml
         ```

      1. Label the gateway

         ```bash
         kubectl label gateway llm-gateway istio.io/enable-inference-extproc=true
         ```

      1. Confirm that the Gateway was assigned an IP address and reports a `Programmed=True` status:
      
         ```bash
         $ kubectl get gateway inference-gateway
         NAME                CLASS               ADDRESS         PROGRAMMED   AGE
         inference-gateway   inference-gateway   <MY_ADDRESS>    True         22s
         ```

=== "Kgateway"

      [Kgateway](https://kgateway.dev/) v2.0.0 adds support for inference extension as a **technical preview**. This means do not
      run Kgateway with inference extension in production environments. Refer to [Issue 10411](https://github.com/kgateway-dev/kgateway/issues/10411)
      for the list of caveats, supported features, etc.

      1. Requirements

         - [Helm](https://helm.sh/docs/intro/install/) installed.
         - Gateway API [CRDs](https://gateway-api.sigs.k8s.io/guides/#installing-gateway-api) installed.

      1. Install Kgateway CRDs

         ```bash
         helm upgrade -i --create-namespace --namespace kgateway-system --version v2.0.0-main kgateway-crds https://github.com/danehans/toolbox/raw/refs/heads/main/charts/338661f3be-kgateway-crds-1.0.1-dev.tgz
         ```

      1. Install Kgateway

         ```bash
         helm upgrade --install kgateway "https://github.com/danehans/toolbox/raw/refs/heads/main/charts/338661f3be-kgateway-1.0.1-dev.tgz" \
         -n kgateway-system \
         --set image.registry=danehans \
         --set image.pullPolicy=Always \
         --set inferenceExtension.enabled="true" \
         --version 1.0.1-dev
         ```

      1. Deploy Gateway

         ```bash
         kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/kgateway/gateway.yaml
         ```

         Confirm that the Gateway was assigned an IP address and reports a `Programmed=True` status:
         ```bash
         $ kubectl get gateway inference-gateway
         NAME                CLASS               ADDRESS         PROGRAMMED   AGE
         inference-gateway   kgateway            <MY_ADDRESS>    True         22s
         ```

### Deploy the HTTPRoute

   ```bash
   kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/httproute.yaml
   ```

### Try it out

   Wait until the gateway is ready.

   ```bash
   IP=$(kubectl get gateway/inference-gateway -o jsonpath='{.status.addresses[0].value}')
   PORT=80

   curl -i ${IP}:${PORT}/v1/completions -H 'Content-Type: application/json' -d '{
   "model": "tweet-summary",
   "prompt": "Write as if you were a critic: San Francisco",
   "max_tokens": 100,
   "temperature": 0
   }'
   ```

### Cleanup

   The following cleanup assumes you would like to clean ALL resources that were created in this quickstart guide.  
   please be careful not to delete resources you'd like to keep.
   ```bash
   kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/gke/gateway.yaml --ignore-not-found
   kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/gke/healthcheck.yaml --ignore-not-found
   kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/istio/gateway.yaml --ignore-not-found
   kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/istio/destination-rule.yaml --ignore-not-found
   kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/kgateway/gateway.yaml --ignore-not-found
   kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/httproute.yaml --ignore-not-found
   kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/inferencepool.yaml --ignore-not-found
   kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/inferencemodel.yaml --ignore-not-found
   kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/crd/bases/inference.networking.x-k8s.io_inferencepools.yaml --ignore-not-found
   kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/crd/bases/inference.networking.x-k8s.io_inferencemodels.yaml --ignore-not-found
   kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/cpu-deployment.yaml --ignore-not-found
   kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/gpu-deployment.yaml --ignore-not-found
   kubectl delete secret hf-token --ignore-not-found
   ```
