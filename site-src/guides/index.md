# Getting started with Gateway API Inference Extension

This quickstart guide is intended for engineers familiar with k8s and model servers (vLLM in this instance). The goal of this guide is to get a first, single InferencePool up and running! 

## **Prerequisites**
 - Envoy Gateway [v1.3.0](https://gateway.envoyproxy.io/docs/install/install-yaml/#install-with-yaml) or higher
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
   kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/crd/bases/inference.networking.x-k8s.io_inferencepools.yaml
   kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/crd/bases/inference.networking.x-k8s.io_inferencemodels.yaml
   ```
   
### Deploy InferenceModel

   Deploy the sample InferenceModel which is configured to load balance traffic between the `tweet-summary-0` and `tweet-summary-1`
   [LoRA adapters](https://docs.vllm.ai/en/latest/features/lora.html) of the sample model server.
   ```bash
   kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/inferencemodel.yaml
   ```

### Update Envoy Gateway Config to enable Patch Policy**

   Our custom LLM Gateway ext-proc is patched into the existing envoy gateway via `EnvoyPatchPolicy`. To enable this feature, we must extend the Envoy Gateway config map. To do this, simply run:
   ```bash
   kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/enable_patch_policy.yaml
   kubectl rollout restart deployment envoy-gateway -n envoy-gateway-system
   ```
   Additionally, if you would like to enable the admin interface, you can uncomment the admin lines and run this again.

### Deploy Gateway

   ```bash
   kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/gateway.yaml
   ```
   > **_NOTE:_** This file couples together the gateway infra and the HTTPRoute infra for a convenient, quick startup. Creating additional/different InferencePools on the same gateway will require an additional set of: `Backend`, `HTTPRoute`, the resources included in the `./config/manifests/gateway/ext-proc.yaml` file, and an additional `./config/manifests/gateway/patch_policy.yaml` file. ***Should you choose to experiment, familiarity with xDS and Envoy are very useful.***

   Confirm that the Gateway was assigned an IP address and reports a `Programmed=True` status:
   ```bash
   $ kubectl get gateway inference-gateway
   NAME                CLASS               ADDRESS         PROGRAMMED   AGE
   inference-gateway   inference-gateway   <MY_ADDRESS>    True         22s
   ```
### Deploy the InferencePool and Extension

   ```bash
   kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/inferencepool.yaml
   ```
### Deploy Envoy Gateway Custom Policies

   ```bash
   kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/patch_policy.yaml
   ```
   > **_NOTE:_** This is also per InferencePool, and will need to be configured to support the new pool should you wish to experiment further.
   
### **OPTIONALLY**: Apply Traffic Policy

   For high-traffic benchmarking you can apply this manifest to avoid any defaults that can cause timeouts/errors.

   ```bash
   kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/traffic_policy.yaml
   ```

### Try it out

   Wait until the gateway is ready.

   ```bash
   IP=$(kubectl get gateway/inference-gateway -o jsonpath='{.status.addresses[0].value}')
   PORT=8081

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
   kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/traffic_policy.yaml --ignore-not-found
   kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/extension_policy.yaml --ignore-not-found
   kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/patch_policy.yaml --ignore-not-found
   kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/inferencepool.yaml --ignore-not-found
   kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/gateway.yaml --ignore-not-found
   kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/gateway/enable_patch_policy.yaml --ignore-not-found
   kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/inferencemodel.yaml --ignore-not-found
   kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/crd/bases/inference.networking.x-k8s.io_inferencepools.yaml --ignore-not-found
   kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/crd/bases/inference.networking.x-k8s.io_inferencemodels.yaml --ignore-not-found
   kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/cpu-deployment.yaml --ignore-not-found
   kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/gpu-deployment.yaml --ignore-not-found
   kubectl delete secret hf-token --ignore-not-found
   ```
