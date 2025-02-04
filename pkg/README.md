## Quickstart

This quickstart guide is intended for engineers familiar with k8s and model servers (vLLM in this instance). The goal of this guide is to get a first, single InferencePool up and running! 

### Requirements
 - Envoy Gateway [v1.2.1](https://gateway.envoyproxy.io/docs/install/install-yaml/#install-with-yaml) or higher
 - A cluster with:
   - Support for Services of type `LoadBalancer`. (This can be validated by ensuring your Envoy Gateway is up and running). For example, with Kind,
     you can follow [these steps](https://kind.sigs.k8s.io/docs/user/loadbalancer).
   - 3 GPUs to run the sample model server. Adjust the number of replicas in `./manifests/vllm/deployment.yaml` as needed.

### Steps

1. **Deploy Sample Model Server**

   Create a Hugging Face secret to download the model [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf). Ensure that the token grants access to this model.
   Deploy a sample vLLM deployment with the proper protocol to work with the LLM Instance Gateway.
   ```bash
   kubectl create secret generic hf-token --from-literal=token=$HF_TOKEN # Your Hugging Face Token with access to Llama2
   kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/pkg/manifests/vllm/deployment.yaml
   ```

1. **Install the Inference Extension CRDs:**

   ```sh
   kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v0.1.0-rc1/manifests.yaml
   ```

1. **Deploy InferenceModel**

   Deploy the sample InferenceModel which is configured to load balance traffic between the `tweet-summary-0` and `tweet-summary-1`
   [LoRA adapters](https://docs.vllm.ai/en/latest/features/lora.html) of the sample model server.
   ```bash
   kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/pkg/manifests/inferencemodel.yaml
   ```

1. **Update Envoy Gateway Config to enable Patch Policy**

   Our custom LLM Gateway ext-proc is patched into the existing envoy gateway via `EnvoyPatchPolicy`. To enable this feature, we must extend the Envoy Gateway config map. To do this, simply run:
   ```bash
   kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/pkg/manifests/gateway/enable_patch_policy.yaml
   kubectl rollout restart deployment envoy-gateway -n envoy-gateway-system
   ```
   Additionally, if you would like to enable the admin interface, you can uncomment the admin lines and run this again.

1. **Deploy Gateway**

   ```bash
   kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/pkg/manifests/gateway/gateway.yaml
   ```
   > **_NOTE:_** This file couples together the gateway infra and the HTTPRoute infra for a convenient, quick startup. Creating additional/different InferencePools on the same gateway will require an additional set of: `Backend`, `HTTPRoute`, the resources included in the `./manifests/gateway/ext-proc.yaml` file, and an additional `./manifests/gateway/patch_policy.yaml` file. ***Should you choose to experiment, familiarity with xDS and Envoy are very useful.***

   Confirm that the Gateway was assigned an IP address and reports a `Programmed=True` status:
   ```bash
   $ kubectl get gateway inference-gateway
   NAME                CLASS               ADDRESS         PROGRAMMED   AGE
   inference-gateway   inference-gateway   <MY_ADDRESS>    True         22s
   ```

1. **Deploy the Inference Extension and InferencePool**

   ```bash
   kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/pkg/manifests/ext_proc.yaml
   ```

1. **Deploy Envoy Gateway Custom Policies**

   ```bash
   kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/pkg/manifests/gateway/extension_policy.yaml
   kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/pkg/manifests/gateway/patch_policy.yaml
   ```
   > **_NOTE:_** This is also per InferencePool, and will need to be configured to support the new pool should you wish to experiment further.

1. **OPTIONALLY**: Apply Traffic Policy

   For high-traffic benchmarking you can apply this manifest to avoid any defaults that can cause timeouts/errors.

   ```bash
   kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/pkg/manifests/gateway/traffic_policy.yaml
   ```

1. **Try it out**

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