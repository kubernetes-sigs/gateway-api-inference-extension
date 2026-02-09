# Deploy As A Standalone Request Scheduler
The endpoint picker (EPP) at its core is a smart request scheduler for LLM requests, it currently implements a number of LLM-specific load balancing optimizations including:

* Prefix-cache aware scheduling
* Load-aware scheduling

When using EPP with Gateway API, it works as an ext-proc to an envoy-based proxy fronting model servers running in a k8s cluster; 
examples of such proxies are cloud managed ones like GKE’s L7LB and open source counterparts like Istio and kGateway.
EPP as an ext-proc here offers several key advantages:

* It utilizes robust, pre-existing L7 proxies, including both managed and open-source options.
* Seamless integration with the Kubernetes networking ecosystem, the Gateway API, allows for:Transforming a Kubernetes gateway into an inference scheduler using familiar APIs. 
Leveraging Gateway API features like traffic splitting for gradual rollouts and HTTP rule matching. 
Access to provider-specific features.

These benefits are critical for online services, including MaaS (Model-as-a-Service), which require support for multi-tenancy, demand high availability, scalability, and streamlined operations.

However, for some batch inference, a tight integration with the Gateway API and requiring an external proxy to be deployed separately is in practice an operational overhead. 
Consider an offline RL post-training job, where the sampler, the inference service in the job, is a single tenant/workload with a lifecycle tied with the training job; 
this inference service is specific to the job, it is continuously updated during post-training, and so it is not one that would be serving any other traffic.
A simpler deployment mode would reduce the barrier to adopting the EPP for such single-tenant workloads.

## How
A proxy is deployed as a sidecar to the EPP. The proxy and EPP continue to communicate via ext-proc protocol over localhost.
For the endpoint discovery, you have two options:

* **With Inference APIs Support**: The EPP is configured using the Inference CRDs, the pool is expressed using an instance of the InferencePool API and the entire suite of inference APIs are supported, including the use of InferenceObjectives for defining priorities.
* **Without Inference APIs Support**: The EPP is configured using command line flags. This is the simplest method for standalone jobs which doesn't require installing the inference extension apis, which means no support for the features expressed using the inference APIs (such as InferenceObjectives).

## Example

### **Prerequisites**

--8<-- "site-src/_includes/prereqs.md"

### **Steps**

#### Deploy Sample Model Server

--8<-- "site-src/_includes/vllm-gpu.md"

    ```bash
    kubectl create secret generic hf-token --from-literal=token=$HF_TOKEN # Your Hugging Face Token with access to the set of Llama models
    kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/gpu-deployment.yaml
    ```

--8<-- "site-src/_includes/vllm-cpu.md"

    ```bash
    kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/cpu-deployment.yaml
    ```

--8<-- "site-src/_includes/model-server-sim.md"

    ```bash
    kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/sim-deployment.yaml
    ```

#### Deploy Endpoint Picker Extension with Envoy sidecar

Choose one of the following options to deploy an Endpoint Picker Extension with Envoy sidecar.

=== "With Inference APIs Support"

      Deploy an InferencePool named `vllm-llama3-8b-instruct` that selects from endpoints with label app: vllm-llama3-8b-instruct and
      listening on port 8000. The Helm install command automatically deploys an InferencePool instance, the epp along with provider specific resources.
        
      Set the chart version and then select a tab to follow the provider-specific instructions.
           ```bash
            # Install the Inference Extension CRDs
            kubectl apply -k https://github.com/kubernetes-sigs/gateway-api-inference-extension/config/crd
            
            export STANDALONE_CHART_VERSION=v0
            export PROVIDER=<YOUR_PROVIDER> #optional, can be gke as gke needed it specific epp monitoring resources.
            helm install vllm-llama3-8b-instruct-standalone \
            --dependency-update \
            --set inferencePool.modelServers.matchLabels.app=vllm-llama3-8b-instruct \
            --set provider.name=$PROVIDER \
            --version $STANDALONE_CHART_VERSION \
             oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/standalone
           ```

=== "Without Inference APIs Support"

     Deploy an Endpoint Picker Extension named `vllm-llama3-8b-instruct` that selects from endpoints with label `app=vllm-llama3-8b-instruct` and listening on port 8000. 
     The Helm install command automatically deploys the epp along with provider specific resources.
        
     Set the chart version and then select a tab to follow the provider-specific instructions.
           ```bash
            export STANDALONE_CHART_VERSION=v0
            export PROVIDER=<YOUR_PROVIDER> #optional, can be gke as gke needed it specific epp monitoring resources.
            helm install vllm-llama3-8b-instruct-standalone \
            --dependency-update \
            --set inferenceExtension.endpointsServer.endpointSelector="app=vllm-llama3-8b-instruct" \
            --set inferenceExtension.endpointsServer.createInferencePool=false
            --set provider.name=$PROVIDER \
            --version $STANDALONE_CHART_VERSION \
             oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/standalone
           ```

#### Try it out

Wait until the EPP deployment is ready.

Once the EPP pod is running,
Install the curl pod as follows:
```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: curl
  labels:
    app: curl
spec:
  containers:
  - name: curl
    image: curlimages/curl:7.83.1
    imagePullPolicy: IfNotPresent
    command:
      - tail
      - -f
      - /dev/null
  restartPolicy: Never
EOF
```
Send an inference request via
```bash
kubectl exec curl -- curl -i http://vllm-llama3-8b-instruct-epp:8081/v1/completions \
-H 'Content-Type: application/json' \
-d '{"model": "food-review-1","prompt": "Write as if you were a critic: San Francisco","max_tokens": 100,"temperature": 0}'
```

#### Cleanup
Run the following commands to remove all resources created by this guide.

The following instructions assume you would like to cleanup ALL resources that were created in this guide.
Please be careful not to delete resources you'd like to keep.

1. Uninstall the EPP, curl pod and model server resources:

   ```bash
   helm uninstall vllm-llama3-8b-instruct
   kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/inferenceobjective.yaml --ignore-not-found
   kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/cpu-deployment.yaml --ignore-not-found
   kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/gpu-deployment.yaml --ignore-not-found
   kubectl delete -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/sim-deployment.yaml --ignore-not-found
   kubectl delete -k https://github.com/kubernetes-sigs/gateway-api-inference-extension/config/crd --ignore-not-found
   kubectl delete secret hf-token --ignore-not-found
   kubectl delete pod curl --ignore-not-found
   ```
