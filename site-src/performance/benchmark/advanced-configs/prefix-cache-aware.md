# Prefix Cache Aware Benchmarking
This guide shows how to deploy a prefix-cache-aware benchmarking config using inference-perf.

## Prerequisites

Before you begin, ensure you have the following:

*   **Helm 3+**: [Installation Guide](https://helm.sh/docs/intro/install/)
*   **Kubernetes Cluster**: Access to a Kubernetes cluster
*   **Hugging Face Token Secret**: A Hugging Face token to pull models.
*   **Gateway Deployed**: Your inference server/gateway must be deployed and accessible within the cluster.

Follow [benchmarking guide](https://gateway-api-inference-extension.sigs.k8s.io/performance/benchmark/#benchmark) for more information on how to set up gateway and how to validate benchmark results.

## Shared Prefix Dataset Configuration

The chart uses the `shared_prefix` dataset type, which is designed to test caching efficiency. These parameters are located under config.data.shared_prefix:

*   `num_groups`: The number of shared prefix groups.
*   `num_prompts_per_group`: The number of prompts within each shared prefix group.
*   `system_prompt_len`: The length of the system prompt.
*   `question_len`: The length of the question part of the prompt.
*   `output_len`: The desired length of the model's output.

The default values for the dataset are defined in the chart, but you can override them using `--set config.data.shared_prefix.<parameter>` flags. 

Example:

```bash
helm install my-release ../inference-perf -f high-cache-values.yaml --set config.data.shared_prefix.num_groups=512
```

## Deployment

This chart supports two main configurations, defined in `high-cache-values.yaml` and `low-cache-values.yaml`.

### 1. Check out the repo.

```bash
git clone https://github.com/kubernetes-sigs/gateway-api-inference-extension
cd gateway-api-inference-extension/benchmarking/prefix-cache-aware
```

### 2. Get the target IP. 

  The examples below shows how to get the IP of a gateway or a k8s service.

  ```bash
  # Get gateway IP
  GW_IP=$(kubectl get gateway/inference-gateway -o jsonpath='{.status.addresses[0].value}')
  # Get LoadBalancer k8s service IP
  SVC_IP=$(kubectl get service/vllm-llama3-8b-instruct -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

  echo $GW_IP
  echo $SVC_IP
  ```

### 3. Deploying the High-Cache Configuration

This configuration is optimized for scenarios where a high cache hit rate is expected. It uses the `high-cache-values.yaml` file.

```bash
export IP='<YOUR_IP>'
export PORT='<YOUR_PORT>'

# HUGGINGFACE PARAMETERS
# Option A: Pass Token Directly
export HF_TOKEN='<YOUR_HUGGINGFACE_TOKEN>'
# Option B: Use Existing Kubernetes Secret
export HF_SECRET_NAME='<YOUR_SECRET_NAME>'
export HF_SECRET_KEY='<YOUR_SECRET_KEY>'

helm install high-cache ../inference-perf -f high-cache-values.yaml \
  --set "config.server.base_url=http://${IP}:${PORT}"
  # ------------------------------------------------
  # HUGGINGFACE OPTION A
  --set token.hfToken=${HF_TOKEN} \ 
  # ------------------------------------------------
  # HUGGINGFACE OPTION B
  # --set token.hfSecret.name=${HF_SECRET_NAME} \
  # --set token.hfSecret.key=${HF_SECRET_KEY} \
  # ------------------------------------------------
```

**Parameters to customize:**

*   `high-cache`: A unique name for this deployment.
*   `token.hfToken`: Your hugging face token. Inference Perf chart will create a new kubernetes secret containing this token.
*   `hfSecret.name`: The name of your Kubernetes Secret containing the Hugging Face token (default: `hf-token`).
*   `hfSecret.key`: The key in your Kubernetes Secret pointing to the Hugging Face token (default: `token`).
*   `config.server.base_url`: The base URL (IP and port) of your inference server for the high-cache scenario.

### 4. Deploying the Low-Cache Configuration

This configuration is designed for scenarios with a lower cache hit rate. It uses the `low-cache-values.yaml` file.

```bash
cd gateway-api-inference-extension/benchmarking/prefix-cache-aware
export IP='<YOUR_IP>'
export PORT='<YOUR_PORT>'

# HUGGINGFACE PARAMETERS
# Option A: Pass Token Directly
export HF_TOKEN='<YOUR_HUGGINGFACE_TOKEN>'
# Option B: Use Existing Kubernetes Secret
export HF_SECRET_NAME='<YOUR_SECRET_NAME>'
export HF_SECRET_KEY='<YOUR_SECRET_KEY>'

helm install low-cache ../inference-perf -f low-cache-values.yaml \
  --set "config.server.base_url=http://${IP}:${PORT}"
  # ------------------------------------------------
  # HUGGINGFACE OPTION A
  --set token.hfToken=${HF_TOKEN} \ 
  # ------------------------------------------------
  # HUGGINGFACE OPTION B
  # --set token.hfSecret.name=${HF_SECRET_NAME} \
  # --set token.hfSecret.key=${HF_SECRET_KEY} \
  # ------------------------------------------------
```

**Parameters to customize:**

*   `low-cache`: A unique name for this deployment.
*   `token.hfToken`: Your hugging face token. Inference Perf chart will create a new kubernetes secret containing this token.
*   `hfSecret.name`: The name of your Kubernetes Secret containing the Hugging Face token (default: `hf-token`).
*   `hfSecret.key`: The key in your Kubernetes Secret pointing to the Hugging Face token (default: `token`).
*   `config.server.base_url`: The base URL (IP and port) of your inference server for the high-cache scenario.

## Clean Up

To uninstall the deployed charts:

```bash
helm uninstall my-high-cache-release
helm uninstall my-low-cache-release
```

## Post Benchmark Analysis
Follow the benchmarking guide instructions to [compare benchmark results](https://gateway-api-inference-extension.sigs.k8s.io/performance/benchmark/#analyze-the-results).

## Running E2E Tests

The following E2E test runs on GitHub using GitHub Actions.

> If you have `MAINTAINER` access or above, you can trigger the workflow run from the GitHub Actions page, or by leaving a comment on the PR. 

> Please make sure there is no other GKE tests of the same type running at the same time, as they can interfere with each other.


| :--- | :--- | :--- |
| GKE Prefix Cache Aware Test	| https://github.com/gateway-api-inference-extension/.github/workflows/e2e-prefix-cache-aware-gke.yaml | /run-gke-prefix-cache |