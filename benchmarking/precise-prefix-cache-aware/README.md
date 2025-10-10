# Precise Prefix Cache Aware Benchmarking Helm Chart

This Helm chart deploys the `inference-perf` benchmarking tool with two distinct configurations: a high-cache scenario and a low-cache scenario. This chart specifically utilizes the **shared prefix dataset** for benchmarking. This guide will walk you through deploying both.

## Prerequisites

Before you begin, ensure you have the following:

*   **Helm 3+**: [Installation Guide](https://helm.sh/docs/intro/install/)
*   **Kubernetes Cluster**: Access to a Kubernetes cluster
*   **Gateway Deployed**: Your inference server/gateway must be deployed and accessible within the cluster.


**Hugging Face Token Secret**

The benchmark requires a Hugging Face token to pull models. Create a Kubernetes Secret named `hf-token` (or a custom name you provide) in your target namespace, containing your Hugging Face token.

    To create this secret:
    ```bash
    export _HF_TOKEN='<YOUR_HF_TOKEN>'
    kubectl create secret generic hf-token --from-literal=token=$_HF_TOKEN
    ```

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
helm install my-release . -f high-cache-values.yaml --set config.data.shared_prefix.num_groups=512
```

## Deployment

This chart supports two main configurations, defined in `high-cache-values.yaml` and `low-cache-values.yaml`.

### 1. Deploying the High-Cache Configuration

This configuration is optimized for scenarios where a high cache hit rate is expected. It uses the `high-cache-values.yaml` file.

```bash
export IP='<YOUR_IP>'
export PORT='<YOUR_PORT>'
helm install high-cache . -f high-cache-values.yaml \
  --set hfTokenSecret.name=hf-token \
  --set hfTokenSecret.key=token \
  --set "config.server.base_url=http://${IP}:${PORT}"
```

**Parameters to customize:**

*   `high-cache`: A unique name for this deployment.
*   `hfTokenSecret.name`: The name of your Kubernetes Secret containing the Hugging Face token (default: `hf-token`).
*   `hfTokenSecret.key`: The key in your Kubernetes Secret pointing to the Hugging Face token (default: `token`).
*   `config.server.base_url`: The base URL (IP and port) of your inference server for the high-cache scenario.

### 2. Deploying the Low-Cache Configuration

This configuration is designed for scenarios with a lower cache hit rate. It uses the `low-cache-values.yaml` file.

```bash
export IP='<YOUR_IP>'
export PORT='<YOUR_PORT>'
helm install low-cache . -f low-cache-values.yaml \
  -f high-cache-values.yaml \
  --set hfTokenSecret.name=hf-token \
  --set hfTokenSecret.key=token \
  --set "config.server.base_url=http://${IP}:${PORT}"
```

**Parameters to customize:**

*   `low-cache`: A unique name for this deployment.
*   `hfTokenSecret.name`: The name of your Kubernetes Secret containing the Hugging Face token (default: `hf-token`).
*   `hfTokenSecret.key`: The key in your Kubernetes Secret pointing to the Hugging Face token (default: `token`).
*   `config.server.base_url`: The base URL (IP and port) of your inference server for the high-cache scenario.

## Uninstalling the Charts

To uninstall the deployed charts:

```bash
helm uninstall my-high-cache-release
helm uninstall my-low-cache-release
```
