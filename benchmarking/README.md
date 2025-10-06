# Benchmarking Helm Chart

This Helm chart deploys the `inference-perf` benchmarking tool. This guide will walk you through deploying a basic benchmarking job. By default, the `shareGPT` dataset is used for configuration.

## Prerequisites

Before you begin, ensure you have the following:

*   **Helm 3+**: [Installation Guide](https://helm.sh/docs/intro/install/)
*   **Kubernetes Cluster**: Access to a Kubernetes cluster
*   **Gateway Deployed**: Your inference server/gateway must be deployed and accessible within the cluster.


**Hugging Face Token Secret**

The benchmark requires a Hugging Face token to pull tokenizers. Create a Kubernetes Secret named `hf-token` (or a custom name you provide) in your target namespace, containing your Hugging Face token.

    To create this secret:
    ```bash
    export _HF_TOKEN='<YOUR_HF_TOKEN>'
    kubectl create secret generic hf-token --from-literal=token=$_HF_TOKEN
    ```

## Deployment

To deploy the benchmarking chart:

```bash
export IP='<YOUR_IP>'
export PORT='<YOUR_PORT>'
helm install benchmark . -f benchmark-values.yaml \
  --set hfTokenSecret.name=hf-token \
  --set hfTokenSecret.key=token \
  --set "config.server.base_url=http://${IP}:${PORT}"
```

**Parameters to customize:**

*   `benchmark`: A unique name for this deployment.
*   `hfTokenSecret.name`: The name of your Kubernetes Secret containing the Hugging Face token (default: `hf-token`).
*   `hfTokenSecret.key`: The key in your Kubernetes Secret pointing to the Hugging Face token (default: `token`).
*   `config.server.base_url`: The base URL (IP and port) of your inference server.

### Storage Parameters

The following is how to add storage to the config. 
By default we save to local storage however once the inference-perf job is completed the pod is deleted.

```yaml
storage:
  local_storage:
    path: "reports-{timestamp}"       # Local directory path
    report_file_prefix: null          # Optional filename prefix
  google_cloud_storage:               # Optional GCS configuration
    bucket_name: "your-bucket-name"   # Required GCS bucket
    path: "reports-{timestamp}"       # Optional path prefix
    report_file_prefix: null          # Optional filename prefix
  simple_storage_service:
    bucket_name: "your-bucket-name"   # Required S3 bucket
    path: "reports-{timestamp}"       # Optional path prefix
    report_file_prefix: null          # Optional filename prefix
```

## Uninstalling the Chart

To uninstall the deployed chart:

```bash
helm uninstall my-benchmark
```

