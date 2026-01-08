# Prefill Heavy Workload Benchmarking
This guide shows how to deploy a prefill-heavy benchmarking config using inference-perf.

## Prerequisites

Before you begin, ensure you have the following:

*   **Helm 3+**: [Installation Guide](https://helm.sh/docs/intro/install/)
*   **Kubernetes Cluster**: Access to a Kubernetes cluster
*   **Hugging Face Token Secret**: A Hugging Face token to pull models.
*   **Gateway Deployed**: Your inference server/gateway must be deployed and accessible within the cluster.

Follow [benchmarking guide](https://gateway-api-inference-extension.sigs.k8s.io/performance/benchmark/#benchmark) for more information on how to set up gateway and how to validate benchmark results.

## Infinity Instruct Dataset Configuration

The chart uses the `infinity_instruct` [dataset type](https://huggingface.co/datasets/BAAI/Infinity-Instruct). 

>NOTE: Currently, we need to download and supply the dataset for inference-perf to ingest. Currently using helm, we can supply the dataset by uploading to a gcs or s3 bucket. Otherwise, you can follow inference perf guides to run locally with a local dataset file path.

## Deployment

### 1. Check out the repo.

```bash
git clone https://github.com/kubernetes-sigs/gateway-api-inference-extension
cd gateway-api-inference-extension/benchmarking/single-workload
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

### 3. Deploying the Prefill Heavy Configuration

This configuration is optimized for scenarios where a high cache hit rate is expected. It uses the `prefill-heavy-values.yaml` file.

=== "Google Cloud Storage (GCS)"
    Use the `gcsPath` field to provide your dataset stored on GCS. The dataset will be downloaded from the bucket and stored locally on the cluster at `/dataset/gcs-dataset.json`. 
    ```bash
    export IP='<YOUR_IP>'
    export PORT='<YOUR_PORT>'

    # HUGGINGFACE PARAMETERS
    # Option A: Pass Token Directly
    export HF_TOKEN='<YOUR_HUGGINGFACE_TOKEN>'
    # Option B: Use Existing Kubernetes Secret
    export HF_SECRET_NAME='<YOUR_SECRET_NAME>'
    export HF_SECRET_KEY='<YOUR_SECRET_KEY>'

    helm install prefill-heavy ../inference-perf -f prefill-heavy-values.yaml \
      --set "config.server.base_url=http://${IP}:${PORT}" \
      --set "config.data.path=/dataset/gcs-dataset.json" \
      --set "gcsPath=<PATH TO DATASET FILE ON GCS BUCKET>"
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
    
    *   `prefill-heavy`: A unique name for this deployment.
    *   `token.hfToken`: Your hugging face token. Inference Perf chart will create a new kubernetes secret containing this token.
    *   `hfSecret.name`: The name of your Kubernetes Secret containing the Hugging Face token (default: `hf-token`).
    *   `hfSecret.key`: The key in your Kubernetes Secret pointing to the Hugging Face token (default: `token`).
    *   `config.server.base_url`: The base URL (IP and port) of your inference server for the high-cache scenario.
    *   `gcsPath`: The path to the downloaded dataset file hosted on your gcs bucket. 

=== "Simple Storage Service (S3)"
    Use the `s3Path` field to provide your dataset stored on S3. The dataset will be downloaded from the bucket and stored locally on the cluster at `/dataset/s3-dataset.json`. 
    ```bash
    export IP='<YOUR_IP>'
    export PORT='<YOUR_PORT>'

    # HUGGINGFACE PARAMETERS
    # Option A: Pass Token Directly
    export HF_TOKEN='<YOUR_HUGGINGFACE_TOKEN>'
    # Option B: Use Existing Kubernetes Secret
    export HF_SECRET_NAME='<YOUR_SECRET_NAME>'
    export HF_SECRET_KEY='<YOUR_SECRET_KEY>'

    helm install prefill-heavy ../inference-perf -f prefill-heavy-values.yaml \
      --set "config.server.base_url=http://${IP}:${PORT}" \
      --set "config.data.path=/dataset/s3-dataset.json" \
      --set "s3Path=<PATH TO DATASET FILE ON S3 BUCKET>"
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
    
    *   `prefill-heavy`: A unique name for this deployment.
    *   `token.hfToken`: Your hugging face token. Inference Perf chart will create a new kubernetes secret containing this token.
    *   `hfSecret.name`: The name of your Kubernetes Secret containing the Hugging Face token (default: `hf-token`).
    *   `hfSecret.key`: The key in your Kubernetes Secret pointing to the Hugging Face token (default: `token`).
    *   `config.server.base_url`: The base URL (IP and port) of your inference server for the high-cache scenario.
    *   `s3Path`: The path to the downloaded dataset file hosted on your s3 bucket. 

## Clean Up

To uninstall the deployed charts:

```bash
helm uninstall prefill-heavy
```

## Post Benchmark Analysis
Follow the benchmarking guide instructions to [compare benchmark results](https://gateway-api-inference-extension.sigs.k8s.io/performance/benchmark/#analyze-the-results).


## Running E2E Tests

The following E2E test runs on GitHub using GitHub Actions.

> If you have `MAINTAINER` access or above, you can trigger the workflow run from the GitHub Actions page, or by leaving a comment on the PR. 

> Please make sure there is no other GKE tests of the same type running at the same time, as they can interfere with each other.


| Test name	| Link | PR comment trigger
| :--- | :--- | :--- |
| GKE Prefill Heavy Test | https://github.com/gateway-api-inference-extension/.github/workflows/e2e-prefill-heavy-gke.yaml | /run-gke-prefill-heavy |