# Benchmark

This user guide shows how to run benchmarks against a vLLM model server deployment by using both Gateway API
Inference Extension, and a Kubernetes service as the load balancing strategy. The benchmark uses the
[Latency Profile Generator](https://github.com/AI-Hypercomputer/inference-benchmark) (LPG) tool to generate
load and collect results.

## Prerequisites

### Deploy the inference extension and sample model server

Follow the [getting started guide](https://gateway-api-inference-extension.sigs.k8s.io/guides/#getting-started-with-gateway-api-inference-extension)
to deploy the vLLM model server, CRDs, etc.

__Note:__ Only the GPU-based model server deployment option is supported for benchmark testing.

### [Optional] Scale the sample vLLM deployment

You are more likely to see the benefits of the inference extension when there are a decent number of replicas to make the optimal routing decision.

```bash
kubectl scale deployment vllm-llama3-8b-instruct --replicas=8
```

### Expose the model server via a k8s service

To establish a baseline, expose the vLLM deployment as a k8s service:

```bash
kubectl expose deployment vllm-llama3-8b-instruct --port=80 --target-port=8000 --type=LoadBalancer
```

## Run benchmark

The inference perf tool works by sending traffic to the specified target IP and port, and collecting the results.
Follow the steps below to run a single benchmark. Multiple benchmarking instances can be deployed to run benchmarks in
parallel against different targets.

#### Parameters to customize:

For more parameter customizations, refer to inference-perf [guides](https://github.com/kubernetes-sigs/inference-perf/blob/main/docs/config.md)

*   `benchmark`: A unique name for this deployment.
*   `hfToken`: Your hugging face token.
*   `config.server.base_url`: The base URL (IP and port) of your inference server.

### Storage Parameters

    Note: Currently inference-perf outputs benchmark results to standard output only, and results will be deleted once pod is finished running the job.


#### 1. Local Storage (Default)

By default, reports are saved locally but **lost when the Pod terminates**.
```yaml
storage:
  local_storage:
    path: "reports-{timestamp}"       # Local directory path
    report_file_prefix: null          # Optional filename prefix
```

#### 2. Google Cloud Storage (GCS)

Use the `google_cloud_storage` block to save reports to a GCS bucket.

```yaml
storage:
  google_cloud_storage:               # Optional GCS configuration
    bucket_name: "your-bucket-name"   # Required GCS bucket
    path: "reports-{timestamp}"       # Optional path prefix
    report_file_prefix: null          # Optional filename prefix
```

###### 🚨 GCS Permissions Checklist (Required for Write Access)

1. **IAM Role (Service Account):** Bound to the target bucket.

   * **Minimum:** **Storage Object Creator** (`roles/storage.objectCreator`)

   * **Full:** **Storage Object Admin** (`roles/storage.objectAdmin`)

2. **Node Access Scope (GKE Node Pool):** Set during node pool creation.

   * **Required Scope:** **`devstorage.read_write`** or **`cloud-platform`**

#### 3. Simple Storage Service (S3)

Use the `simple_storage_service` block for S3-compatible storage. Requires appropriate AWS credentials configured in the runtime environment.

```yaml
storage:
  simple_storage_service:
    bucket_name: "your-bucket-name"   # Required S3 bucket
    path: "reports-{timestamp}"       # Optional path prefix
    report_file_prefix: null          # Optional filename prefix
```
### Steps to Deploy

1. Check out the repo.

    ```bash
    git clone https://github.com/kubernetes-sigs/gateway-api-inference-extension
    cd gateway-api-inference-extension/benchmarking
    ```

1. Get the target IP. The examples below shows how to get the IP of a gateway or a k8s service.

    ```bash
    # Get gateway IP
    GW_IP=$(kubectl get gateway/inference-gateway -o jsonpath='{.status.addresses[0].value}')
    # Get LoadBalancer k8s service IP
    SVC_IP=$(kubectl get service/vllm-llama3-8b-instruct -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

    echo $GW_IP
    echo $SVC_IP
    ```

1. Deploy Benchmark Tool. 

    ```bash
    export PORT='<YOUR_PORT>'
    export HF_TOKEN='<YOUR_HUGGINGFACE_TOKEN>'
    helm install igw-benchmark inference-perf/ -f benchmark-values.yaml \
    --set hfToken=${HF_TOKEN} \
    --set "config.server.base_url=http://${GW_IP}:${PORT}"

    export PORT='<YOUR_PORT>'
    export HF_TOKEN='<YOUR_HUGGINGFACE_TOKEN>'
    helm install k8s-benchmark inference-perf/ -f benchmark-values.yaml \
    --set hfToken=${HF_TOKEN} \
    --set "config.server.base_url=http://${SVC_IP}:${PORT}"
    ```

1. Wait for benchmark to finish and download the results. Follow inference-perf [guides](https://github.com/kubernetes-sigs/inference-perf on how to access logs. At this moment logs are deleted from the pod if using local storage.

    #### GCS Benchmarking Script

     If storing results in GCS, you can use the `download-gcs-results.bash` script.

    Use the `benchmark_id` environment variable to specify what this
   benchmark is for. For instance, `inference-extension` or `k8s-svc`.

    ```bash
    benchmark_id='k8s-svc' ./download-gcs-results.bash <GCS_BUCKET> <GCS_FOLDER>

    benchmark_id='inference-extension' ./download-gcs-results.bash <GCS_BUCKET> <GCS_FOLDER>
    ```

    After the script finishes, you should see benchmark results under `./benchmarking/output/default-run/k8s-svc/results/json/<GCS_FOLDER>`.

1. Uninstall the chart to tear down resources

    ```bash
    helm uninstall igw-benchmark k8s-benchmark
    ```


### Tips

* When using a `benchmark_id` other than `k8s-svc` or `inference-extension`, the labels in `./tools/benchmark/benchmark.ipynb` must be
  updated accordingly to analyze the results.
* You can specify `run_id="runX"` environment variable when running the `./download-benchmark-results.bash` script.
This is useful when you run benchmarks multiple times to get a more statistically meaningful results and group the results accordingly.
* Update the `stages` to request rates that best suit your benchmark environment.

### Advanced Benchmark Configurations

Refer to the inference-perf [guides](https://github.com/kubernetes-sigs/inference-perf/blob/main/docs/config.md) for a
detailed list of configuration knobs.

## Analyze the results

This guide shows how to run the jupyter notebook using vscode after completing k8s service and inference extension benchmarks.

1. Create a python virtual environment.

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

1. Install the dependencies.

    ```bash
    pip install -r ./tools/benchmark/requirements.txt
    ```

1. Open the notebook `./benchmarking/benchmark.ipynb`, and run each cell. In the last cell update the benchmark ids with`inference-extension` and `k8s-svc`. At the end you should
    see a bar chart like below where **"ie"** represents inference extension. This chart is generated using this benchmarking tool with 6 vLLM (v1) model servers (H100 80 GB), [llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main) and the [ShareGPT dataset](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json).
    
    ![alt text](example-bar-chart.png)