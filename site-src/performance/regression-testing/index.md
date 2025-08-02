# Regression Testing

Regression testing verifies that recent code changes have not adversely affected the performance or stability of the Inference Gateway.

This guide explains how to run regression tests against the Gateway API inference extension using the [Latency Profile Generator (LPG)](https://github.com/AI-Hypercomputer/inference-benchmark/) to simulate traffic and collect performance metrics.

## Prerequisites

Refer to the [benchmark guide](/site-src/performance/benchmark/index.md) for common setup steps, including deployment of the inference extension, model server setup, scaling the vLLM deployment, and obtaining the Gateway IP.

## Create the LPG Docker Image

Follow the detailed instructions [here](https://github.com/AI-Hypercomputer/inference-benchmark/blob/1c92df607751a7ddb04e2152ed7f6aaf85bd9ca7/README.md) to build the LPG Docker image:

* Create an artifact repository:

  ```bash
  gcloud artifacts repositories create ai-benchmark --location=us-central1 --repository-format=docker
  ```

* Prepare datasets for [Infinity-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and [billsum]((https://huggingface.co/datasets/FiscalNote/billsum)):

  ```bash
  pip install datasets transformers numpy pandas tqdm matplotlib
  python datasets/import_dataset.py --hf_token YOUR_TOKEN
  ```

* Build the benchmark Docker image:

  ```bash
  docker build -t inference-benchmark .
  ```

* Push the Docker image to your artifact registry:

  ```bash
  docker tag inference-benchmark us-central1-docker.pkg.dev/{project-name}/ai-benchmark/inference-benchmark
  docker push us-central1-docker.pkg.dev/{project-name}/ai-benchmark/inference-benchmark
  ```

## Conduct Regression Tests

Run benchmarks using the configurations below, which are optimized for NVIDIA H100 GPUs (80 GB). Adjust configurations for other hardware as necessary.

### Test Case 1: Single Workload

- **Dataset:** `billsum_conversations.json` (created from [HuggingFace billsum dataset](https://huggingface.co/datasets/FiscalNote/billsum)).  
  *This dataset features long prompts, making it prefill-heavy and ideal for testing scenarios that emphasize initial token generation.*
- **Model:** [Llama 3 (8B)](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) (*critical*)
- **Replicas:** 10 (vLLM)
- **Request Rates:** 300–350 QPS (increments of 10)

Refer to example manifest:  
`./config/manifests/regression-testing/single-workload-regression.yaml`

### Test Case 2: Multi-LoRA

- **Dataset:** `Infinity-Instruct_conversations.json` (created from [HuggingFace Infinity-Instruct dataset](https://huggingface.co/datasets/BAAI/Infinity-Instruct)).  
  *This dataset has long outputs, making it decode-heavy and useful for testing scenarios focusing on sustained token generation.*
- **Model:** [Llama 3 (8B)](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- **LoRA Adapters:** 15 adapters (`nvidia/llama-3.1-nemoguard-8b-topic-control`, rank 8, critical)
- **Traffic Distribution:**  
  - 60 % on first 5 adapters (12 % each)  
  - 30 % on next 5 adapters (6 % each)  
  - 10 % on last 5 adapters (2 % each)  
- **Max LoRA:** 3
- **Replicas:** 10 (vLLM)
- **Request Rates:** 20–200 QPS (increments of 20)

Optionally, you can also run benchmarks against the `ShareGPT` dataset for additional coverage.

Update deployments for multi-LoRA support:  
- vLLM Deployment: `./config/manifests/regression-testing/vllm/multi-lora-deployment.yaml`  
- InferenceObjective.yaml: `./config/manifests/inferenceobjective.yaml.yaml`

Refer to example manifest:  
`./config/manifests/regression-testing/multi-lora-regression.yaml`

### Execute Benchmarks

Benchmark in two phases: before and after applying your changes:

- **Before changes:**

  ```bash
  benchmark_id='regression-before' ./tools/benchmark/download-benchmark-results.bash
  ```

- **After changes:**

  ```bash
  benchmark_id='regression-after' ./tools/benchmark/download-benchmark-results.bash
  ```

## Analyze Benchmark Results

Use the provided Jupyter notebook (`./tools/benchmark/benchmark.ipynb`) to analyze results:

- Update benchmark IDs to `regression-before` and `regression-after`.
- Compare latency and throughput metrics, performing regression analysis.
- Check R² values specifically:
  - **Prompts Attempted/Succeeded:** Expect R² ≈ 1
  - **Output Tokens per Minute, P90 per Output Token Latency, P90 Latency:** Expect R² close to 1 (allow minor variance).

Identify significant deviations, investigate causes, and confirm performance meets expected standards.

# Nightly Benchmarking

To catch regressions early, we run a fully automated benchmark suite every night against the **latest `main` image** of the Gateway API. This pipeline uses LPG and the same manifests as above, but against two standard datasets:

1. **Prefill-Heavy** (`billsum_conversations.json`)  
   Emphasizes TTFT performance.
2. **Decode-Heavy** (`Infinity-Instruct_conversations.json`)  
   Stresses sustained TPOT behavior.
3. **Multi-LoRA**  (`billsum_conversations.json`)  
   Uses 15 adapters with the traffic split defined above to capture complex adapter-loading and lora affinity scenarios.

**How it works**:

- The benchmarking runs are triggered every 6 hours.
- It provisions a GKE cluster with several NVIDIA H100 (80 GB) GPUs, deploys N vLLM server replicas along with the latest Gateway API extension and monitoring manifests, then launches the benchmarking script.
- In the step above we deploy the latest Endpoint Picker from the `main` branch’s latest Docker image:
  ```
  us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/epp:main
  ```
- It sequentially launches three benchmark runs (as described above) using the existing regression manifests.
- Results are uploaded to a central GCS bucket.
- A Looker Studio dashboard automatically refreshes to display key metrics:  
  https://lookerstudio.google.com/u/0/reporting/c7ceeda6-6d5e-4688-bcad-acd076acfba6/page/6S4MF
- After the benchmarking runs are complete it tears down the cluster.

**Alerting**:

- If any regression is detected oncall is setup (internally in GKE) for further investigation.