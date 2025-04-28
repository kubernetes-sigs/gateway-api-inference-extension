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

Run benchmarks using the configurations below, which are optimized for NVIDIA H100 GPUs (80 GB). Adjust configurations for other hardware as necessary.

### Test Case 1: Single Workload

- **Dataset:** `billsum_conversations.json` (created from [HuggingFace billsum dataset](https://huggingface.co/datasets/FiscalNote/billsum)).
    * This dataset features long prompts, making it prefill-heavy and ideal for testing scenarios that emphasize initial token generation.
- **Model:** [Llama 3 (8B)](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) (*critical*)
- **Replicas:** 10 (vLLM)
- **Request Rates:** 300–350 (increments of 10)

Refer to example manifest:
`./config/manifests/regression-testing/single-workload-regression.yaml`

### Test Case 2: Multi-LoRA

- **Dataset:** `Infinity-Instruct_conversations.json` (created from [HuggingFace Infinity-Instruct dataset](https://huggingface.co/datasets/BAAI/Infinity-Instruct)).
    * This dataset has long outputs, making it decode-heavy and useful for testing scenarios focusing on sustained token generation.
- **Model:** [Llama 3 (8B)](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- **LoRA Adapters:** 15 adapters (`nvidia/llama-3.1-nemoguard-8b-topic-control`, rank 8, critical)
- **Hardware:** NVIDIA H100 GPUs (80 GB)
- **Traffic Distribution:** 60% (first 5 adapters, each 12%), 30% (next 5, each 6%), 10% (last 5, each 2%) simulating prod/dev/test tiers
- **Max LoRA:** 3
- **Replicas:** 10 (vLLM)
- **Request Rates:** 20–200 (increments of 20)

Optionally, you can also run benchmarks using the `ShareGPT` dataset for additional coverage.

Update deployments for multi-LoRA support:
- vLLM Deployment: `./config/manifests/regression-testing/vllm/multi-lora-deployment.yaml`
- InferenceModel: `./config/manifests/inferencemodel.yaml`

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
  - **Prompts Attempted/Succeeded:** Expect R² ≈ 1
  - **Output Tokens per Minute, P90 per Output Token Latency, P90 Latency:** Expect R² close to 1 (allow minor variance).

Identify significant deviations, investigate causes, and confirm performance meets expected standards.