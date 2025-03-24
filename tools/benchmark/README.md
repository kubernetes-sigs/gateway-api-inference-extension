This folder contains resources to run performance benchmarks. Pls follow the benchmark guide here https://gateway-api-inference-extension.sigs.k8s.io/performance/benchmark.

## Features

1. **Config driven benchmarks**. Use the `./proto/benchmark.proto` API to write benchmark configurations, without the need to craft complex yamls.
2. **Reproducibility**. The tool will snapshot all the manifests needed for the benchmark run and mark them immutable (unless the user explicitly overrides it). 
3. **Benchmark inheritance**. Extend an existing benchmark configuration by overriding a subset of parameters, instead of re-writing everything from scratch.
4. **Benchmark orchestration**. The tool automatically deploys benchmark environment into a cluster, and waits to collects results, and then tears down the environment. The tool deploys the benchmark resources in new namespaces so each benchmark runs independently.
5. **Auto generated request rate**. The tool can automatically generate request rates for known models and accelerators to cover a wide range of model server load from low latency to fully saturated throughput.
6. **Visulization tools**. The results can be analyzed with a jupyter notebook.
7. **Model server metrics**. The tool uses the latency profile generator benchmark tool to scrape metrics from Google Cloud monitoring. It also provides a link to a Google Cloud monitoring dashboard for detailed analysis.

### Future Improvements

1. The benchmark config and results are stored in protobuf format. The results can be persisted in a database such as Google Cloud Spanner to allow complex query and dashboarding use cases.
2. Support running benchmarks in parallel with user configured parallelism.

## Prerequisite

1. [Install helm](https://helm.sh/docs/intro/quickstart/#install-helm)
2. Install InferenceModel and InferencePool [CRDs](https://gateway-api-inference-extension.sigs.k8s.io/guides/#install-the-inference-extension-crds) 
3. [Enable Envoy patch policy](https://gateway-api-inference-extension.sigs.k8s.io/guides/#update-envoy-gateway-config-to-enable-patch-policy).
4. Install [RBACs](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/12bcc9a85dad828b146758ad34a69053dca44fa9/config/manifests/inferencepool.yaml#L78) for EPP to read pods.
5. Create a secret in the default namespace containing the HuggingFace token. 
   
   ```bash
   kubectl create secret generic hf-token --from-literal=token=$HF_TOKEN # Your Hugging Face Token with access to Llama2
   ```

6. [Optional, GCP only] Create a `gmp-test-sa` service account with `monitoring.Viewer` role to read additional model server metrics from cloud monitoring. 
   
   ```bash
    gcloud iam service-accounts create gmp-test-sa \
    &&
    gcloud projects add-iam-policy-binding ${BENCHMARK_PROJECT} \
    --member=serviceAccount:gmp-test-sa@${BENCHMARK_PROJECT}.iam.gserviceaccount.com \
    --role=roles/monitoring.viewer
   ```

## Get started

Run all existing benchmarks:

```bash
# Run all benchmarks in the ./catalog/benchmark folder
./scripts/run_all_benchmarks.bash
```

View the benchmark results:

* To view raw results, watch for a new results folder to be created `./output/{run_id}/`. 
* To visualize the results, use the jupyter notebook.

## Common usage

### Run all benchmarks in a particular benchmark config file and upload results to GCS

```bash
gcs_bucket='my-bucket' benchmarks=benchmarks ./scripts/run_benchmarks_file.bash
```

### Generate benchmark manifests only

```bash
# All available environment variables.
benchmarks=benchmarks ./scripts/generate_manifests.bash
```

### Run particular benchmarks in a benchmark config file, by matching a benchmark name refex

```bash
# Run all benchmarks with Nvidia H100
gcs_bucket='my-bucket' benchmarks=benchmarks benchmark_name_regex='.*h100.*'  ./scripts/run_benchmarks_file.bash
```

### Resume a benchmark run from an existing run_id

You may resume benchmarks from previously generated manifests. The tool will skip benchmarks which have the `results` folder, and continue those without results.

```bash
run_id='existing-run-id' benchmarks=benchmarks ./scripts/run_benchmarks_file.bash
```

### Keep the benchmark environment after benchmark is complete (for debugging)

```bash
# All available environment variables.
skip_tear_down='true' benchmarks=benchmarks  ./scripts/run_benchmarks_file.bash
```

## Command references

```bash
# All available environment variables
regex='my-benchmark-file-name-regex' dry_run='false'  gcs_bucket='my-bucket' skip_tear_down='false' benchmark_name_regex='my-benchmark-name-regex' ./scripts/run_all_benchmarks.bash
```

```bash
# All available environment variables.
run_id='existing-run-id' dry_run='false'  gcs_bucket='my-bucket' skip_tear_down='false' benchmarks=benchmarks benchmark_name_regex='my-benchmark-name-regex'  ./scripts/run_benchmarks_file.bash
```

```bash
# All available environment variables.
run_id='existing-run-id' benchmarks=benchmarks ./scripts/generate_manifests.bash
```

## How does it work?

The tool will automate the following steps:

1. Read the benchmark config file in `./catalog/{benchmarks_config_file}`. The file contains a list of benchmarks. The config API is defined in `./proto/benchmark.proto`.
2. Generates a new run_id and namespace `{benchmark_name}-{run_id}` to run the benchmarks. If the `run_id` environment variable is provided, it will reuse it instead of creating a new one. This is useful when resuming a previous benchmark run, or run multiple sets of benchmarks in parallel (e.g., run benchmarks on different accelerator types in parallel using the same run_id).
3. Based on the config, generates manifests in `./output/{run_id}/{benchmark_name}-{run_id}/manifests`
4. Applies the manifests to the cluster, and wait for resources to be ready.
5. Once the benchmark finishes, downloads benchmark results to `./output/{run_id}/{benchmark}-{run_id}/results`
6. [Optional] If a GCS bucket is specified, uploads the output folder to a GCS bucket.

## Create a new benchmark

You can either add new benchmarks to an existing benchmark config file, or create new benchmark config files. Each benchmark config file contains a list of benchmarks.

An example benchmark with all available parameters is as follows:

```
benchmarks {
    name: "base-benchmark"
    config {
        model_server {
            image: "vllm/vllm-openai@sha256:8672d9356d4f4474695fd69ef56531d9e482517da3b31feb9c975689332a4fb0"
            accelerator: "nvidia-h100-80gb"
            replicas: 1
            vllm {
                tensor_parallelism: "1"
                model: "meta-llama/Llama-2-7b-hf"
            }
        }
        load_balancer {
            gateway {
                envoy {
                    epp {
                        image: "us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/epp:v0.1.0"
                    }
                }
            }
        }
        benchmark_tool {
            image: "us-docker.pkg.dev/gke-inference-gateway-dev/benchmark/benchmark-tool@sha256:1fe4991ec1e9379b261a62631e1321b8ea15772a6d9a74357932771cea7b0500"
            lpg {
                dataset: "sharegpt_v3_unfiltered_cleaned_split"
                models: "meta-llama/Llama-2-7b-hf"
                ip: "to-be-populated-automatically"
                port: "8081"
                benchmark_time_seconds: "60"
                output_length: "1024"
            }
        }
    }
}
```

### Create a benchmark from a base benchmark

It's recommended to create a benchmark from an existing benchmark by overriding a few parameters. This inheritance feature is powerful in creating a large number of benchmarks conveniently. Below is an example that overrides the replica count of a base benchmark:

```
benchmarks {
    name: "new-benchmark"
    base_benchmark_name: "base-benchmark"
    config {
        model_server {
            replicas: 2
        }
    }
}
```

## Environment configurations

The tool has default configurations (such as the cluster name) in `./scripts/env.sh`. You can tweak those for your own needs.

## The benchmark.proto

The `./proto/benchmark.proto` is the core of this tool, it drives the generation of the benchmark manifests, as well as the query and dashboarding of the results.

Why do we need it?

* An API to clearly capture the intent, instead of making various assumptions.
* It lets the user to focus only on the core parameters of the benchmark itself, rather than the toil of configuring the environment and crafting the manifests.
* It is the single source of truth that drives the entre lifecycle of the benchmark, including post analysis.

## Contribute

Refer to the [dev guide](./dev.md).