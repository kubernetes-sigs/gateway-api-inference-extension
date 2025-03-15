# Benchmark

This user guide shows how to run benchmarks against a vLLM deployment, by using both the Gateway API
inference extension, and a Kubernetes service as the load balancing strategy. The
benchmark uses the [Latency Profile Generator](https://github.com/AI-Hypercomputer/inference-benchmark) (LPG)
tool to generate load and collect results.

## Prerequisites

### Deploy the inference extension and sample model server

Follow this user guide https://gateway-api-inference-extension.sigs.k8s.io/guides/ to deploy the
sample vLLM application, and the inference extension.

### [Optional] Scale the sample vLLM deployment

You will more likely to see the benefits of the inference extension when there are a decent number of replicas to make the optimal routing decision. 

```bash
kubectl scale deployment my-pool --replicas=8
```

### Expose the model server via a k8s service

As the baseline, let's also expose the vLLM deployment as a k8s service by simply applying the yaml:

```bash
kubectl apply -f .manifests/ModelServerService.yaml
```

## Run benchmark

### Run benchmark using the inference extension as the load balancing strategy

1. Get the gateway IP: 

    ```bash
    IP=$(kubectl get gateway/inference-gateway -o jsonpath='{.status.addresses[0].value}')
    echo "Update the <gateway-ip> in ./manifests/BenchmarkInferenceExtension.yaml to: $IP"
    ```

1. Then update the `<gateway-ip>` in `./manifests/BenchmarkInferenceExtension.yaml` to the IP
of the gateway. Feel free to adjust other parameters such as request_rates as well.

1. Start the benchmark tool. `kubectl apply -f ./manifests/BenchmarkInferenceExtension.yaml`

1. Wait for benchmark to finish and download the results. Use the `benchmark_id` environment variable
to specify what this benchmark is for. In this case, the result is for the `inference-extension`. You
can use any id you like.

    ```bash
    benchmark_id='inference-extension' ./download-benchmark-results.bash
    ```

1. After the script finishes, you should see benchmark results under `./output/default-run/inference-extension/results/json` folder.

### Run benchmark using k8s service as the load balancing strategy

1. Get the service IP: 

    ```bash
    IP=$(kubectl get service/my-pool-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    echo "Update the <svc-ip> in ./manifests/BenchmarkK8sService.yaml to: $IP"
    ```

2. Then update the `<svc-ip>` in `./manifests/BenchmarkK8sService.yaml` to the IP
of the service. Feel free to adjust other parameters such as **request_rates** as well.

1. Start the benchmark tool. `kubectl apply -f ./manifests/BenchmarkK8sService.yaml`

2. Wait for benchmark to finish and download the results.

    ```bash
    benchmark_id='k8s-svc' ./download-benchmark-results.bash
    ```

3. After the script finishes, you should see benchmark results under `./output/default-run/k8s-svc/results/json` folder.

### Tips

* You can specify `run_id="runX"` environment variable when running the `./download-benchmark-results.bash` script.
This is useful when you run benchmarks multiple times and group the results accordingly.

## Analyze the results

This guide shows how to run the jupyter notebook using vscode.

1. Create a python virtual environment.

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

1. Install the dependencies.

    ```bash
    pip install -r requirements.txt
    ```

1. Open the notebook `Inference_Extension_Benchmark.ipynb`, and run each cell. At the end you should
    see a bar chart like below:
    
    ![alt text](image.png)