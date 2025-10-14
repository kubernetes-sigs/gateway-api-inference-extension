   Three options are supported for running the model server:

   1. GPU-based model server.
      Requirements: a Hugging Face access token that grants access to the model [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

   1. CPU-based model server (not using GPUs).
      The sample uses the model [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).

   1. [vLLM Simulator](https://github.com/llm-d/llm-d-inference-sim/tree/main) model server (not using GPUs).
      The sample is configured to simulate the [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) model.

   Choose one of these options and follow the steps below. Please do not deploy more than one, as the deployments have the same name and will override each other.

=== "GPU-Based Model Server"

    For this setup, you will need 3 GPUs to run the sample model server. Adjust the number of replicas in `./config/manifests/vllm/gpu-deployment.yaml` as needed.
    Create a Hugging Face secret to download the model [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct). Ensure that the token grants access to this model.

    Deploy a sample vLLM deployment with the proper protocol to work with the LLM Instance Gateway.
