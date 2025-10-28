=== "GPU-Based Model Server"

    For this setup, you will need 3 GPUs to run the sample model server. Adjust the number of replicas as needed.
    Create a Hugging Face secret to download the model [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).
    Ensure that the token grants access to this model.

    Deploy a sample vLLM deployment with the proper protocol to work with the LLM Instance Gateway.
