=== "GPU-Based vLLM deployment"

    For this setup, you will need 3 GPUs to run the sample model server (one GPU per replica). Adjust the number of replicas as needed based on your available GPU resources.
    
    Create a Hugging Face secret to download the model [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).
    You'll need a Hugging Face account and access token - see the [Hugging Face security tokens documentation](https://huggingface.co/docs/hub/security-tokens) for setup instructions.
    Ensure that the token grants access to this model (you may need to request access for gated models).

    Deploy a sample vLLM deployment with the proper protocol to work with the LLM Instance Gateway.