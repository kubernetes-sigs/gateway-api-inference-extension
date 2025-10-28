=== "CPU-Based Model Server"

    ???+ warning

        CPU deployment can be unreliable i.e. the pods may crash/restart because of resource contraints.

    This setup is using the formal `vllm-cpu` image, which according to the documentation can run vLLM on x86 CPU platform.
    For this setup, we use approximately 9.5GB of memory and 12 CPUs for each replica.

    While it is possible to deploy the model server with less resources, this is not recommended. For example, in our tests, loading the model using 8GB of memory and 1 CPU was possible but took almost 3.5 minutes and inference requests took unreasonable time. In general, there is a tradeoff between the memory and CPU we allocate to our pods and the performance. The more memory and CPU we allocate the better performance we can get.

    After running multiple configurations of these values we decided in this sample to use 9.5GB of memory and 12 CPUs for each replica, which gives reasonable response times. You can increase those numbers and potentially may even get better response times. For modifying the allocated resources, adjust the numbers in [cpu-deployment.yaml](https://github.com/kubernetes-sigs/gateway-api-inference-extension/raw/main/config/manifests/vllm/cpu-deployment.yaml) as needed.

    Deploy a sample vLLM deployment with the proper protocol to work with the LLM Instance Gateway.
