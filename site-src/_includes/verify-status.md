   1. Verify the `HttpRoute` status:

      Check that the HTTPRoute was successfully configured and references were resolved:

      ```bash
      kubectl get httproute vllm-llama3-8b-instruct -o yaml
      ```

      The `HttpRoute` status should include `Accepted=True` and `ResolvedRefs=True`.

   1. Verify the `InferencePool` Status:

      Make sure the `InferencePool` is active before sending traffic.

      ```bash
      kubectl describe inferencepools.inference.networking.k8s.io vllm-llama3-8b-instruct
      ```

      Check that the status shows `Accepted=True` and `ResolvedRefs=True`. This confirms the InferencePool is ready to handle traffic.