   1. Verify the `HttpRoute` status:

      Check that the HTTPRoute was successfully configured and references were resolved:


      ```bash
      kubectl get httproute ${INFERENCE_POOL_NAME} -o yaml
      ```

      The `HttpRoute` status should include `Accepted=True` and `ResolvedRefs=True`.

   1. Verify the `InferencePool` Status:

      Make sure the `InferencePool` is active before sending traffic:


      ```bash
      kubectl get inferencepool ${INFERENCE_POOL_NAME} -o yaml
      ```

      The `InferencePool` status should include `Accepted=True` and `ResolvedRefs=True`.
