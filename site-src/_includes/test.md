### Try it out

   Wait until the gateway is ready.

   ```bash
   IP=$(kubectl get gateway/inference-gateway -o jsonpath='{.status.addresses[0].value}')
   PORT=80

   curl -i ${IP}:${PORT}/v1/completions -H 'Content-Type: application/json' -d '{
   "model": "food-review-1",
   "prompt": "Write as if you were a critic: San Francisco",
   "max_tokens": 100,
   "temperature": 0
   }'
   ```
