# ACK (AlibabaCloud Container Service for Kubernetes) Gateway with Inference Extension

## Table of Contents

| Extension Version Tested | Profile Tested | Implementation Version | Mode    | Report                                                                     |
|--------------------------|----------------|------------------------|---------|----------------------------------------------------------------------------|
| v0.5.1                  | Gateway        | v1.4.0-apsara.3         | default | [v1.4.0-apsara.3 Gateway report](./v1.4.0-apsara.3-gateway-report.yaml) |

## Reproduce

ACK Gateway with Inference Extension conformance report can be reproduced by the following steps.

1. Create an ACK managed cluster following [guide](https://www.alibabacloud.com/help/en/ack/ack-managed-and-ack-dedicated/user-guide/create-an-ack-managed-cluster-2/).

2. Install ACK Gateway with Inference Extension following Step 2 in [documentation](https://www.alibabacloud.com/help/en/cs/user-guide/intelligent-routing-and-traffic-management-with-ack-gateway-inference-extension).

3. Open `SELF_HOST_EPP` mode for the Gateway controller with following command:
    ```
    kubectl patch deployment envoy-gateway -n envoy-gateway-system --type='json' -p='[
    {
        "op": "add",
        "path": "/spec/template/spec/containers/0/env/-",
        "value": {
          "name": "SELF_HOST_EPP",
          "value": "true"
        }
      },
      {
        "op": "add",
        "path": "/spec/template/spec/containers/1/env/-",
        "value": {
          "name": "SELF_HOST_EPP",
          "value": "true"
        }
      }
    ]'
    ```

4. Run the following command from within the [Gateway API inference extension repo](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/v0.5.1).

    ```
    go test -timeout 30m ./conformance -v -args \
        --gateway-class=ack-gateway \
        --conformance-profiles=Gateway \
        --organization=AlibabaCloud \
        --project=ack-gateway-with-inference-extension \
        --url=https://www.alibabacloud.com/help/en/cs/user-guide/gateway-with-inference-extension-overview \
        --version=v1.4.0-apsara.3 \
        --contact=https://smartservice.console.aliyun.com/service/create-ticket \
        --report-output="/path/to/report"
    ```
