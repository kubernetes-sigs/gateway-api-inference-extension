# ACK (AlibabaCloud Container Service for Kubernetes) Gateway with Inference Extension

## Table of Contents

| Extension Version Tested | Profile Tested | Implementation Version | Mode    | Report                                                                                          |
|--------------------------|----------------|------------------------|---------|-------------------------------------------------------------------------------------------------|
| v1.5.0                   | Gateway        | [v1.6.1-apsara.3](https://www.alibabacloud.com/help/en/cs/user-guide/gateway-with-inference-extension-overview) | default | [v1.6.1-apsara.3 default Gateway report](./v1.6.1-apsara.3-default-gateway-report.yaml)          |

## Reproduce

ACK Gateway with Inference Extension conformance report can be reproduced by the following steps.

1. Create an ACK managed cluster following [guide](https://www.alibabacloud.com/help/en/ack/ack-managed-and-ack-dedicated/user-guide/create-an-ack-managed-cluster-2/).

2. Install ACK Gateway with Inference Extension at version **v1.6.1-apsara.3** following Step 2 in [documentation](https://www.alibabacloud.com/help/en/cs/user-guide/intelligent-routing-and-traffic-management-with-ack-gateway-inference-extension).

3. Run the following command from within the [Gateway API inference extension repo](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/v1.5.0).

    ```
    go test -timeout 45m ./conformance -v -args \
        --gateway-class=ack-gateway \
        --conformance-profiles=Gateway \
        --organization=AlibabaCloud \
        --project=ack-gateway-with-inference-extension \
        --url=https://www.alibabacloud.com/help/en/cs/user-guide/gateway-with-inference-extension-overview \
        --version=v1.6.1-apsara.3 \
        --contact=https://smartservice.console.aliyun.com/service/create-ticket \
        --allow-crds-mismatch \
        --report-output="/path/to/report"
    ```
