# GKE (Google Kubernetes Engine) Gateway

## Table of Contents

| Extension Version Tested | Profile Tested | Implementation Version | Mode    | Report                                                                     |
|--------------------------|----------------|------------------------|---------|----------------------------------------------------------------------------|
| v0.5.0                   | Gateway        | 1.32.4-gke.1415000                | gke-l7-regional-external-managed | [v1.32.4 Gateway report](./standard-v1.32.4-rxlb-gateway-report.yaml) |
| ...                      | ...            | ...                    | ...     | ...                                                                        |

## Reproduce

GKE Gateway conformance report can be reproduced by the following steps.

1. Create a GKE cluster with Gateway API enabled.

    ```
    gcloud container clusters create "${cluster_name}" --gateway-api=standard --location="${location}"
    ```
1. Deploy GKE gateway following [guide](https://cloud.google.com/kubernetes-engine/docs/how-to/deploying-gateways#configure_a_proxy-only_subnet).

1. Install the InferencePool and InferenceObjective Custom Resource Definition (CRDs) in your GKE cluster, run the following command:
    ```
    kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v0.5.0/manifests.yaml
    ```

1. Run the following command from within the [Gateway API inference extension repo](https://github.com/kubernetes-sigs/gateway-api-inference-extension).

    ```
    go test -timeout 30m ./conformance -v -args \
        --gateway-class=gke-l7-regional-external-managed \
        --conformance-profiles=Gateway \
        --organization=GKE \
        --project=gke-gateway \
        --url=https://cloud.google.com/kubernetes-engine/docs/concepts/gateway-api \
        --version=1.32.4-gke.1415000 \
        --contact=gke-gateway-dev@google.com \
        --report-output="/path/to/report"
    ```

    or run a single conformance test case.

    ```
    go test ./conformance -v -args \
        -gateway-class=gke-l7-regional-external-managed \
        -run-test=InferencePoolAccepted
    ```