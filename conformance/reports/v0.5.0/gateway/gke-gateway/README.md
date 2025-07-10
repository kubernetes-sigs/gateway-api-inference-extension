# GKE (Google Kubernetes Engine) Gateway

## Table of Contents

| Extension Version Tested | Profile Tested | Implementation Version | Mode    | Report                                                                     |
|--------------------------|----------------|------------------------|---------|----------------------------------------------------------------------------|
| v0.5.0-dev                   | Gateway        | v1.32.3-gke.1211000                | default | [v1.32.3 Gateway report](./experimental-v1.32.3-default-gateway-report.yaml) |
| ...                      | ...            | ...                    | ...     | ...                                                                        |

## Reproduce

GKE Gateway conformance report can be reproduced by the following steps.

1. Create a GKE cluster with Gateway API enabled

```
gcloud container clusters create "${cluster_name}" --gateway-api=standard --location="${location}"
```

2. Install the InferencePool and InferenceModel Custom Resource Definition (CRDs) in your GKE cluster, run the following command:
```
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v0.5.0/manifests.yaml
```

3. run the following command from within the [Gateway API inference extension repo](https://github.com/kubernetes-sigs/gateway-api-inference-extension)

```
go test -timeout 30m ./conformance -v -args \
    --gateway-class=gke-l7-regional-external-managed \
    --conformance-profiles=Gateway \
    --organization=GKE \
    --project=gke-gateway \
    --url=https://cloud.google.com/kubernetes-engine/docs/concepts/gateway-api \
    --version=v1.32.3-gke.1211000 \
    --contact=gke-gateway-dev@google.com \
    --report-output="/path/to/report"
```

or run a single conformance test case

```
go test ./conformance -v -args \
    -gateway-class=gke-l7-regional-external-managed \
    -run-test=InferencePoolAccepted
```