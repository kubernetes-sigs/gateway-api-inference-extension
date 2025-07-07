# GKE (Google Kubernetes Engine) Gateway

## Table of Contents

|API channel|Implementation version|Mode|Report|
|-----------|----------------------|----|------|
|standard|1.32.3-gke.1211000|gke-l7-regional-external-managed|[v1.32.3 rxlb report](./standard-1.32.3-rxlb-report.yaml)|

## Reproduce

GKE Gateway conformance report can be reproduced by the following steps.

1. Create a GKE cluster with Gateway API enabled

```
gcloud container clusters create "${cluster_name}" --gateway-api=standard --location="${location}"
```

2. Install the InferencePool and InferenceModel Custom Resource Definition (CRDs) in your GKE cluster, run the following command:
```
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v0.3.0/manifests.yaml
```

3. run the following command from within the [Gateway API inference extension repo](https://github.com/kubernetes-sigs/gateway-api-inference-extension)

```
go test -timeout 30m ./conformance -v -args \
    -gateway-class=gke-l7-regional-external-managed \
    -organization=GKE \
    -project=gke-gateway \
    -url=https://github.com/kubernetes-sigs/gateway-api-inference-extension \
    -version=1.32.3-gke.1211000 \
    -contact=gke-gateway-dev@google.com \
    -report-output="/path/to/report"
```

or run a single conformance test case

```
go test ./conformance -v -args \
    -gateway-class=gke-l7-regional-external-managed \
    -run-test=InferencePoolAccepted
```