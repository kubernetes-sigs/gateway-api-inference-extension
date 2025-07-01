# Istio (Gateway Profile Conformance)


## Table of Contents

| Extension Version Tested | Profile Tested | Implementation Version | Mode    | Report                                                                     |
|--------------------------|----------------|------------------------|---------|----------------------------------------------------------------------------|
| v0.4.0                   | Gateway        | 1.27-alpha...          | default | [v1.27-alpha Dev report](./istio-1.27-alpha-dev-report.yaml) |
| ...                      | ...            | ...                    | ...     | ...                                                                        |

## Reproduce

```
# minikube

minikube start
minikube addons enable metallb

cat << EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
data:
  config: |
    address-pools:
    - name: default
      protocol: layer2
      addresses:
      - 192.168.49.100-192.168.49.200
metadata:
  name: config
  namespace: metallb-system
EOF

# istio

istioctl install --set profile=minimal --set values.global.hub=gcr.io/istio-testing --set values.global.tag=1.27-alpha.5bcfdaceaca13cfbbd876d142ab069bb14731700

# CRDs

kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.3.0/standard-install.yaml
kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/heads/main/config/crd/bases/inference.networking.x-k8s.io_inferencepools.yaml
kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/heads/main/config/crd/bases/inference.networking.x-k8s.io_inferencemodels.yaml


# Enforce TLS for EPP

kubectrl create namespace gateway-conformance-app-backend

cat << EOF | kubectl apply -f - 
apiVersion: networking.istio.io/v2
kind: DestinationRule
metadata:
  name: primary-endpoint-picker-tls
  namespace: gateway-conformance-app-backend
spec:
  host: primary-endpoint-picker-svc
  trafficPolicy:
      tls:
        mode: SIMPLE
        insecureSkipVerify: true
---
apiVersion: networking.istio.io/v1
kind: DestinationRule
metadata:
  name: secondary-endpoint-picker-tls
  namespace: gateway-conformance-app-backend
spec:
  host: secondary-endpoint-picker-svc
  trafficPolicy:
      tls:
        mode: SIMPLE
        insecureSkipVerify: true
EOF

# gie conformance test suite

git clone git@github.com:kubernetes-sigs/gateway-api-inference-extension.git; cd gateway-api-inference-extension

#go test...

```