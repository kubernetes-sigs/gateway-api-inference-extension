# Higress

[Higress](https://higress.ai/) is a cloud-native API gateway built on Istio and
Envoy and is a CNCF project.

## Table of Contents

| Extension Version Tested | Profile Tested | Implementation Version | Mode | Report |
|--------------------------|----------------|------------------------|------|--------|
| v1.4.0 | Gateway | [v2.2.4](https://github.com/higress-group/higress/releases/tag/v2.2.4) | default | [v2.2.4 Gateway report](./v2.2.4-default-gateway-report.yaml) |

## Reproduce

Check out the Higress v2.2.4 release and create a Kubernetes cluster with
Gateway API v1.5.0 standard CRDs, the InferencePool CRD from Gateway API
Inference Extension v1.4.0, and MetalLB. Install `helm/core` with the v2.2.4
controller, Pilot, and Gateway images and `global.enableInferenceExtension=true`.

Apply `test/inference-extension/manifests/epp-tls.yaml`, then run the official
v1.4.0 conformance suite through the Higress wrapper:

```sh
INFERENCE_EXTENSION_VERSION=v1.4.0 \
INFERENCE_EXTENSION_SOURCE_DIR="$PWD/out/gateway-api-inference-extension-source/v1.4.0" \
INFERENCE_EXTENSION_REPORT="$PWD/out/gateway-api-inference-extension-v1.4.0-report.yaml" \
INFERENCE_EXTENSION_EXPECTED_PASSED=12 \
INFERENCE_EXTENSION_CONTACT=@higress-group/maintainers \
HIGRESS_CONFORMANCE_VERSION=v2.2.4 \
tools/hack/run-inference-extension-conformance.sh
```

The test uses the release images referenced by the Higress v2.2.4 Helm chart.
