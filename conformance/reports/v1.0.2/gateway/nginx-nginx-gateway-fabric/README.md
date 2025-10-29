# Nginx NGINX Gateway Fabric

## Table of Contents

| Extension Version Tested | Profile Tested | Implementation Version | Mode    | Report                                                                     |
|--------------------------|----------------|------------------------|---------|----------------------------------------------------------------------------|
| v1.0.2                   | Gateway        | v2.2.0                 | default | [v2.2.0 report](./inference-v2.2.0-report.yaml)    

## Reproduce

To reproduce results, clone the NGF repository:

```shell
git clone https://github.com/nginx/nginx-gateway-fabric.git && cd nginx-gateway-fabric/tests
```

Follow the steps in the [NGINX Gateway Fabric Testing](https://github.com/nginx/nginx-gateway-fabric/blob/main/tests/README.md#conformance-testing) document to run the conformance tests. If you are running tests on the `edge` version, then you don't need to build any images. Otherwise, you'll need to check out the specific release tag that you want to test, and then build and load the images onto your cluster, per the steps in the README.

Note: Enable this flag to install all CRDs and required resources:

```shell
export ENABLE_INFERENCE_EXTENSION=true
```

After running, see the conformance report:

```shell
cat conformance-profile-inference.yaml
```
