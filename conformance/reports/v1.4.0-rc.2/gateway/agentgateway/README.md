# Agentgateway

## Table of Contents

| Extension Version Tested | Profile Tested | Implementation Version | Mode    | Report                                                |
|--------------------------|----------------|------------------------|---------|-------------------------------------------------------|
| v1.4.0-rc.2              | Gateway        | [v1.0.0-alpha.4](https://github.com/agentgateway/agentgateway/releases/tag/v1.0.0-alpha.4) | default | [inference-v1.0.0-alpha.4 report](./inference-v1.0.0-alpha.4-report.yaml) |

## Reproduce

This follows the upstream [Agentgateway Gateway API conformance report](https://github.com/kubernetes-sigs/gateway-api/tree/main/conformance/reports/v1.5.0/agentgateway-agentgateway), but runs the Gateway API Inference Extension suite instead of the Gateway API suite.

### Steps

1. Clone the agentgateway repository and check out the tested release:

   ```sh
   export VERSION=v1.0.0-alpha.4
   git clone https://github.com/agentgateway/agentgateway.git
   cd agentgateway
   git checkout tags/$VERSION
   ```

2. Bootstrap a KinD cluster with the required Gateway API and Gateway API Inference Extension components:

   ```sh
   export GIE_CRD_VERSION=v1.4.0-rc.2
   ./controller/test/setup/setup-kind-ci.sh
   ```

3. Run the Gateway API Inference Extension conformance suite:

   ```sh
   make -C controller gie-conformance
   ```

4. View and verify the conformance report:

   ```sh
   cat controller/_test/conformance/inference-$VERSION-report.yaml
   ```
