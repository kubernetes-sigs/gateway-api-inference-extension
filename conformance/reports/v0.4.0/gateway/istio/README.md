# Istio (Gateway Profile Conformance)


## Table of Contents

| Extension Version Tested | Profile Tested | Implementation Version | Mode    | Report                                                                     |
|--------------------------|----------------|------------------------|---------|----------------------------------------------------------------------------|
| v0.4.0 | Gateway | 1.27-alpha.0551127f00634403cddd4634567e65a8ecc499a7 | default | [1.27-alpha.0551127f00634403cddd4634567e65a8ecc499a7-default-gateway-report.yaml](./1.27-alpha.0551127f00634403cddd4634567e65a8ecc499a7-default-gateway-report.yaml) |
| ...                      | ...            | ...                    | ...     | ...                                                                        |

## Reproduce

To reproduce the test environment and run the conformance tests, choose **either** minikube **or** OpenShift as your deployment environment:

### Option A: Minikube Environment

```bash
# Set up the complete environment (minikube, istio, CRDs, TLS)
make setup-env-minikube

# Run conformance tests
make run-tests

# Clean up resources when done
make clean
```

### Option B: OpenShift Environment

```bash
# Set up the complete environment (openshift, istio, CRDs, TLS)
make setup-env-openshift

# Run conformance tests
make run-tests

# Clean up resources when done
make clean
```

### Individual Setup Steps

If you prefer to run individual setup steps:

```bash
# Choose ONE of these deployment environments:
make setup-minikube     # Option A: Set up minikube with metallb
# OR
make setup-openshift    # Option B: Set up OpenShift environment

# Then continue with common setup steps:
make setup-istio                        # Install Istio (uses ISTIO_PROFILE variable)
# OR use environment-specific targets:
make setup-istio-minikube               # Install Istio for minikube environment
make setup-istio-openshift              # Install Istio for OpenShift environment

make setup-gateway-api-crds             # Apply Gateway API CRDs
make setup-inference-extension-crds     # Apply Inference Extension CRDs
# OR
make setup-crds                         # Apply all CRDs (convenience target)
make setup-tls                          # Set up TLS for EPP
```

### CRD Setup Options

The CRD setup is split into two targets for flexibility:

- `make setup-gateway-api-crds` - Install only Gateway API CRDs (if you have a base Gateway API environment)
- `make setup-inference-extension-crds` - Install only Inference Extension CRDs (if Gateway API is already installed)
- `make setup-crds` - Install both (convenience target for full setup)

Different environments might require different combinations depending on what's already installed.

### Version Configuration

The Makefile uses configurable version variables that can be overridden:

```bash
# Use default versions
make setup-env-minikube

# Override specific versions
make setup-env-minikube ISTIO_VERSION=1.28.0
make setup-env-openshift GATEWAY_API_VERSION=v1.4.0 INFERENCE_EXTENSION_VERSION=v0.5.0

# Use specific Istio profile
make setup-istio ISTIO_PROFILE=openshift

# Use custom Istio registry
make setup-istio ISTIO_HUB=docker.io/istio

# Run tests from custom directory
make run-tests TEST_BASE_DIR=../../../../../

# See all available options
make help
```

### Available Variables:

**Setup Variables:**
- `GATEWAY_API_VERSION` - Gateway API version (default: v1.3.0)
- `INFERENCE_EXTENSION_VERSION` - Inference Extension version (default: v0.4.0)  
- `ISTIO_VERSION` - Istio version (default: 1.27-alpha.0551127f00634403cddd4634567e65a8ecc499a7)
- `ISTIO_HUB` - Istio container registry hub (default: gcr.io/istio-testing)
- `ISTIO_PROFILE` - Istio profile (default: minimal)

**Conformance Test Variables:**
- `IMPLEMENTATION_VERSION` - Implementation version for report (default: same as ISTIO_VERSION)
- `MODE` - Test mode (default: default)
- `PROFILE` - Conformance profile (default: gateway)
- `ORGANIZATION` - Organization name (default: istio)
- `PROJECT` - Project name (default: istio)
- `URL` - Project URL (default: https://istio.io)
- `CONTACT` - Contact information (default: @istio/maintainers)

**Directory Variables:**
- `TEST_BASE_DIR` - Test suite base directory (default: ../../../../..)
- `REPORT_BASE_DIR` - Report output directory (default: .)

> **Note**: This Makefile expects to be run from within the `gateway-api-inference-extension` repository structure. The test suite is located at the repository root, while conformance reports are generated in the current directory.

### Conformance Testing

For detailed information about conformance testing, report generation, and requirements, see the [conformance README](../../README.md).

For available commands and options specific to this Istio implementation, run `make help` or see the `Makefile` in this directory.
