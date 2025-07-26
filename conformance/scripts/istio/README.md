# Istio Conformance Testing

This directory contains the Makefile and scripts for running Gateway API Inference Extension conformance tests against Istio implementations.

## Quick Start

To reproduce the test environment and run the conformance tests, choose **one** of these deployment environments:

### Option A: Minikube Environment

```bash
# Set up the complete environment (minikube, istio, CRDs, TLS)
make setup-env-minikube

# Run conformance tests
make run-tests

# Clean up resources when done
make clean
```

### Option B: Kind Environment

```bash
# Set up the complete environment (kind, istio, CRDs, TLS)
make setup-env-kind

# Run conformance tests
make run-tests

# Clean up resources when done
make clean
```

### Option C: OpenShift Environment

```bash
# Set up the complete environment (openshift, istio, CRDs, TLS)
make setup-env-openshift

# Run conformance tests
make run-tests

# Clean up resources when done
make clean
```

## Individual Setup Steps

If you prefer to run individual setup steps:

```bash
# Choose ONE of these deployment environments:
make setup-minikube     # Option A: Set up minikube with metallb
# OR
make setup-kind         # Option B: Set up kind cluster with metallb
# OR
make setup-openshift    # Option C: Set up OpenShift environment

# Use specific Kubernetes version (works for both kind and minikube):
make setup-minikube KUBERNETES_VERSION=v1.31.9
make setup-kind KUBERNETES_VERSION=v1.31.9

# Then continue with common setup steps:
make setup-istio                        # Install Istio (automatically detects hub)
# OR use environment-specific targets:
make setup-istio-minikube               # Install Istio for minikube environment
make setup-istio-kind                   # Install Istio for kind environment
make setup-istio-openshift              # Install Istio for OpenShift environment

make setup-gateway-api-crds             # Apply Gateway API CRDs
make setup-inference-extension-crds     # Apply Inference Extension CRDs
# OR
make setup-crds                         # Apply all CRDs (convenience target)
make setup-tls                          # Set up TLS for EPP
```

## CRD Setup Options

The CRD setup is split into two targets for flexibility:

- `make setup-gateway-api-crds` - Install only Gateway API CRDs (if you have a base Gateway API environment)
- `make setup-inference-extension-crds` - Install only Inference Extension CRDs (if Gateway API is already installed)
- `make setup-crds` - Install both (convenience target for full setup)

Different environments might require different combinations depending on what's already installed.

## Version Configuration

The Makefile uses configurable version variables that can be overridden:

```bash
# Use default versions
make setup-env-minikube

# Override specific versions
make setup-env-minikube ISTIO_VERSION=1.28.0
make setup-env-kind GATEWAY_API_VERSION=v1.4.0 INFERENCE_EXTENSION_VERSION=v0.5.0
make setup-env-openshift GATEWAY_API_VERSION=v1.4.0 INFERENCE_EXTENSION_VERSION=v0.5.0

# Use specific Kubernetes version (works for both kind and minikube)
make setup-env-minikube KUBERNETES_VERSION=v1.31.9
make setup-env-kind KUBERNETES_VERSION=v1.31.9

# Combine version overrides
make setup-env-minikube KUBERNETES_VERSION=v1.31.9 ISTIO_VERSION=1.28.0

# Use environment-specific Istio installation
make setup-istio-minikube                  # Install Istio for minikube environment
make setup-istio-kind                      # Install Istio for kind environment
make setup-istio-openshift                 # Install Istio for OpenShift environment

# Run conformance tests
make run-tests                             # Run all conformance tests
make run-tests RUN_TEST=InferencePoolAccepted  # Run specific test only

# See all available options
make help
```

## Report Generation

Reports are automatically generated in the appropriate directory structure based on the variables you use:

```bash
# Default: reports in ../../reports/v0.4.0/gateway/istio/
make run-tests

# Custom: reports in ../../reports/v0.5.0/mesh/envoy/
make run-tests PROJECT=envoy PROFILE=mesh INFERENCE_EXTENSION_VERSION=v0.5.0
```

The report structure follows: `reports/[INFERENCE_EXTENSION_VERSION]/[PROFILE]/[PROJECT]/`

## Available Variables

### Main Variables:
- `GATEWAY_API_VERSION` - Gateway API version (default: v1.3.0)
- `INFERENCE_EXTENSION_VERSION` - Inference Extension version (default: v0.4.0)  
- `KUBERNETES_VERSION` - Kubernetes version for kind/minikube (default: v1.30.13)
- `ISTIO_VERSION` - Istio version (default: 1.27-alpha.0551127f00634403cddd4634567e65a8ecc499a7)

### Conformance Test Variables:
- `MODE` - Test mode (default: default)
- `PROFILE` - Conformance profile (default: gateway)
- `ORGANIZATION` - Organization name (default: istio)
- `PROJECT` - Project name (default: istio)
- `URL` - Project URL (default: https://istio.io)
- `CONTACT` - Contact information (default: @istio/maintainers)
- `RUN_TEST` - Run specific test (default: empty, runs all tests)

### Advanced Variables:
For advanced users who need to customize internal behavior, additional variables are available including `ISTIO_HUB`, `ISTIO_PROFILE`, `TEST_BASE_DIR`, `REPORT_BASE_DIR`, and others. See `make help` for a complete list.

## Prerequisites

- Docker (for kind or container-based istioctl)
- OR minikube + virtualization driver
- OR OpenShift cluster access
- kubectl configured for your cluster
- Go (for running conformance tests)

## Troubleshooting

### Common Issues

1. **istioctl download fails**: The Makefile tries GitHub releases first, then container extraction. If both fail, manually install istioctl and set `ISTIOCTL_BIN` to point to it.

2. **Kubernetes version mismatch**: Ensure your `KUBERNETES_VERSION` is compatible with your chosen Istio version.

3. **Container registry issues**: If using a private registry, set `ISTIO_HUB` to your registry location.

4. **Network issues with metallb**: For kind/minikube, metallb configuration is automatically handled, but may require network adjustments in some environments.

### Getting Help

```bash
# See all available commands and variables
make help

# Check current variable values
make help | grep "current:"
```

## Directory Structure

> **Note**: This Makefile expects to be run from within the `conformance/scripts/istio/` directory in the `gateway-api-inference-extension` repository. The test suite is located at the repository root (`../../..`), while conformance reports are generated in the reports directory (`../../reports/`).

For detailed information about conformance testing, report generation, and requirements, see the [conformance README](../../reports/README.md). 