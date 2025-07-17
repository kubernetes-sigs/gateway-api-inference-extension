# Developer Guide

Welcome to the Gateway API Inference Extension developer guide! This guide will help you set up your development environment, understand the project structure, and contribute effectively to the project.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Testing](#testing)
- [Code Generation](#code-generation)
- [Building and Deployment](#building-and-deployment)
- [Contributing Guidelines](#contributing-guidelines)
- [Debugging and Troubleshooting](#debugging-and-troubleshooting)

## Prerequisites

Before you start developing with Gateway API Inference Extension, ensure you have the following tools installed:

### Required Tools

- **Go 1.24+**: The project is written in Go. Check the [go.mod](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/go.mod) file for the exact version requirement.
- **Docker**: For building container images and running local clusters.
- **kubectl**: Kubernetes command-line tool for cluster interaction.
- **Kind**: For creating local Kubernetes clusters for development and testing.
- **Git**: For version control.

### Optional but Recommended Tools

- **golangci-lint**: For code linting (automatically installed via Makefile).
- **controller-gen**: For generating Kubernetes CRDs and RBAC (automatically installed via Makefile).
- **kustomize**: For Kubernetes manifest management (automatically installed via Makefile).

### Installation Commands

```bash
# Install Go (example for Linux/macOS)
# Visit https://golang.org/doc/install for platform-specific instructions

# Install Docker
# Visit https://docs.docker.com/get-docker/ for platform-specific instructions

# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Install Kind
go install sigs.k8s.io/kind@latest

# Verify installations
go version
docker --version
kubectl version --client
kind version
```

## Environment Setup

### 1. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/kubernetes-sigs/gateway-api-inference-extension.git
cd gateway-api-inference-extension

# Set up Go modules (if not using Go 1.11+ with modules enabled)
export GO111MODULE=on
```

### 2. Install Development Dependencies

The project uses a Makefile to manage dependencies. Most tools will be automatically installed when needed:

```bash
# Install all development dependencies
make help  # See all available targets

# Install specific tools manually if needed
make controller-gen
make golangci-lint
make kustomize
```

### 3. Set Up Local Kubernetes Cluster

```bash
# Create a Kind cluster for development
kind create cluster --name gaie-dev

# Verify cluster is running
kubectl cluster-info --context kind-gaie-dev
```

## Project Structure

Understanding the project structure is crucial for effective development:

```
gateway-api-inference-extension/
├── api/                          # Kubernetes API definitions
│   ├── config/                   # Configuration for API generation
│   └── v1alpha2/                 # API version definitions
├── cmd/                          # Main applications
│   ├── bbr/                      # Body-based Routing extension
│   └── epp/                      # Endpoint Picker (main component)
├── config/                       # Kubernetes manifests and configurations
│   ├── charts/                   # Helm charts
│   ├── crd/                      # Custom Resource Definitions
│   └── manifests/                # Deployment manifests
├── docs/                         # Documentation and proposals
├── hack/                         # Development scripts and utilities
├── internal/                     # Internal packages (not for external use)
├── pkg/                          # Public packages
│   ├── bbr/                      # Body-based Routing implementation
│   └── epp/                      # Endpoint Picker implementation
├── site-src/                     # Website source files
├── test/                         # Test files and utilities
│   ├── e2e/                      # End-to-end tests
│   ├── integration/              # Integration tests
│   └── testdata/                 # Test data and fixtures
└── tools/                        # Additional tools and utilities
```

### Key Components

- **EPP (Endpoint Picker)**: The main inference scheduler component
- **BBR (Body-based Routing)**: Extension for routing based on request body content
- **API**: Kubernetes Custom Resource Definitions for inference workloads
- **Controllers**: Kubernetes controllers for managing inference resources

## Development Workflow

### 1. Making Changes

When developing new features or fixing bugs, follow this workflow:

```bash
# 1. Create a new branch for your work
git checkout -b feature/your-feature-name

# 2. Make your changes
# Edit code, add tests, update documentation

# 3. Generate code and manifests
make generate

# 4. Run tests
make test

# 5. Verify code quality
make verify

# 6. Commit your changes
git add .
git commit -m "feat: add your feature description"

# 7. Push and create a pull request
git push origin feature/your-feature-name
```

### 2. Code Style and Standards

- Follow Go best practices and idioms
- Use meaningful variable and function names
- Add comments for complex logic
- Ensure all public functions have documentation comments
- Follow the existing code structure and patterns

### 3. Logging Guidelines

The project follows specific logging guidelines as documented in [docs/dev.md](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/docs/dev.md):

- Use `logr.Logger` interface for all logging
- Load logger from `context.Context` or pass as argument
- Follow Kubernetes instrumentation logging guidelines
- Use appropriate verbosity levels:
  - `logger.Info` (V(0)): Always visible to cluster operators
  - `logger.V(2).Info`: Default level for steady state information
  - `logger.V(3).Info`: Extended information about changes
  - `logger.V(4).Info`: Debug level verbosity
  - `logger.V(5).Info`: Trace level verbosity

## Testing

The project includes multiple types of tests to ensure code quality and functionality:

### Unit Tests

Run unit tests for the core packages:

```bash
# Run all unit tests
make test-unit

# Run tests for specific package
go test ./pkg/epp/...

# Run tests with coverage
go test -race -coverprofile=coverage.out ./pkg/...
```

### Integration Tests

Integration tests verify component interactions:

```bash
# Run integration tests
make test-integration

# Run specific integration tests
go test ./test/integration/epp/...
```

### End-to-End Tests

E2E tests validate the complete system functionality:

```bash
# Run e2e tests (requires running cluster)
make test-e2e

# Run e2e tests with custom manifest
E2E_MANIFEST_PATH=config/manifests/your-manifest.yaml make test-e2e
```

### Test Data and Fixtures

Test data is organized in several locations:

- `test/testdata/`: General test data and fixtures
- `internal/xds/translator/testdata/`: XDS translation test cases
- `internal/gatewayapi/testdata/`: Gateway API translation test cases

### Adding New Tests

When adding new functionality, ensure you add tests in the appropriate locations:

1. **Unit tests**: Add to the same package as your code
2. **Integration tests**: Add to `test/integration/`
3. **E2E tests**: Add to `test/e2e/e2e_test.go`
4. **CEL validation tests**: Add to `test/cel-validation/`

## Code Generation

The project uses code generation for Kubernetes resources and client code:

### Generate All Code

```bash
# Generate all code (CRDs, clients, manifests)
make generate
```

### Individual Generation Commands

```bash
# Generate CRDs and RBAC
make manifests

# Generate client code
make code-generator

# Generate API documentation
make api-ref-docs
```

### When to Regenerate

Regenerate code when you:

- Modify API types in `api/v1alpha2/`
- Add new Kubernetes resources
- Change controller RBAC requirements
- Update API documentation

## Building and Deployment

### Building Container Images

The project supports building multiple container images:

#### EPP (Endpoint Picker) Image

```bash
# Build EPP image locally
make image-local-build

# Build and load into local Docker registry
make image-local-load

# Build and push to registry
make image-local-push

# Build for Kind cluster
make image-kind
```

#### BBR (Body-based Routing) Image

```bash
# Build BBR image locally
make bbr-image-local-build

# Build and load into local Docker registry
make bbr-image-local-load

# Build and push to registry
make bbr-image-local-push

# Build for Kind cluster
make bbr-image-kind
```

#### LoRA Syncer Image

```bash
# Build syncer image
make syncer-image-build

# Build and push syncer image
make syncer-image-push
```

### Deployment to Kubernetes

#### Install CRDs

```bash
# Install Custom Resource Definitions
make install

# Uninstall CRDs
make uninstall
```

#### Deploy with Helm

```bash
# Push Helm charts
make inferencepool-helm-chart-push
make bbr-helm-chart-push
```

#### Custom Deployment

You can customize deployment by setting environment variables:

```bash
# Custom image registry
export IMAGE_REGISTRY=your-registry.com/your-project

# Custom image tag
export EXTRA_TAG=your-custom-tag

# Custom platforms for multi-arch builds
export PLATFORMS=linux/amd64,linux/arm64

# Build with custom settings
make image-build
```

### Local Development Setup

For local development, you can set up a complete environment:

```bash
# 1. Create Kind cluster
kind create cluster --name gaie-dev

# 2. Install CRDs
make install

# 3. Build and load images
make image-kind

# 4. Deploy test manifests
kubectl apply -f test/testdata/

# 5. Verify deployment
kubectl get pods -A
```

## Contributing Guidelines

### Before Contributing

1. **Read the Code of Conduct**: Familiarize yourself with the [Kubernetes Code of Conduct](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/code-of-conduct.md)
2. **Sign the CLA**: Complete the [Contributor License Agreement](https://git.k8s.io/community/CLA.md)
3. **Join the Community**:
   - Join the [#gateway-api-inference-extension](https://kubernetes.slack.com/messages/gateway-api-inference-extension) Slack channel
   - Attend weekly community meetings (Thursdays 10AM PDT)

### Development Process

1. **Find or Create an Issue**: Look for issues labeled `good first issue` or `help wanted`
2. **Discuss Your Approach**: Comment on the issue to discuss your proposed solution
3. **Fork and Branch**: Create a fork and feature branch for your work
4. **Develop and Test**: Implement your changes with appropriate tests
5. **Submit a Pull Request**: Follow the PR template and guidelines

### Pull Request Requirements

Before your PR can be merged, ensure:

- [ ] All tests pass (`make test`)
- [ ] Code passes linting (`make lint`)
- [ ] Code is properly formatted (`make fmt`)
- [ ] Documentation is updated if needed
- [ ] Commit messages follow conventional commit format
- [ ] PR description clearly explains the changes

### Required Make Targets

Ensure these make targets pass before submitting:

```bash
# Run all tests
make test

# Run linting
make lint

# Verify all checks
make verify
```

## Debugging and Troubleshooting

### Common Issues

#### Build Failures

```bash
# Clean and rebuild
make clean
make generate

# Check Go version compatibility
go version  # Should be 1.24+

# Update dependencies
go mod tidy
```

#### Test Failures

```bash
# Run tests with verbose output
go test -v ./pkg/...

# Run specific test
go test -v -run TestSpecificFunction ./pkg/epp/

# Check test environment
kubectl cluster-info
```

#### Image Build Issues

```bash
# Check Docker daemon
docker info

# Verify Kind cluster
kind get clusters

# Check image loading
docker images | grep epp
```

### Debugging Running Components

#### Enable Debug Logging

```bash
# Set log verbosity
kubectl patch deployment epp-controller -p '{"spec":{"template":{"spec":{"containers":[{"name":"manager","args":["--v=4"]}]}}}}'

# View logs
kubectl logs -f deployment/epp-controller
```

#### Inspect Resources

```bash
# Check CRDs
kubectl get crds

# Check inference pools
kubectl get inferencepools -A

# Check events
kubectl get events --sort-by=.metadata.creationTimestamp
```

### Getting Help

If you encounter issues:

1. **Check existing issues**: Search [GitHub issues](https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues)
2. **Ask in Slack**: Use the [#gateway-api-inference-extension](https://kubernetes.slack.com/messages/gateway-api-inference-extension) channel
3. **Attend community meetings**: Join weekly meetings for real-time help
4. **Create a detailed issue**: Include logs, environment details, and reproduction steps

### Useful Resources

- [Project README](https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/README.md)
- [API Documentation](https://gateway-api-inference-extension.sigs.k8s.io/)
- [Proposals and Design Docs](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/docs/proposals)
- [Gateway API Documentation](https://gateway-api.sigs.k8s.io/)
- [Kubernetes Development Guide](https://github.com/kubernetes/community/tree/master/contributors/devel)

---

Happy coding! 🚀

For questions or suggestions about this developer guide, please open an issue or reach out in the Slack channel.
