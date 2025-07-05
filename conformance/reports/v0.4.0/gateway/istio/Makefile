# Version variables - can be overridden via environment or command line
# Example: make all ISTIO_VERSION=1.28.0
GATEWAY_API_VERSION ?= v1.3.0
INFERENCE_EXTENSION_VERSION ?= v0.4.0
ISTIO_VERSION ?= 1.27-alpha.0551127f00634403cddd4634567e65a8ecc499a7
ISTIO_HUB ?= 
ISTIO_PROFILE ?= minimal

# Conformance test variables
IMPLEMENTATION_VERSION ?= $(ISTIO_VERSION)
MODE ?= default
PROFILE ?= gateway
ORGANIZATION ?= istio
PROJECT ?= istio
URL ?= https://istio.io
CONTACT ?= @istio/maintainers
RUN_TEST ?=

# Directory variables
TEST_BASE_DIR ?= ../../../../..
REPORT_BASE_DIR ?= .

# Kubernetes version (shared between kind and minikube)
KUBERNETES_VERSION ?= v1.30.13

# Internal variables (derived from KUBERNETES_VERSION)
KIND_NODE_IMAGE ?= kindest/node:$(KUBERNETES_VERSION)
MINIKUBE_K8S_VERSION ?= $(KUBERNETES_VERSION)

# Istioctl binary variables
ISTIOCTL_DIR ?= /tmp
ISTIOCTL_BIN ?= $(ISTIOCTL_DIR)/istioctl-$(ISTIO_VERSION)

# YAML content for readability
define METALLB_CONFIG
apiVersion: v1
kind: ConfigMap
data:
  config: |
    address-pools:
    - name: default
      protocol: layer2
      addresses:
      - NETWORK_PREFIX.100-NETWORK_PREFIX.200
metadata:
  name: config
  namespace: metallb-system
endef

# Multi-line YAML configuration for metallb (kind)
define METALLB_KIND_CONFIG
apiVersion: metallb.io/v1beta1
kind: IPAddressPool
metadata:
  name: example
  namespace: metallb-system
spec:
  addresses:
  - NETWORK_PREFIX.200-NETWORK_PREFIX.250
---
apiVersion: metallb.io/v1beta1
kind: L2Advertisement
metadata:
  name: empty
  namespace: metallb-system
endef

define TLS_DESTINATION_RULES
apiVersion: networking.istio.io/v1
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
endef

.PHONY: setup-env-minikube setup-env-kind setup-env-openshift setup-minikube setup-kind setup-openshift setup-istio setup-istio-minikube setup-istio-kind setup-istio-openshift setup-gateway-api-crds setup-inference-extension-crds setup-crds setup-tls run-tests update-readme clean clean-reports help

# Setup complete environment with minikube
setup-env-minikube: setup-minikube setup-istio-minikube setup-gateway-api-crds setup-inference-extension-crds setup-tls

# Setup complete environment with kind
setup-env-kind: setup-kind setup-istio-kind setup-gateway-api-crds setup-inference-extension-crds setup-tls

# Setup complete environment with OpenShift
setup-env-openshift: setup-openshift setup-istio-openshift setup-gateway-api-crds setup-inference-extension-crds setup-tls

# Show help information
help:
	@echo "Gateway API Inference Extension Conformance Test Setup"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "📋 QUICK SETUP - All-in-one targets"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  setup-env-minikube               - Setup complete environment with minikube"
	@echo "  setup-env-kind                   - Setup complete environment with kind"
	@echo "  setup-env-openshift              - Setup complete environment with OpenShift"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "🚀 MINIKUBE ENVIRONMENT - For local development with VMs"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  setup-minikube                   - Setup minikube with metallb"
	@echo "  setup-istio-minikube             - Install Istio for minikube environment"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "🐳 KIND ENVIRONMENT - For local development with containers"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  setup-kind                       - Setup kind cluster with metallb"
	@echo "  setup-istio-kind                 - Install Istio for kind environment"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "🏢 OPENSHIFT ENVIRONMENT - For enterprise deployments"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  setup-openshift                  - Setup OpenShift environment"
	@echo "  setup-istio-openshift            - Install Istio for OpenShift environment"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "⚙️  COMMON SETUP - Environment-agnostic targets"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  setup-istio                      - Install Istio (auto-downloads istioctl)"
	@echo "  setup-gateway-api-crds           - Install Gateway API CRDs"
	@echo "  setup-inference-extension-crds   - Install Inference Extension CRDs"
	@echo "  setup-crds                       - Install all CRDs"
	@echo "  setup-tls                        - Setup TLS for EPP"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "🧪 TESTING & REPORTING - Test execution and documentation"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  run-tests                        - Run conformance tests"
	@echo "  update-readme                    - Update README table with all available reports"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "🧹 CLEANUP - Resource cleanup"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  clean                            - Clean up resources"
	@echo "  clean-reports                    - Clean up generated reports and README table"
	@echo ""
	@echo "Main variables (can be overridden):"
	@echo "  GATEWAY_API_VERSION              - Gateway API version (current: $(GATEWAY_API_VERSION))"
	@echo "  INFERENCE_EXTENSION_VERSION      - Inference Extension version (current: $(INFERENCE_EXTENSION_VERSION))"
	@echo "  KUBERNETES_VERSION               - Kubernetes version for kind/minikube (current: $(KUBERNETES_VERSION))"
	@echo "  ISTIO_VERSION                    - Istio version (current: $(ISTIO_VERSION))"
	@echo ""
	@echo "📥 Automatic istioctl Management:"
	@echo "  • istioctl binary will be automatically downloaded based on ISTIO_VERSION"
	@echo "  • Hub (container registry) will be auto-detected from download source:"
	@echo "    - GitHub release → gcr.io/istio-release"
	@echo "    - Container extraction → gcr.io/istio-testing"
	@echo "  • Override with ISTIO_HUB if needed for custom registries"
	@echo ""
	@echo "Conformance test variables (can be overridden):"
	@echo "  MODE                             - Test mode (current: $(MODE))"
	@echo "  PROFILE                          - Conformance profile (current: $(PROFILE))"
	@echo "  ORGANIZATION                     - Organization name (current: $(ORGANIZATION))"
	@echo "  PROJECT                          - Project name (current: $(PROJECT))"
	@echo "  URL                              - Project URL (current: $(URL))"
	@echo "  CONTACT                          - Contact information (current: $(CONTACT))"
	@echo "  RUN_TEST                         - Run specific test (current: $(RUN_TEST))"
	@echo ""
	@echo "Internal variables (advanced users only):"
	@echo "  IMPLEMENTATION_VERSION           - Implementation version for report (current: $(IMPLEMENTATION_VERSION), derived from ISTIO_VERSION)"
	@echo "  ISTIO_HUB                        - Istio container registry hub (current: $(ISTIO_HUB), auto-detected if empty)"
	@echo "  ISTIO_PROFILE                    - Istio profile (current: $(ISTIO_PROFILE))"
	@echo "  TEST_BASE_DIR                    - Test suite base directory (current: $(TEST_BASE_DIR))"
	@echo "  REPORT_BASE_DIR                  - Report output directory (current: $(REPORT_BASE_DIR))"
	@echo "  ISTIOCTL_DIR                     - Directory for istioctl binaries (current: $(ISTIOCTL_DIR))"
	@echo "  ISTIOCTL_BIN                     - Path to versioned istioctl binary (current: $(ISTIOCTL_BIN))"
	@echo "  KIND_NODE_IMAGE                  - Kind node image (current: $(KIND_NODE_IMAGE))"
	@echo "  MINIKUBE_K8S_VERSION             - Minikube K8s version (current: $(MINIKUBE_K8S_VERSION))"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "💡 USAGE EXAMPLES"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "📋 Quick Setup Examples:"
	@echo "  make setup-env-minikube                    - Use default versions with minikube"
	@echo "  make setup-env-kind                        - Use default versions with kind"
	@echo "  make setup-env-openshift                   - Use default versions with OpenShift"
	@echo "  make setup-env-minikube ISTIO_VERSION=1.28.0 - Override Istio version"
	@echo "  make setup-env-minikube KUBERNETES_VERSION=v1.31.9 - Override Kubernetes version"
	@echo ""
	@echo "🚀 Minikube Examples:"
	@echo "  make setup-minikube                        - Setup minikube cluster with metallb"
	@echo "  make setup-istio-minikube                  - Install Istio for minikube environment"
	@echo "  make setup-minikube KUBERNETES_VERSION=v1.31.9 - Use Kubernetes 1.31.9"
	@echo ""
	@echo "🐳 Kind Examples:"
	@echo "  make setup-kind                            - Setup kind cluster with metallb"
	@echo "  make setup-istio-kind                      - Install Istio for kind environment"
	@echo "  make setup-kind KUBERNETES_VERSION=v1.31.9 - Use Kubernetes 1.31.9"
	@echo ""
	@echo "🏢 OpenShift Examples:"
	@echo "  make setup-openshift                       - Setup OpenShift environment"
	@echo "  make setup-istio-openshift                 - Install Istio for OpenShift environment"
	@echo ""
	@echo "⚙️  Common Setup Examples:"
	@echo "  make setup-istio ISTIO_PROFILE=openshift   - Install Istio with specific profile"
	@echo "  make setup-istio ISTIO_HUB=docker.io/istio - Install Istio from custom registry"
	@echo "  make setup-crds                            - Install all CRDs"
	@echo ""
	@echo "🧪 Testing Examples:"
	@echo "  make run-tests                             - Run conformance tests with default settings"
	@echo "  make run-tests TEST_BASE_DIR=../../../../../ - Run tests from custom directory"
	@echo "  make run-tests RUN_TEST=InferencePoolAccepted - Run specific test only"
	@echo "  make update-readme                         - Update README table with all available reports"
	@echo ""

# Setup minikube with metallb
setup-minikube:
	@echo "Setting up minikube with metallb..."
	@echo "Starting minikube with Kubernetes $(KUBERNETES_VERSION)..."
	minikube start --kubernetes-version=$(MINIKUBE_K8S_VERSION)
	minikube addons enable metallb
	@echo "Configuring metallb address pool..."
	$(file >/tmp/metallb-config.yaml,$(METALLB_CONFIG))
	@MINIKUBE_IP=$$(minikube ip); \
	NETWORK_PREFIX=$${MINIKUBE_IP%.*}; \
	echo "Using IP range: $${NETWORK_PREFIX}.100-$${NETWORK_PREFIX}.200"; \
	sed -i "s/NETWORK_PREFIX/$${NETWORK_PREFIX}/g" /tmp/metallb-config.yaml
	kubectl apply -f /tmp/metallb-config.yaml
	@rm -f /tmp/metallb-config.yaml

# Setup kind with metallb
setup-kind:
	@echo "Setting up kind with metallb..."
	@if ! kind get clusters | grep -q "^kind$$"; then \
		echo "Creating kind cluster with Kubernetes $(KUBERNETES_VERSION)..."; \
		kind create cluster --image $(KIND_NODE_IMAGE); \
	else \
		echo "Kind cluster already exists"; \
	fi
	@echo "Installing metallb..."
	kubectl apply -f https://raw.githubusercontent.com/metallb/metallb/v0.13.7/config/manifests/metallb-native.yaml
	@echo "Waiting for metallb to be ready..."
	kubectl wait --namespace metallb-system --for=condition=available deployment/controller --timeout=90s
	kubectl wait --namespace metallb-system --for=condition=ready pod --selector=component=controller --timeout=90s
	kubectl wait --namespace metallb-system --for=condition=ready pod --selector=component=speaker --timeout=90s
	@echo "Configuring metallb address pool..."
	$(file >/tmp/metallb-kind-config.yaml,$(METALLB_KIND_CONFIG))
	@DOCKER_NETWORK=$$(docker network inspect -f '{{range .IPAM.Config}}{{.Gateway}}{{end}}' kind); \
	NETWORK_PREFIX=$${DOCKER_NETWORK%.*}; \
	echo "Using IP range: $${NETWORK_PREFIX}.200-$${NETWORK_PREFIX}.250"; \
	sed -i "s/NETWORK_PREFIX/$${NETWORK_PREFIX}/g" /tmp/metallb-kind-config.yaml
	kubectl apply -f /tmp/metallb-kind-config.yaml
	@rm -f /tmp/metallb-kind-config.yaml

# Setup OpenShift (alternative to minikube)
setup-openshift:
	@echo "Setting up OpenShift environment..."
	@echo "Assuming OpenShift cluster is already running and kubectl/oc is configured"
	@echo "Verifying cluster connection..."
	kubectl cluster-info
	@echo "OpenShift environment ready"

# Download istioctl binary for the specific version (file target)
$(ISTIOCTL_BIN):
	@echo "Downloading istioctl version $(ISTIO_VERSION)..."
	@mkdir -p $(ISTIOCTL_DIR)
	@OS=$$(uname -s | tr '[:upper:]' '[:lower:]'); \
	ARCH=$$(uname -m); \
	case "$$ARCH" in \
		x86_64) ARCH=amd64 ;; \
		aarch64) ARCH=arm64 ;; \
		armv7l) ARCH=armv7 ;; \
	esac; \
	echo "Detecting OS: $$OS, Architecture: $$ARCH"; \
	SUCCESS=false; \
	echo "Trying GitHub releases..."; \
	if curl -fsSL "https://github.com/istio/istio/releases/download/$(ISTIO_VERSION)/istioctl-$(ISTIO_VERSION)-$$OS-$$ARCH.tar.gz" -o /tmp/istioctl.tar.gz 2>/dev/null; then \
		echo "Download successful, extracting..."; \
		if tar -xzf /tmp/istioctl.tar.gz -C /tmp 2>/dev/null; then \
			mv /tmp/istioctl $(ISTIOCTL_BIN); \
			rm -f /tmp/istioctl.tar.gz; \
			chmod +x $(ISTIOCTL_BIN); \
			echo "istioctl $(ISTIO_VERSION) downloaded to $(ISTIOCTL_BIN)"; \
			echo "github" > $(ISTIOCTL_BIN).source; \
			SUCCESS=true; \
		else \
			echo "ERROR: Failed to extract istioctl archive"; \
			rm -f /tmp/istioctl.tar.gz; \
		fi; \
	fi; \
	if [ "$$SUCCESS" = "false" ]; then \
		echo "Binary download failed, trying container-based extraction..."; \
		CONTAINER_IMAGE="gcr.io/istio-testing/istioctl:$(ISTIO_VERSION)"; \
		echo "Pulling container image: $$CONTAINER_IMAGE"; \
		if docker pull "$$CONTAINER_IMAGE" 2>/dev/null; then \
			echo "Extracting istioctl binary from container..."; \
			TEMP_CONTAINER=$$(docker create "$$CONTAINER_IMAGE"); \
			if docker cp "$$TEMP_CONTAINER:/usr/local/bin/istioctl" "$(ISTIOCTL_BIN)" 2>/dev/null; then \
				docker rm "$$TEMP_CONTAINER" 2>/dev/null || true; \
				chmod +x "$(ISTIOCTL_BIN)"; \
				echo "istioctl $(ISTIO_VERSION) extracted from container to $(ISTIOCTL_BIN)"; \
				echo "container" > $(ISTIOCTL_BIN).source; \
				SUCCESS=true; \
			else \
				echo "ERROR: Failed to extract istioctl binary from container"; \
				docker rm "$$TEMP_CONTAINER" 2>/dev/null || true; \
			fi; \
		else \
			echo "ERROR: Failed to pull container image $$CONTAINER_IMAGE"; \
		fi; \
	fi; \
	if [ "$$SUCCESS" = "false" ]; then \
		echo "ERROR: All download methods failed for istioctl $(ISTIO_VERSION)"; \
		echo "Tried: 1) GitHub releases, 2) Container extraction"; \
		echo "Please check if the version exists or install istioctl manually"; \
		exit 1; \
	fi
	@$(ISTIOCTL_BIN) version --remote=false


# Install Istio (generic target using ISTIO_PROFILE variable)
setup-istio: $(ISTIOCTL_BIN)
	@echo "Installing Istio version $(ISTIO_VERSION) with profile $(ISTIO_PROFILE)..."
	@if [ -n "$(ISTIO_HUB)" ]; then \
		HUB="$(ISTIO_HUB)"; \
		echo "Using specified hub: $$HUB"; \
	elif [ -f "$(ISTIOCTL_BIN).source" ]; then \
		SOURCE=$$(cat $(ISTIOCTL_BIN).source); \
		if [ "$$SOURCE" = "github" ]; then \
			HUB="gcr.io/istio-release"; \
			echo "Using hub for GitHub release: $$HUB"; \
		elif [ "$$SOURCE" = "container" ]; then \
			HUB="gcr.io/istio-testing"; \
			echo "Using hub for container image: $$HUB"; \
		else \
			HUB="gcr.io/istio-testing"; \
			echo "Unknown source, defaulting to testing hub: $$HUB"; \
		fi; \
	else \
		HUB="gcr.io/istio-release"; \
		echo "WARNING: Could not detect istioctl source, defaulting to release hub: $$HUB"; \
		echo "If this is incorrect, specify ISTIO_HUB manually"; \
	fi; \
	echo "Installing Istio from $$HUB..."; \
	$(ISTIOCTL_BIN) install -y --set profile=$(ISTIO_PROFILE) --set values.global.hub=$$HUB --set values.global.tag=$(ISTIO_VERSION)

# Install Istio for minikube environment
setup-istio-minikube:
	$(MAKE) setup-istio ISTIO_PROFILE=minimal

# Install Istio for kind environment
setup-istio-kind:
	$(MAKE) setup-istio ISTIO_PROFILE=minimal

# Install Istio for OpenShift environment
setup-istio-openshift:
	$(MAKE) setup-istio ISTIO_PROFILE=openshift

# Apply Gateway API CRDs
setup-gateway-api-crds:
	@echo "Applying Gateway API CRDs version $(GATEWAY_API_VERSION)..."
	kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/$(GATEWAY_API_VERSION)/standard-install.yaml

# Apply Inference Extension CRDs
setup-inference-extension-crds:
	@echo "Applying Inference Extension CRDs version $(INFERENCE_EXTENSION_VERSION)..."
	kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/$(INFERENCE_EXTENSION_VERSION)/config/crd/bases/inference.networking.x-k8s.io_inferencepools.yaml
	kubectl apply -f https://raw.githubusercontent.com/kubernetes-sigs/gateway-api-inference-extension/refs/tags/$(INFERENCE_EXTENSION_VERSION)/config/crd/bases/inference.networking.x-k8s.io_inferencemodels.yaml

# Apply all CRDs (convenience target)
setup-crds: setup-gateway-api-crds setup-inference-extension-crds

# Setup TLS for EPP
setup-tls:
	@echo "Setting up TLS for EPP..."
	kubectl create namespace gateway-conformance-app-backend
	$(file >/tmp/tls-destination-rules.yaml,$(TLS_DESTINATION_RULES))
	kubectl apply -f /tmp/tls-destination-rules.yaml
	@rm -f /tmp/tls-destination-rules.yaml

# Run conformance tests
run-tests:
	@echo "Running conformance tests..."
	@echo "Test base directory: $(TEST_BASE_DIR)"
	@echo "Report will be saved as: $(REPORT_BASE_DIR)/$(IMPLEMENTATION_VERSION)-$(MODE)-$(PROFILE)-report.yaml"
	@if [ -n "$(RUN_TEST)" ]; then \
		echo "Running specific test: $(RUN_TEST)"; \
	else \
		echo "Running all conformance tests"; \
	fi
	cd $(TEST_BASE_DIR) && go test ./conformance -args -gateway-class istio \
		-cleanup-base-resources=false \
		-report-output=$(shell pwd)/$(IMPLEMENTATION_VERSION)-$(MODE)-$(PROFILE)-report.yaml \
		-organization="$(ORGANIZATION)" \
		-project="$(PROJECT)" \
		-url="$(URL)" \
		-contact="$(CONTACT)" \
		-version="$(IMPLEMENTATION_VERSION)" \
		-mode="$(MODE)" \
		-run-test="$(RUN_TEST)"
	@echo ""
	@echo "Test completed! To update the README table, run:"
	@echo "  make update-readme"

# Update README table with all available reports
update-readme:
	@echo "Updating README table with all available reports..."
	@echo "Scanning for conformance reports..."
	@REPORTS=$$(ls *-*-*-report.yaml 2>/dev/null || echo ""); \
	if [ -z "$$REPORTS" ]; then \
		echo "No conformance reports found. Run 'make run-tests' to generate reports first."; \
		exit 1; \
	fi; \
	echo "Found reports: $$REPORTS"; \
	echo "Building table entries from YAML content..."; \
	TABLE_ENTRIES=""; \
	for report in $$REPORTS; do \
		echo "Processing $$report..."; \
		if [ ! -f "$$report" ]; then \
			echo "Warning: Report file $$report not found, skipping"; \
			continue; \
		fi; \
		GATEWAY_API_VERSION=$$(grep '^gatewayAPIVersion:' "$$report" | sed 's/gatewayAPIVersion: *//'); \
		IMPL_VERSION=$$(grep '^  version:' "$$report" | sed 's/  version: *//'); \
		MODE=$$(grep '^mode:' "$$report" | sed 's/mode: *//'); \
		PROFILE_NAME=$$(grep '^- name:' "$$report" | sed 's/- name: *//'); \
		DATE=$$(grep '^date:' "$$report" | sed 's/date: *"//' | sed 's/".*//'); \
		PASSED=$$(grep '^      Passed:' "$$report" | sed 's/      Passed: *//'); \
		FAILED=$$(grep '^      Failed:' "$$report" | sed 's/      Failed: *//'); \
		SKIPPED=$$(grep '^      Skipped:' "$$report" | sed 's/      Skipped: *//'); \
		RESULT=$$(grep '^    result:' "$$report" | sed 's/    result: *//'); \
		if [ -z "$$GATEWAY_API_VERSION" ]; then GATEWAY_API_VERSION="$(GATEWAY_API_VERSION)"; fi; \
		if [ -z "$$IMPL_VERSION" ]; then IMPL_VERSION="unknown"; fi; \
		if [ -z "$$MODE" ]; then MODE="default"; fi; \
		if [ -z "$$PROFILE_NAME" ]; then PROFILE_NAME="Gateway"; fi; \
		if [ -z "$$DATE" ]; then DATE="unknown"; fi; \
		if [ -z "$$PASSED" ]; then PASSED="0"; fi; \
		if [ -z "$$FAILED" ]; then FAILED="0"; fi; \
		if [ -z "$$SKIPPED" ]; then SKIPPED="0"; fi; \
		if [ -z "$$RESULT" ]; then RESULT="unknown"; fi; \
		STATUS="NOT"; \
		if [ "$$RESULT" = "success" ]; then \
			STATUS="PASS"; \
		fi; \
		TABLE_ENTRY="| $$GATEWAY_API_VERSION | $$PROFILE_NAME | $$IMPL_VERSION | $$MODE | [$$report](./$$report) | $$STATUS |"; \
		if [ -z "$$TABLE_ENTRIES" ]; then \
			TABLE_ENTRIES="$$TABLE_ENTRY"; \
		else \
			TABLE_ENTRIES="$$TABLE_ENTRIES\n$$TABLE_ENTRY"; \
		fi; \
	done; \
	echo "Updating README.md..."; \
	sed -i '/^| Extension Version Tested | Profile Tested | Implementation Version | Mode    | Report | Status |$$/,/^| \.\.\. *| \.\.\. *| \.\.\. *| \.\.\. *| \.\.\. *| \.\.\. *|$$/{//!d;}' README.md; \
	sed -i '/^| Extension Version Tested | Profile Tested | Implementation Version | Mode    | Report | Status |$$/a\|--------------------------|----------------|------------------------|---------|--------|--------|\n'"$$TABLE_ENTRIES" README.md; \
	echo "README table updated successfully!"

# Clean up resources
clean:
	@echo "Cleaning up..."
	kubectl delete namespace gateway-conformance-app-backend --ignore-not-found=true
	@echo "Cleaning up downloaded istioctl binaries..."
	@rm -f $(ISTIOCTL_DIR)/istioctl-*
	@echo "Note: If using minikube, run 'minikube delete' to completely clean up"
	@echo "Note: If using kind, run 'kind delete cluster' to completely clean up"
	@echo "Note: If using OpenShift, additional cleanup may be needed depending on your cluster setup"
	@echo "Note: Run 'make clean-reports' to clean up generated reports and README table"

# Clean up generated conformance reports
clean-reports:
	@echo "Cleaning up generated conformance reports..."
	rm -f *-*-*-report.yaml
	@echo "Cleaning up README table entries..."
	@sed -i '/^| [^E][^t].*\.yaml.*|$$/d' README.md
	@echo "Conformance reports and README table cleaned up"



 