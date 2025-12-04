# Image URL to use all building/pushing image targets
IMG ?= controller:latest
# ENVTEST_K8S_VERSION refers to the version of kubebuilder assets to be downloaded by envtest binary.
ENVTEST_K8S_VERSION = 1.31.0

# Get the currently used golang install path (in GOPATH/bin, unless GOBIN is set)
ifeq (,$(shell go env GOBIN))
GOBIN=$(shell go env GOPATH)/bin
else
GOBIN=$(shell go env GOBIN)
endif

# CONTAINER_TOOL defines the container tool to be used for building images.
# Be aware that the target commands are only tested with Docker which is
# scaffolded by default. However, you might want to replace it to use other
# tools. (i.e. podman)
CONTAINER_TOOL ?= docker

# Setting SHELL to bash allows bash commands to be executed by recipes.
# Options are set to exit when a recipe line exits non-zero or a piped command fails.
SHELL = /usr/bin/env bash -o pipefail
.SHELLFLAGS = -ec

GIT_COMMIT_SHA ?= "$(shell git rev-parse HEAD 2>/dev/null)"
GIT_TAG ?= $(shell git describe --tags --dirty --always)
TARGETARCH ?= $(shell go env GOARCH)
PLATFORMS ?= linux/$(TARGETARCH)
DOCKER_BUILDX_CMD ?= docker buildx
IMAGE_BUILD_CMD ?= $(DOCKER_BUILDX_CMD) build
IMAGE_BUILD_EXTRA_OPTS ?=
SYNCER_IMAGE_BUILD_EXTRA_OPTS ?=
BBR_IMAGE_BUILD_EXTRA_OPTS ?=
STAGING_IMAGE_REGISTRY ?= us-central1-docker.pkg.dev/k8s-staging-images
IMAGE_REGISTRY ?= $(STAGING_IMAGE_REGISTRY)/gateway-api-inference-extension
IMAGE_NAME := epp
IMAGE_REPO ?= $(IMAGE_REGISTRY)/$(IMAGE_NAME)
IMAGE_TAG ?= $(IMAGE_REPO):$(GIT_TAG)
PROJECT_DIR := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))
# The path to the E2E manifest file. It can be overridden by setting the
# E2E_MANIFEST_PATH environment variable. Note that HF_TOKEN must be set when using the GPU-based manifest.
E2E_MANIFEST_PATH ?= config/manifests/vllm/sim-deployment.yaml
# E2E_IMAGE specifies the image to be used when running e2e tests using make test-e2e.
# it defaults to current image tag, but can be overwritten to test specific tags, releases, etc.
E2E_IMAGE ?= $(IMAGE_TAG)
# E2E_USE_KIND is a flag used in test-e2e target. when set to true it will load the e2e image into the kind cluster.
# it is possible though to run e2e tests against clusters other than kind. in such a case, it is the user's responsibility to load
# the image into the cluster.
E2E_USE_KIND ?= true

SYNCER_IMAGE_NAME := lora-syncer
SYNCER_IMAGE_REPO ?= $(IMAGE_REGISTRY)/$(SYNCER_IMAGE_NAME)
SYNCER_IMAGE_TAG ?= $(SYNCER_IMAGE_REPO):$(GIT_TAG)

BBR_IMAGE_NAME := bbr
BBR_IMAGE_REPO ?= $(IMAGE_REGISTRY)/$(BBR_IMAGE_NAME)
BBR_IMAGE_TAG ?= $(BBR_IMAGE_REPO):$(GIT_TAG)

BASE_IMAGE ?= gcr.io/distroless/static:nonroot
BUILDER_IMAGE ?= golang:1.24
ifdef GO_VERSION
BUILDER_IMAGE = golang:$(GO_VERSION)
endif

BUILD_REF ?= $(shell git describe --abbrev=0 2>/dev/null)
ifdef EXTRA_TAG
IMAGE_EXTRA_TAG ?= $(IMAGE_REPO):$(EXTRA_TAG)
SYNCER_IMAGE_EXTRA_TAG ?= $(SYNCER_IMAGE_REPO):$(EXTRA_TAG)
BBR_IMAGE_EXTRA_TAG ?= $(BBR_IMAGE_REPO):$(EXTRA_TAG)
BUILD_REF = $(EXTRA_TAG)
endif
ifdef IMAGE_EXTRA_TAG
IMAGE_BUILD_EXTRA_OPTS += -t $(IMAGE_EXTRA_TAG)
SYNCER_IMAGE_BUILD_EXTRA_OPTS += -t $(SYNCER_IMAGE_EXTRA_TAG)
BBR_IMAGE_BUILD_EXTRA_OPTS += -t $(BBR_IMAGE_EXTRA_TAG)
endif


# The name of the kind cluster to use for the "kind-load" target.
KIND_CLUSTER ?= kind

##@ General

# The help target prints out all targets with their descriptions organized
# beneath their categories. The categories are represented by '##@' and the
# target descriptions by '##'. The awk command is responsible for reading the
# entire set of makefiles included in this invocation, looking for lines of the
# file as xyz: ## something, and then pretty-format the target and help. Then,
# if there's a line with ##@ something, that gets pretty-printed as a category.
# More info on the usage of ANSI control characters for terminal formatting:
# https://en.wikipedia.org/wiki/ANSI_escape_code#SGR_parameters
# More info on the awk command:
# http://linuxcommand.org/lc3_adv_awk.php

.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development

.PHONY: generate
generate: controller-gen code-generator ## Generate WebhookConfiguration, ClusterRole, CustomResourceDefinition objects, code containing DeepCopy, DeepCopyInto, and DeepCopyObject method implementations.
	$(CONTROLLER_GEN) object:headerFile="hack/boilerplate/boilerplate.generatego.txt" paths="./..."
	$(CONTROLLER_GEN) crd output:dir="./config/crd/bases" paths="./..."
	./hack/update-codegen.sh

# Use same code-generator version as k8s.io/api
CODEGEN_VERSION := $(shell go list -m -f '{{.Version}}' k8s.io/api)
CODEGEN = $(shell pwd)/bin/code-generator
CODEGEN_ROOT = $(shell go env GOMODCACHE)/k8s.io/code-generator@$(CODEGEN_VERSION)
.PHONY: code-generator
code-generator:
	@GOBIN=$(PROJECT_DIR)/bin GO111MODULE=on go install k8s.io/code-generator/cmd/client-gen@$(CODEGEN_VERSION)
	cp -f $(CODEGEN_ROOT)/generate-groups.sh $(PROJECT_DIR)/bin/
	cp -f $(CODEGEN_ROOT)/generate-internal-groups.sh $(PROJECT_DIR)/bin/
	cp -f $(CODEGEN_ROOT)/kube_codegen.sh $(PROJECT_DIR)/bin/

.PHONY: fmt
fmt: ## Run go fmt against code.
	go fmt ./...

.PHONY: fmt-imports
fmt-imports: gci ## Run gci formatter on go files and verify.
	$(GCI) write -s standard -s default -s 'prefix(sigs.k8s.io/gateway-api-inference-extension)' --skip-generated --skip-vendor .

.PHONY: fmt-verify
fmt-verify:
	@out=`gofmt -w -l -d $$(find . -name '*.go')`; \
	if [ -n "$$out" ]; then \
	    echo "$$out"; \
	    exit 1; \
	fi

.PHONY: vet
vet: ## Run go vet against code.
	go vet ./...

.PHONY: test
test: generate fmt vet envtest image-build verify-crds verify-helm-charts ## Run tests.
	CGO_ENABLED=1 KUBEBUILDER_ASSETS="$(shell $(ENVTEST) use $(ENVTEST_K8S_VERSION) --bin-dir $(LOCALBIN) -p path)" go test $$(go list ./... | grep -v /e2e | grep -v /conformance) -race -coverprofile cover.out

.PHONY: test-unit
test-unit: ## Run unit tests.
	CGO_ENABLED=1 KUBEBUILDER_ASSETS="$(shell $(ENVTEST) use $(ENVTEST_K8S_VERSION) --bin-dir $(LOCALBIN) -p path)" go test ./pkg/... -race -coverprofile cover.out; \
	go tool cover -func=cover.out; \
	rm cover.out

.PHONY: test-benchmark
test-benchmark: ## Run benchmarks.
	CGO_ENABLED=1 KUBEBUILDER_ASSETS="$(shell $(ENVTEST) use $(ENVTEST_K8S_VERSION) --bin-dir $(LOCALBIN) -p path)" go test ./pkg/... -bench=. -benchmem;

.PHONY: test-integration
test-integration: envtest ## Run integration tests.
	CGO_ENABLED=1 KUBEBUILDER_ASSETS="$(shell $(ENVTEST) use $(ENVTEST_K8S_VERSION) --bin-dir $(LOCALBIN) -p path)" go test ./test/integration/epp/... -race -coverprofile cover.out

.PHONY: test-e2e
test-e2e: ## Run end-to-end tests against an existing Kubernetes cluster.
	MANIFEST_PATH=$(PROJECT_DIR)/$(E2E_MANIFEST_PATH) E2E_IMAGE=$(E2E_IMAGE) USE_KIND=$(E2E_USE_KIND) ./hack/test-e2e.sh

.PHONY: lint
lint: golangci-lint ## Run golangci-lint linter
	$(GOLANGCI_LINT) run --timeout 5m

.PHONY: lint-fix
lint-fix: golangci-lint ## Run golangci-lint linter and perform fixes
	$(GOLANGCI_LINT) run --fix

.PHONY: ci-lint
ci-lint: golangci-lint
	$(GOLANGCI_LINT) run --timeout 15m0s

.PHONY: api-lint
api-lint: golangci-api-lint
	$(GOLANGCI_API_LINT) run -c .golangci-kal.yml --timeout 15m0s ./...

.PHONY: verify
verify: vet fmt-verify generate ci-lint api-lint verify-all
	git --no-pager diff --exit-code config api client-go

.PHONY: verify-crds
verify-crds: kubectl-validate
	hack/verify-manifests.sh

.PHONY: verify-helm-charts
verify-helm-charts: helm-install
	hack/verify-helm.sh

# Run static analysis.
.PHONY: verify-all
verify-all:
	hack/verify-all.sh -v

##@ Build

# Build the container image
.PHONY: image-local-build
image-local-build: ## Build the EPP image using Docker Buildx for local development.
	BUILDER=$(shell $(DOCKER_BUILDX_CMD) create --use)
	$(MAKE) image-build PUSH=$(PUSH)
	$(MAKE) image-build LOAD=$(LOAD)
	$(DOCKER_BUILDX_CMD) rm $$BUILDER

.PHONY: image-local-push
image-local-push: PUSH=--push ## Build the EPP image for local development and push it to $IMAGE_REPO.
image-local-push: image-local-build

.PHONY: image-local-load
image-local-load: LOAD=--load ## Build the EPP image for local development and load it in the local Docker registry.
image-local-load: image-local-build

.PHONY: image-build
image-build: ## Build the EPP image using Docker Buildx.
	$(IMAGE_BUILD_CMD) -t $(IMAGE_TAG) \
		--platform=$(PLATFORMS) \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg BUILDER_IMAGE=$(BUILDER_IMAGE) \
		--build-arg COMMIT_SHA=${GIT_COMMIT_SHA} \
		--build-arg BUILD_REF=${BUILD_REF} \
		$(PUSH) \
		$(LOAD) \
		$(IMAGE_BUILD_EXTRA_OPTS) ./

.PHONY: image-push
image-push: PUSH=--push ## Build the EPP image and push it to $IMAGE_REPO.
image-push: image-build

.PHONY: image-load
image-load: LOAD=--load ## Build the EPP image and load it in the local Docker registry.
image-load: image-build

.PHONY: image-kind
image-kind: image-build ## Build the EPP image and load it to kind cluster $KIND_CLUSTER ("kind" by default).
	kind load docker-image $(IMAGE_TAG) --name $(KIND_CLUSTER)

##@ Lora Syncer

.PHONY: syncer-image-local-build
syncer-image-local-build:
	BUILDER=$(shell $(DOCKER_BUILDX_CMD) create --use)
	$(MAKE) image-build PUSH=$(PUSH)
	$(DOCKER_BUILDX_CMD) rm $$BUILDER

.PHONY: syncer-image-local-push
syncer-image-local-push: PUSH=--push
syncer-image-local-push: syncer-image-local-build

.PHONY: syncer-image-build
syncer-image-build:
	$ cd $(CURDIR)/tools/dynamic-lora-sidecar && $(IMAGE_BUILD_CMD) -t $(SYNCER_IMAGE_TAG) \
		--platform=$(PLATFORMS) \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg BUILDER_IMAGE=$(BUILDER_IMAGE) \
		$(PUSH) \
		$(SYNCER_IMAGE_BUILD_EXTRA_OPTS) ./

.PHONY: syncer-image-push
syncer-image-push: PUSH=--push
syncer-image-push: syncer-image-build

##@ Body-based Routing extension

# Build the container image
.PHONY: bbr-image-local-build
bbr-image-local-build: ## Build the image using Docker Buildx for local development.
	BUILDER=$(shell $(DOCKER_BUILDX_CMD) create --use)
	$(MAKE) bbr-image-build PUSH=$(PUSH)
	$(MAKE) bbr-image-build LOAD=$(LOAD)
	$(DOCKER_BUILDX_CMD) rm $$BUILDER

.PHONY: bbr-image-local-push
bbr-image-local-push: PUSH=--push ## Build the image for local development and push it to $IMAGE_REPO.
bbr-image-local-push: bbr-image-local-build

.PHONY: bbr-image-local-load
bbr-image-local-load: LOAD=--load ## Build the image for local development and load it in the local Docker registry.
bbr-image-local-load: bbr-image-local-build

.PHONY: bbr-image-build
bbr-image-build: ## Build the image using Docker Buildx.
	$(IMAGE_BUILD_CMD) -f bbr.Dockerfile -t $(BBR_IMAGE_TAG) \
		--platform=$(PLATFORMS) \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg BUILDER_IMAGE=$(BUILDER_IMAGE) \
		$(PUSH) \
		$(LOAD) \
		$(BBR_IMAGE_BUILD_EXTRA_OPTS) ./

.PHONY: bbr-image-push
bbr-image-push: PUSH=--push ## Build the image and push it to $IMAGE_REPO.
bbr-image-push: bbr-image-build

.PHONY: bbr-image-load
bbr-image-load: LOAD=--load ## Build the image and load it in the local Docker registry.
bbr-image-load: bbr-image-build

.PHONY: bbr-image-kind
bbr-image-kind: bbr-image-build ## Build the image and load it to kind cluster $KIND_CLUSTER ("kind" by default).
	kind load docker-image $(BBR_IMAGE_TAG) --name $(KIND_CLUSTER)

##@ Docs

.PHONY: build-docs
build-docs:
	docker build --pull -t gaie/mkdocs hack/mkdocs/image
	docker run --rm -v ${PWD}:/docs gaie/mkdocs build

.PHONY: build-docs-netlify
build-docs-netlify:
	pip install -r hack/mkdocs/image/requirements.txt
	python -m mkdocs build

.PHONY: live-docs
live-docs:
	docker build -t gaie/mkdocs hack/mkdocs/image
	docker run --rm -it -p 3000:3000 -v ${PWD}:/docs gaie/mkdocs

.PHONY: api-ref-docs-all
api-ref-docs-all: apix-v1a1-ref-docs apix-v1a2-ref-docs api-ref-docs

.PHONY: apix-v1a1-ref-docs
apix-v1a1-ref-docs: crd-ref-docs
	${CRD_REF_DOCS} \
		--source-path=${PWD}/apix/v1alpha1 \
		--config=crd-ref-docs.yaml \
		--renderer=markdown \
		--output-path=${PWD}/site-src/reference/x-v1a1-spec.md

.PHONY: apix-v1a2-ref-docs
apix-v1a2-ref-docs: crd-ref-docs
	${CRD_REF_DOCS} \
		--source-path=${PWD}/apix/v1alpha2 \
		--config=crd-ref-docs.yaml \
		--renderer=markdown \
		--output-path=${PWD}/site-src/reference/x-v1a2-spec.md

.PHONY: api-ref-docs
api-ref-docs: crd-ref-docs
	${CRD_REF_DOCS} \
		--source-path=${PWD}/api \
		--config=crd-ref-docs.yaml \
		--renderer=markdown \
		--output-path=${PWD}/site-src/reference/spec.md

##@ Deployment

ifndef ignore-not-found
  ignore-not-found = false
endif

.PHONY: install
install: generate kustomize ## Install CRDs into the K8s cluster specified in ~/.kube/config.
	$(KUSTOMIZE) build config/crd | $(KUBECTL) apply -f -

.PHONY: uninstall
uninstall: generate kustomize ## Uninstall CRDs from the K8s cluster specified in ~/.kube/config. Call with ignore-not-found=true to ignore resource not found errors during deletion.
	$(KUSTOMIZE) build config/crd | $(KUBECTL) delete --ignore-not-found=$(ignore-not-found) -f -

##@ Helm
.PHONY: inferencepool-helm-chart-push
inferencepool-helm-chart-push: yq helm-install
	CHART=inferencepool EXTRA_TAG="$(EXTRA_TAG)" IMAGE_REGISTRY="$(IMAGE_REGISTRY)" YQ="$(YQ)" HELM="$(HELM)" ./hack/push-chart.sh

.PHONY: bbr-helm-chart-push
bbr-helm-chart-push: yq helm-install
	CHART=body-based-routing EXTRA_TAG="$(EXTRA_TAG)" IMAGE_REGISTRY="$(IMAGE_REGISTRY)" YQ="$(YQ)" HELM="$(HELM)" ./hack/push-chart.sh

##@ Release

.PHONY: release-quickstart
release-quickstart: ## Update the quickstart guide for a release.
	./hack/release-quickstart.sh

.PHONY: artifacts
artifacts: kustomize yq
	if [ -d artifacts ]; then rm -rf artifacts; fi
	mkdir -p artifacts
	$(KUSTOMIZE) build config/crd -o artifacts/manifests.yaml
	$(YQ) -P 'select(.spec.versions[].name == "v1")' artifacts/manifests.yaml > artifacts/v1-manifests.yaml
	$(YQ) -P 'select(.spec.versions[].name != "v1")' artifacts/manifests.yaml > artifacts/experimental-manifests.yaml
	@$(call clean-manifests)

.PHONY: release
release: artifacts release-quickstart verify test # Create a release.

##@ Dependencies

## Location to install dependencies to
LOCALBIN ?= $(shell pwd)/bin
$(LOCALBIN):
	[ -d $@ ] || mkdir -p $@

## Tool Binaries
KUBECTL ?= kubectl
KUSTOMIZE ?= $(LOCALBIN)/kustomize
CONTROLLER_GEN ?= $(LOCALBIN)/controller-gen
ENVTEST ?= $(LOCALBIN)/setup-envtest
CRD_REF_DOCS ?= $(LOCALBIN)/crd-ref-docs
GOLANGCI_LINT = $(LOCALBIN)/golangci-lint
GOLANGCI_API_LINT = $(LOCALBIN)/golangci-kube-api-linter
HELM = $(PROJECT_DIR)/bin/helm
YQ = $(PROJECT_DIR)/bin/yq
KUBECTL_VALIDATE = $(PROJECT_DIR)/bin/kubectl-validate
GCI = $(LOCALBIN)/gci

## Tool Versions
KUSTOMIZE_VERSION ?= v5.4.3
CONTROLLER_TOOLS_VERSION ?= v0.19.0
ENVTEST_VERSION ?= release-0.19
CRD_REF_DOCS_VERSION ?= v0.2.0
GOLANGCI_LINT_VERSION ?= v2.7.0
HELM_VERSION ?= v3.17.1
KUBECTL_VALIDATE_VERSION ?= v0.0.4
GCI_VERSION ?= v0.13.6
YQ_VERSION ?= v4.45.1

.PHONY: kustomize
kustomize: $(KUSTOMIZE) ## Download kustomize locally if necessary.
$(KUSTOMIZE): $(LOCALBIN)
	$(call go-install-tool,$(KUSTOMIZE),sigs.k8s.io/kustomize/kustomize/v5,$(KUSTOMIZE_VERSION))

.PHONY: controller-gen
controller-gen: $(CONTROLLER_GEN) ## Download controller-gen locally if necessary.
$(CONTROLLER_GEN): $(LOCALBIN)
	$(call go-install-tool,$(CONTROLLER_GEN),sigs.k8s.io/controller-tools/cmd/controller-gen,$(CONTROLLER_TOOLS_VERSION))

.PHONY: crd-ref-docs
crd-ref-docs: $(CRD_REF_DOCS) ## Download crd-ref-docs locally if necessary.
$(CRD_REF_DOCS): $(LOCALBIN)
	$(call go-install-tool,$(CRD_REF_DOCS),github.com/elastic/crd-ref-docs,$(CRD_REF_DOCS_VERSION))

.PHONY: envtest
envtest: $(ENVTEST) ## Download setup-envtest locally if necessary.
$(ENVTEST): $(LOCALBIN)
	$(call go-install-tool,$(ENVTEST),sigs.k8s.io/controller-runtime/tools/setup-envtest,$(ENVTEST_VERSION))

.PHONY: golangci-lint
golangci-lint: $(GOLANGCI_LINT) ## Download golangci-lint locally if necessary.
$(GOLANGCI_LINT): $(LOCALBIN)
	$(call go-install-tool,$(GOLANGCI_LINT),github.com/golangci/golangci-lint/v2/cmd/golangci-lint,$(GOLANGCI_LINT_VERSION))

.PHONY: golangci-api-lint
golangci-api-lint: golangci-lint $(GOLANGCI_API_LINT) ## Download golangci-lint locally if necessary before building KAL
$(GOLANGCI_API_LINT):
	$(GOLANGCI_LINT) custom

.PHONY: yq
yq: ## Download yq locally if necessary.
	GOBIN=$(PROJECT_DIR)/bin GO111MODULE=on go install github.com/mikefarah/yq/v4@$(YQ_VERSION)

.PHONY: helm-install
helm-install: $(HELM) ## Download helm locally if necessary.
$(HELM): $(LOCALBIN)
	GOBIN=$(PROJECT_DIR)/bin GO111MODULE=on go install helm.sh/helm/v3/cmd/helm@$(HELM_VERSION)

.PHONY: kubectl-validate
kubectl-validate: $(KUBECTL_VALIDATE) ## Download kubectl-validate locally if necessary.
$(KUBECTL_VALIDATE): $(LOCALBIN)
	$(call go-install-tool,$(KUBECTL_VALIDATE),sigs.k8s.io/kubectl-validate,$(KUBECTL_VALIDATE_VERSION))

.PHONY: gci
gci: $(GCI) ## Download gci locally if necessary.
$(GCI): $(LOCALBIN)
	$(call go-install-tool,$(GCI),github.com/daixiang0/gci,$(GCI_VERSION))

# go-install-tool will 'go install' any package with custom target and name of binary, if it doesn't exist
# $1 - target path with name of binary
# $2 - package url which can be installed
# $3 - specific version of package
define go-install-tool
@[ -f "$(1)-$(3)" ] || { \
set -e; \
package=$(2)@$(3) ;\
echo "Downloading $${package}" ;\
rm -f $(1) || true ;\
GOBIN=$(LOCALBIN) go install $${package} ;\
mv $(1) $(1)-$(3) ;\
} ;\
ln -sf $(1)-$(3) $(1)
endef
