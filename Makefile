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

GIT_TAG ?= $(shell git describe --tags --dirty --always)
PLATFORMS ?= linux/amd64
DOCKER_BUILDX_CMD ?= docker buildx
IMAGE_BUILD_CMD ?= $(DOCKER_BUILDX_CMD) build
IMAGE_BUILD_EXTRA_OPTS ?=
SYNCER_IMAGE_BUILD_EXTRA_OPTS ?=
BBR_IMAGE_BUILD_EXTRA_OPTS ?=
IMAGE_REGISTRY ?= us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension
IMAGE_NAME := epp
IMAGE_REPO ?= $(IMAGE_REGISTRY)/$(IMAGE_NAME)
IMAGE_TAG ?= $(IMAGE_REPO):$(GIT_TAG)

SYNCER_IMAGE_NAME := lora-syncer
SYNCER_IMAGE_REPO ?= $(IMAGE_REGISTRY)/$(SYNCER_IMAGE_NAME)
SYNCER_IMAGE_TAG ?= $(SYNCER_IMAGE_REPO):$(GIT_TAG)

BBR_IMAGE_NAME := bbr
BBR_IMAGE_REPO ?= $(IMAGE_REGISTRY)/$(BBR_IMAGE_NAME)
BBR_IMAGE_TAG ?= $(BBR_IMAGE_REPO):$(GIT_TAG)

BASE_IMAGE ?= gcr.io/distroless/static:nonroot
BUILDER_IMAGE ?= golang:1.23
ifdef GO_VERSION
BUILDER_IMAGE = golang:$(GO_VERSION)
endif

ifdef EXTRA_TAG
IMAGE_EXTRA_TAG ?= $(IMAGE_REPO):$(EXTRA_TAG)
SYNCER_IMAGE_EXTRA_TAG ?= $(SYNCER_IMAGE_REPO):$(EXTRA_TAG)
BBR_IMAGE_EXTRA_TAG ?= $(BBR_IMAGE_REPO):$(EXTRA_TAG)
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

.PHONY: manifests
manifests: controller-gen ## Generate WebhookConfiguration, ClusterRole and CustomResourceDefinition objects.
	$(CONTROLLER_GEN) rbac:roleName=manager-role crd webhook paths="./..." output:crd:artifacts:config=config/crd/bases

.PHONY: generate
generate: controller-gen code-generator manifests ## Generate code containing DeepCopy, DeepCopyInto, and DeepCopyObject method implementations.
	$(CONTROLLER_GEN) object:headerFile="hack/boilerplate.go.txt" paths="./..."
	./hack/update-codegen.sh

PROJECT_DIR := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))
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
test: manifests generate fmt vet envtest ## Run tests.
	KUBEBUILDER_ASSETS="$(shell $(ENVTEST) use $(ENVTEST_K8S_VERSION) --bin-dir $(LOCALBIN) -p path)" go test $$(go list ./... | grep -v /e2e) -race -coverprofile cover.out

.PHONY: test-integration
test-integration: manifests generate fmt vet envtest ## Run tests.
	KUBEBUILDER_ASSETS="$(shell $(ENVTEST) use $(ENVTEST_K8S_VERSION) --bin-dir $(LOCALBIN) -p path)" go test ./test/integration -coverprofile cover.out

.PHONY: test-e2e
test-e2e: ## Run end-to-end tests against an existing Kubernetes cluster with at least 3 available GPUs.
	go test ./test/e2e/ -v -ginkgo.v

.PHONY: lint
lint: golangci-lint ## Run golangci-lint linter
	$(GOLANGCI_LINT) run

.PHONY: lint-fix
lint-fix: golangci-lint ## Run golangci-lint linter and perform fixes
	$(GOLANGCI_LINT) run --fix

.PHONY: ci-lint
ci-lint: golangci-lint
	$(GOLANGCI_LINT) run --timeout 15m0s

.PHONY: verify
verify: vet fmt-verify manifests generate ci-lint
	git --no-pager diff --exit-code config api client-go

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
	$(IMAGE_BUILD_CMD) -f body-based-routing.Dockerfile -t $(BBR_IMAGE_TAG) \
		--platform=$(PLATFORMS) \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg BUILDER_IMAGE=$(BUILDER_IMAGE) \
		$(PUSH) \
		$(LOAD) \
		.

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

.PHONY: api-ref-docs
api-ref-docs:
	crd-ref-docs \
		--source-path=${PWD}/api \
		--config=crd-ref-docs.yaml \
		--renderer=markdown \
		--output-path=${PWD}/site-src/reference/spec.md

##@ Deployment

ifndef ignore-not-found
  ignore-not-found = false
endif

.PHONY: install
install: manifests kustomize ## Install CRDs into the K8s cluster specified in ~/.kube/config.
	$(KUSTOMIZE) build config/crd | $(KUBECTL) apply -f -

.PHONY: uninstall
uninstall: manifests kustomize ## Uninstall CRDs from the K8s cluster specified in ~/.kube/config. Call with ignore-not-found=true to ignore resource not found errors during deletion.
	$(KUSTOMIZE) build config/crd | $(KUBECTL) delete --ignore-not-found=$(ignore-not-found) -f -

##@ Release

.PHONY: release-quickstart
release-quickstart: ## Update the quickstart guide for a release.
	./hack/release-quickstart.sh

.PHONY: artifacts
artifacts: kustomize
	if [ -d artifacts ]; then rm -rf artifacts; fi
	mkdir -p artifacts
	$(KUSTOMIZE) build config/crd -o artifacts/manifests.yaml
	@$(call clean-manifests)

.PHONY: release
release: artifacts release-quickstart verify test # Create a release.

##@ Dependencies

## Location to install dependencies to
LOCALBIN ?= $(shell pwd)/bin
$(LOCALBIN):
	mkdir -p $(LOCALBIN)

## Tool Binaries
KUBECTL ?= kubectl
KUSTOMIZE ?= $(LOCALBIN)/kustomize
CONTROLLER_GEN ?= $(LOCALBIN)/controller-gen
ENVTEST ?= $(LOCALBIN)/setup-envtest
GOLANGCI_LINT = $(LOCALBIN)/golangci-lint

## Tool Versions
KUSTOMIZE_VERSION ?= v5.4.3
CONTROLLER_TOOLS_VERSION ?= v0.16.1
ENVTEST_VERSION ?= release-0.19
GOLANGCI_LINT_VERSION ?= v1.62.2

.PHONY: kustomize
kustomize: $(KUSTOMIZE) ## Download kustomize locally if necessary.
$(KUSTOMIZE): $(LOCALBIN)
	$(call go-install-tool,$(KUSTOMIZE),sigs.k8s.io/kustomize/kustomize/v5,$(KUSTOMIZE_VERSION))

.PHONY: controller-gen
controller-gen: $(CONTROLLER_GEN) ## Download controller-gen locally if necessary.
$(CONTROLLER_GEN): $(LOCALBIN)
	$(call go-install-tool,$(CONTROLLER_GEN),sigs.k8s.io/controller-tools/cmd/controller-gen,$(CONTROLLER_TOOLS_VERSION))

.PHONY: envtest
envtest: $(ENVTEST) ## Download setup-envtest locally if necessary.
$(ENVTEST): $(LOCALBIN)
	$(call go-install-tool,$(ENVTEST),sigs.k8s.io/controller-runtime/tools/setup-envtest,$(ENVTEST_VERSION))

.PHONY: golangci-lint
golangci-lint: $(GOLANGCI_LINT) ## Download golangci-lint locally if necessary.
$(GOLANGCI_LINT): $(LOCALBIN)
	$(call go-install-tool,$(GOLANGCI_LINT),github.com/golangci/golangci-lint/cmd/golangci-lint,$(GOLANGCI_LINT_VERSION))

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
