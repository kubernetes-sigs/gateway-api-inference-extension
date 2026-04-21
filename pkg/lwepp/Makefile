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
CONTAINER_TOOL ?= docker

# Setting SHELL to bash allows bash commands to be executed by recipes.
SHELL = /usr/bin/env bash -o pipefail
.SHELLFLAGS = -ec

GIT_COMMIT_SHA ?= "$(shell git rev-parse HEAD 2>/dev/null)"
ROOT_RELEASE_TAG_MATCH ?= v[0-9]*
GIT_TAG ?= $(shell git describe --tags --match '$(ROOT_RELEASE_TAG_MATCH)' --dirty --always)
TARGETARCH ?= $(shell go env GOARCH)
PLATFORMS ?= linux/$(TARGETARCH)
DOCKER_BUILDX_CMD ?= docker buildx
IMAGE_BUILD_CMD ?= $(DOCKER_BUILDX_CMD) build
LWEPP_IMAGE_BUILD_EXTRA_OPTS ?=
STAGING_IMAGE_REGISTRY ?= us-central1-docker.pkg.dev/k8s-staging-images
IMAGE_REGISTRY ?= $(STAGING_IMAGE_REGISTRY)/gateway-api-inference-extension

LWEPP_IMAGE_NAME := lwepp
LWEPP_IMAGE_REPO ?= $(IMAGE_REGISTRY)/$(LWEPP_IMAGE_NAME)
LWEPP_IMAGE_TAG ?= $(LWEPP_IMAGE_REPO):$(GIT_TAG)

BASE_IMAGE ?= gcr.io/distroless/static:nonroot
BUILDER_IMAGE ?= golang:1.25
ifdef GO_VERSION
BUILDER_IMAGE = golang:$(GO_VERSION)
endif

BUILD_REF ?= $(shell git describe --tags --match '$(ROOT_RELEASE_TAG_MATCH)' --abbrev=0 2>/dev/null)
ifdef EXTRA_TAG
LWEPP_IMAGE_EXTRA_TAG ?= $(LWEPP_IMAGE_REPO):$(EXTRA_TAG)
BUILD_REF = $(EXTRA_TAG)
endif
ifdef LWEPP_IMAGE_EXTRA_TAG
LWEPP_IMAGE_BUILD_EXTRA_OPTS += -t $(LWEPP_IMAGE_EXTRA_TAG)
endif

PROJECT_DIR := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))

## Location to install dependencies to
LOCALBIN ?= $(PROJECT_DIR)/../../bin
$(LOCALBIN):
	[ -d $@ ] || mkdir -p $@

## Tool Binaries
KUBECTL ?= kubectl
KUSTOMIZE ?= $(LOCALBIN)/kustomize
CONTROLLER_GEN ?= $(LOCALBIN)/controller-gen
CODEGEN = $(LOCALBIN)/code-generator

## Tool Versions
KUSTOMIZE_VERSION ?= v5.4.3
CONTROLLER_TOOLS_VERSION ?= v0.19.0

.PHONY: help
help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

.PHONY: all
all: help

##@ Lightweight EPP

.PHONY: lwepp-image-local-build
lwepp-image-local-build: ## Build the image using Docker Buildx for local development.
	BUILDER=$(shell $(DOCKER_BUILDX_CMD) create --use)
	$(MAKE) lwepp-image-build PUSH=$(PUSH)
	$(MAKE) lwepp-image-build LOAD=$(LOAD)
	$(DOCKER_BUILDX_CMD) rm $$BUILDER

.PHONY: lwepp-image-local-push
lwepp-image-local-push: PUSH=--push ## Build the image for local development and push it to $IMAGE_REPO.
lwepp-image-local-push: lwepp-image-local-build

.PHONY: lwepp-image-local-load
lwepp-image-local-load: LOAD=--load ## Build the image for local development and load it in the local Docker registry.
lwepp-image-local-load: lwepp-image-local-build

.PHONY: lwepp-image-build
lwepp-image-build: ## Build the image using Docker Buildx.
	$(IMAGE_BUILD_CMD) -f Dockerfile -t $(LWEPP_IMAGE_TAG) \
		--platform=$(PLATFORMS) \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg BUILDER_IMAGE=$(BUILDER_IMAGE) \
		--build-arg COMMIT_SHA=${GIT_COMMIT_SHA} \
		--build-arg BUILD_REF=${BUILD_REF} \
		$(PUSH) \
		$(LOAD) \
		$(LWEPP_IMAGE_BUILD_EXTRA_OPTS) ../../

.PHONY: lwepp-image-push
lwepp-image-push: PUSH=--push ## Build the image and push it to $IMAGE_REPO.
lwepp-image-push: lwepp-image-build

.PHONY: lwepp-image-load
lwepp-image-load: LOAD=--load ## Build the image and load it in the local Docker registry.
lwepp-image-load: lwepp-image-build

.PHONY: lwepp-image-kind
lwepp-image-kind: lwepp-image-build ## Build the image and load it to kind cluster $KIND_CLUSTER ("kind" by default).
	kind load docker-image $(LWEPP_IMAGE_TAG) --name $(KIND_CLUSTER)

##@ Deployment

ifndef ignore-not-found
  ignore-not-found = false
endif

.PHONY: install
install: generate kustomize ## Install CRDs into the K8s cluster specified in ~/.kube/config.
	$(KUSTOMIZE) build ../../config/crd | $(KUBECTL) apply -f -

.PHONY: uninstall
uninstall: generate kustomize ## Uninstall CRDs from the K8s cluster specified in ~/.kube/config. Call with ignore-not-found=true to ignore resource not found errors during deletion.
	$(KUSTOMIZE) build ../../config/crd | $(KUBECTL) delete --ignore-not-found=$(ignore-not-found) -f -

##@ Development

.PHONY: generate
generate: controller-gen code-generator tidy ## Generate WebhookConfiguration, ClusterRole, CustomResourceDefinition objects, code containing DeepCopy, DeepCopyInto, and DeepCopyObject method implementations.
	$(CONTROLLER_GEN) object:headerFile="../../hack/boilerplate/boilerplate.generatego.txt" paths="../../..."
	$(CONTROLLER_GEN) crd output:dir="../../config/crd/bases" paths="../../..."
	cd ../../ && ./hack/update-codegen.sh

# Use same code-generator version as k8s.io/api
CODEGEN_VERSION := $(shell go list -m -f '{{.Version}}' k8s.io/api)
CODEGEN_ROOT = $(shell go env GOMODCACHE)/k8s.io/code-generator@$(CODEGEN_VERSION)
.PHONY: code-generator
code-generator:
	@GOBIN=$(PROJECT_DIR)/../../bin GO111MODULE=on go install k8s.io/code-generator/cmd/client-gen@$(CODEGEN_VERSION)
	cp -f $(CODEGEN_ROOT)/generate-groups.sh $(PROJECT_DIR)/../../bin/
	cp -f $(CODEGEN_ROOT)/generate-internal-groups.sh $(PROJECT_DIR)/../../bin/
	cp -f $(CODEGEN_ROOT)/kube_codegen.sh $(PROJECT_DIR)/../../bin/

.PHONY: tidy
tidy:
	cd ../../ && go work sync
	cd ../../ && find . -name go.mod -execdir sh -c 'go mod tidy' \;

##@ Dependencies

.PHONY: kustomize
kustomize: $(KUSTOMIZE) ## Download kustomize locally if necessary.
$(KUSTOMIZE): $(LOCALBIN)
	$(call go-install-tool,$(KUSTOMIZE),sigs.k8s.io/kustomize/kustomize/v5,$(KUSTOMIZE_VERSION))

.PHONY: controller-gen
controller-gen: $(CONTROLLER_GEN) ## Download controller-gen locally if necessary.
$(CONTROLLER_GEN): $(LOCALBIN)
	$(call go-install-tool,$(CONTROLLER_GEN),sigs.k8s.io/controller-tools/cmd/controller-gen,$(CONTROLLER_TOOLS_VERSION))

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
