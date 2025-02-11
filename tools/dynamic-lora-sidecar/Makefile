IMAGE_NAME := lora-syncer
IMAGE_REGISTRY ?= us-central1-docker.pkg.dev/k8s-staging-images/llm-instance-gateway
IMAGE_REPO ?= $(IMAGE_REGISTRY)/$(IMAGE_NAME)

GIT_TAG ?= $(shell git describe --tags --dirty --always)
EXTRA_TAG ?= $(if $(_PULL_BASE_REF),$(_PULL_BASE_REF),main)
IMAGE_TAG ?= $(IMAGE_REPO):$(GIT_TAG)
EXTRA_IMAGE_TAG ?= $(IMAGE_REPO):$(EXTRA_TAG)


PLATFORMS ?= linux/amd64


DOCKER_BUILDX_CMD ?= docker buildx
IMAGE_BUILD_CMD ?= $(DOCKER_BUILDX_CMD) build
IMAGE_BUILD_EXTRA_OPTS ?=

# --- Targets ---
.PHONY: image-local-build
image-local-build:
	BUILDER=$(shell $(DOCKER_BUILDX_CMD) create --use)
	$(MAKE) image-build PUSH=$(PUSH)
	$(DOCKER_BUILDX_CMD) rm $$BUILDER

.PHONY: image-local-push
image-local-push: PUSH=--push
image-local-push: image-local-build

.PHONY: image-build
image-build:
	$(IMAGE_BUILD_CMD) -t $(IMAGE_TAG) \
		--platform=$(PLATFORMS) \
		--build-arg BASE_IMAGE=$(BASE_IMAGE) \
		--build-arg BUILDER_IMAGE=$(BUILDER_IMAGE) \
		$(PUSH) \
		$(IMAGE_BUILD_EXTRA_OPTS) ./

.PHONY: image-push
image-push: PUSH=--push
image-push: image-build

.PHONY: run
run:
	docker run -v $(CURDIR)/config:/config -u appuser $(IMAGE_TAG) # Use the user name

.PHONY: clean
clean:
	docker rmi $(IMAGE_TAG) $(EXTRA_IMAGE_TAG) 2>/dev/null || true

.PHONY: clean-dangling
clean-dangling:
	docker rmi $(docker images -f "dangling=true" -q) 2>/dev/null || true

.PHONY: test
test:
	python -m unittest discover

.PHONY: all
all: test image-build