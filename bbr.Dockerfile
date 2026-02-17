# Dockerfile has specific requirement to put this ARG at the beginning:
# https://docs.docker.com/engine/reference/builder/#understand-how-arg-and-from-interact
ARG BUILDER_IMAGE=golang:1.25
ARG BASE_IMAGE=gcr.io/distroless/static:nonroot
ARG BUILDPLATFORM

## Multistage build
FROM --platform=$BUILDPLATFORM ${BUILDER_IMAGE} AS builder
ARG TARGETARCH
ENV CGO_ENABLED=0
ENV GOOS=linux
ENV GOARCH=${TARGETARCH}
ARG COMMIT_SHA=unknown
ARG BUILD_REF

# Dependencies
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download

# Sources
COPY cmd/bbr ./cmd/bbr
COPY pkg ./pkg
COPY internal ./internal
COPY api ./api
COPY version ./version
WORKDIR /src/cmd/bbr
RUN go build -ldflags="-X sigs.k8s.io/gateway-api-inference-extension/version.CommitSHA=${COMMIT_SHA} -X sigs.k8s.io/gateway-api-inference-extension/version.BuildRef=${BUILD_REF}" -o /bbr

## Multistage deploy
FROM ${BASE_IMAGE}

WORKDIR /
COPY --from=builder /bbr /bbr

ENTRYPOINT ["/bbr"]
