# Dockerfile has specific requirement to put this ARG at the beginning:
# https://docs.docker.com/engine/reference/builder/#understand-how-arg-and-from-interact
ARG BUILDER_IMAGE=golang:1.24
ARG BASE_IMAGE=gcr.io/distroless/static:nonroot

## Multistage build
FROM ${BUILDER_IMAGE} AS builder
ENV CGO_ENABLED=0
ENV GOOS=linux
ENV GOARCH=amd64
ARG COMMIT_SHA=unknown

# Dependencies
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download

# Sources
COPY cmd/epp ./cmd
COPY pkg/epp ./pkg/epp
COPY internal ./internal
COPY api ./api
WORKDIR /src/cmd
RUN go build -ldflags="-X sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics.CommitSHA=${COMMIT_SHA}" -o /epp

## Multistage deploy
FROM ${BASE_IMAGE}

WORKDIR /
COPY --from=builder /epp /epp

ENTRYPOINT ["/epp"]
