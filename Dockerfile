# Dockerfile has specific requirement to put this ARG at the beginning:
# https://docs.docker.com/engine/reference/builder/#understand-how-arg-and-from-interact
ARG BUILDER_IMAGE=golang:1.23-alpine
ARG BASE_IMAGE=gcr.io/distroless/base-debian10

## Multistage build
FROM ${BUILDER_IMAGE} AS builder
ENV CGO_ENABLED=0
ENV GOOS=linux
ENV GOARCH=amd64

# Dependencies
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download

# Sources
COPY cmd ./cmd
COPY pkg ./pkg
COPY internal ./internal
COPY api ./api
WORKDIR /src/cmd/epp
RUN go build -o /epp

## Multistage deploy
FROM ${BASE_IMAGE}

WORKDIR /
COPY --from=builder /epp /epp

ENTRYPOINT ["/epp"]
