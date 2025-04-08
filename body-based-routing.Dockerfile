# Dockerfile has specific requirement to put this ARG at the beginning:
# https://docs.docker.com/engine/reference/builder/#understand-how-arg-and-from-interact
ARG BUILDER_IMAGE=golang:1.23
ARG BASE_IMAGE=gcr.io/distroless/static:nonroot

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
WORKDIR /src/cmd/bbr
RUN go build -o /body-based-routing

## Multistage deploy
FROM ${BASE_IMAGE}

WORKDIR /
COPY --from=builder /body-based-routing /body-based-routing

ENTRYPOINT ["/body-based-routing"]
