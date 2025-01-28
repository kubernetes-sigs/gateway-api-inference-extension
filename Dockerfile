# Dockerfile has specific requirement to put this ARG at the beginning:
# https://docs.docker.com/engine/reference/builder/#understand-how-arg-and-from-interact
ARG BUILDER_IMAGE=golang:1.23-alpine
ARG BASE_IMAGE=gcr.io/distroless/base-debian10

## Multistage build
FROM ${BUILDER_IMAGE} as builder
ENV CGO_ENABLED=0
ENV GOOS=linux
ENV GOARCH=amd64

WORKDIR /src
COPY . .
WORKDIR /src/pkg/ext-proc
RUN go mod download
RUN go build -o /ext-proc

## Multistage deploy
FROM ${BASE_IMAGE}

WORKDIR /
COPY --from=builder /ext-proc /ext-proc

ENTRYPOINT ["/ext-proc"]