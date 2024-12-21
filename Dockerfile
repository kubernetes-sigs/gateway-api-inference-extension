## Multistage build
FROM golang:1.23-alpine AS build
ENV CGO_ENABLED=0
ENV GOOS=linux
ENV GOARCH=amd64

WORKDIR /src
COPY . .
WORKDIR /src/pkg/ext-proc
RUN go mod download
RUN go build -o /ext-proc
FROM alpine:latest
## Multistage deploy
FROM gcr.io/distroless/base-debian10
# Install bash

WORKDIR /
COPY --from=build /ext-proc /ext-proc

ENTRYPOINT ["/ext-proc"]