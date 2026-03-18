FROM golang:1.25 AS builder

WORKDIR /workspace

# Copy go.mod and go.sum from the root module
COPY go.mod go.sum ./
RUN go mod download

# Copy the entire source tree (needed for local module deps)
COPY . .

# Build the test binary — use package path, not file path
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -o /predictor-client-test \
    ./sidecars/latencypredictorasync/tests/

FROM alpine:3.19

RUN apk add --no-cache ca-certificates

COPY --from=builder /predictor-client-test /usr/local/bin/predictor-client-test

ENTRYPOINT ["/usr/local/bin/predictor-client-test"]