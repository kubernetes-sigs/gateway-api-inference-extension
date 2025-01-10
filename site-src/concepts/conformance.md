# Conformance

Similar to Gateway API, this project will rely on conformance tests to ensure
compatibility across implementations. This will be focused on three different
layers:

## 1. Gateway API Implementations

Conformance tests will verify that:

* InferencePool is supported as a backend type
* Implementations forward requests to the configured extension for an
  InferencePool following the specification defined by this project
* Implementations honor the routing guidance provided by the extension
* Implementations behave appropriately when an extension is either not present
  or fails to respond

## 2. Inference Routing Extensions

Conformance tests will verify that:

* Extensions accept requests that match the protocol specified by this project
* Extensions respond with routing guidance that matches the protocol specified
  by this project

## 3. Model Server Frameworks

Conformance tests will verify that:

* Frameworks serve the expected set of metrics using a format and path specified
  by this project
