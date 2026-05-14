# Conformance

Similar to Gateway API, this project will rely on conformance tests to ensure
compatibility across implementations. This will be focused on three different
layers:

## Conformance Report Deprecation Policy

Conformance is versioned by Gateway API Inference Extension release and profile.
An implementation is listed as conformant only when it has a successful
conformance report for the relevant profile in
[`conformance/reports`](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/conformance/reports).

This policy follows the
[Kubernetes deprecation policy](https://kubernetes.io/docs/reference/using-api/deprecation-policy/)
practice of publicly documenting user-visible removals, and the Gateway API and
[CNCF Kubernetes conformance](https://www.cncf.io/certification/software-conformance/)
model of basing conformance status on submitted, reproducible test artifacts.

Implementations should submit a successful conformance report for each minor
release. To avoid removing entries for short release gaps, this project accepts
reports from the current minor release or either of the two previous minor
releases. Patch releases do not change this window. For example, when `v1.4.x`
is the current release, reports for `v1.4.x`, `v1.3.x`, and `v1.2.x` are
current; reports for `v1.1.x` and older are stale.

This policy applies independently to each implementation and conformance
profile. For example, a gateway implementation can remain listed as conformant
for the Gateway profile only while it has an accepted Gateway conformance
report.

Implementation status follows this lifecycle:

* **Current**: The implementation has a successful report for the current minor
  release or the previous minor release.
* **Update recommended**: The implementation's newest successful report is from
  the second previous minor release. The report is still accepted, but
  maintainers may ask the implementation contacts to submit a new report before
  the next minor release.
* **Stale**: The implementation's newest successful report is older than the
  second previous minor release. Maintainers may remove stale implementations
  from conformant implementation listings until they submit an accepted report.

Before removing a stale implementation, maintainers should make a reasonable
effort to notify the contacts listed in the implementation's conformance report
or report README. Removal should happen through a normal pull request so the
project has a public record of the change. A removed implementation can be
restored by submitting a successful report for an accepted release.

Exceptions should be rare, discussed publicly, and documented in the pull
request or issue that keeps the implementation listed. Exceptions should include
the affected implementation, profile, release window, and expected update path.

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
