# InferencePool port role specification

Author(s): @UgaTheDev

Issue: #2857

## Motivation

`InferencePoolSpec.TargetPorts` is a flat list of `Port{Number}` entries. EPP and other
consumers treat every entry as an equivalent inference endpoint (`podIP:portNumber`). There is
no way to tell the API that a given port serves a different function — metrics scraping or
health/liveness probing — from the ports that actually carry inference traffic.

This gap is structural, not incidental. #1396 and PR #2762 describe a concrete failure mode: in
Istio **mTLS STRICT** environments, EPP scrapes pod metrics by dialing the pod IP directly,
bypassing the Service and its sidecar-aware routing. The sidecar rejects the connection with 503
because it expects mTLS on that port and receives plain HTTP. PR #2762 worked around this by
adding a `metricsPort` field to the `metrics-data-source` **plugin** configuration, but that only
fixes metrics scraping for one plugin's configuration surface. It doesn't help conformance, other
data sources, health checking, or any future consumer that needs a "which port is X for" answer —
each would need its own bespoke config field. The InferencePool API is the natural place to
answer that question once, for every consumer.

## Goals

* Let an InferencePool author designate which of its `targetPorts` serve inference traffic
  ("Serving"), metrics scraping ("Metrics"), and health/liveness probing ("Health").
* Preserve full backward compatibility: an InferencePool written against the current API (no role
  field at all) must continue to validate and behave identically.
* Keep the change additive and small enough to land as a single PR behind this proposal, rather
  than a multi-phase effort.

## Non-Goals

* Rewiring the full production EPP (`pkg/epp/...`)'s metrics/health scraping to consume this
  field. That is real follow-up work once the API shape lands, tracked separately — this proposal
  only changes the API and the parts of the reference EPP implementation (`pkg/lwepp`) that build
  the traffic-endpoint list directly from `targetPorts`.
* Superseding or removing the `metricsPort` field added to the `metrics-data-source` plugin by PR
  #2762. That field is a plugin-scoped override; this proposal is the spec-scoped generalization
  the PR description explicitly called out as unaddressed.
* A generic "labels on ports" mechanism. Three well-known roles are the concrete, requested need;
  an open-ended label/annotation scheme is speculative beyond that.

## Proposed API Changes

Add a `Role` field to the existing `Port` struct, defaulting to `"Serving"` so that every
InferencePool written before this change is unaffected — an absent `role` is indistinguishable
from `role: Serving` at admission time (CRD structural-schema defaulting fills it in before
validation runs).

```go
// Port defines the network port that will be exposed by this InferencePool.
type Port struct {
    // Number defines the port number to access the selected model server Pods.
    // The number must be in the range 1 to 65535.
    //
    // +required
    Number PortNumber `json:"number,omitempty"`

    // Role designates the function this port serves for the InferencePool.
    //
    // Supported values include:
    // * "Serving": the port used for inference traffic. This is the default. At least one
    //   targetPort must have this role.
    // * "Metrics": the port model server metrics are scraped from. Consumers that scrape
    //   metrics should prefer a port with this role and fall back to the "Serving" port
    //   when none is present.
    // * "Health": the port used for liveness/readiness probing. Consumers that perform health
    //   checks should prefer a port with this role and fall back to the "Serving" port when
    //   none is present.
    //
    // Multiple ports may share a role, and a single port entry has exactly one role.
    //
    // +kubebuilder:validation:Enum=Serving;Metrics;Health
    // +kubebuilder:default=Serving
    // +optional
    Role PortRole `json:"role,omitempty"`
}

type PortRole string

const (
    PortRoleServing PortRole = "Serving"
    PortRoleMetrics PortRole = "Metrics"
    PortRoleHealth  PortRole = "Health"
)
```

### Validation

No new CEL rule is added. An earlier draft of this proposal put an
`XValidation` rule on `targetPorts` requiring at least one `Serving`-role port, but adding a
shape constraint to a stable (`v1`) API is a backward-incompatible schema change: it can reject
updates to InferencePools that are already stored in etcd, and the repository's `crdify` gate
correctly rejects it. The precedent is `appProtocol` (#2162), which ships as an enum with a
default and no accompanying CEL rule.

The constraint is not needed for safety. Because defaulting runs before validation, every
existing spec that never mentions `role` gets `Serving` on every port, so the only way to reach
a pool with no `Serving` port is for an author to explicitly set every `targetPort`'s role to
`Metrics` and/or `Health`. Such a pool is well-defined rather than invalid: it simply exposes no
inference endpoints, and `PortNumberForRole` returns `0` for a `Serving` lookup. Enforcement of
"a pool ought to serve traffic" belongs to the controller's status conditions, not the schema.

Port-number uniqueness is unaffected by role: a given port number appears in exactly one
`targetPorts` entry, with exactly one role. If a deployment's serving port also happens to be the
health-check port (the common case, matching today's behavior), authors simply omit a separate
`Health` entry — role-aware consumers fall back to the `Serving` port when no dedicated
`Metrics`/`Health` port is declared, per the sketch in #2857.

### Consumer-facing accessor

To avoid every consumer re-implementing the "look for my role, else fall back to Serving" search,
`InferencePoolSpec` gains a small helper:

```go
// PortNumberForRole returns the port number of the first targetPort with the given role. If no
// targetPort has that role, it falls back to the first targetPort with the "Serving" role, and
// returns 0 when the spec has no "Serving" port at all.
func (s *InferencePoolSpec) PortNumberForRole(role PortRole) PortNumber
```

### `pkg/lwepp` wiring

The reference EPP's `InferencePoolToEndpointPool` conversion (`pkg/lwepp/util/pool`) currently
turns every `targetPorts` entry into a traffic endpoint. With roles, only `Serving`-role ports
(including the pre-existing default) are endpoints; `Metrics`/`Health`-only ports are excluded
from `EndpointPool.TargetPorts` so they are never dialed for inference traffic.

## Backward compatibility

* Existing manifests with no `role` field: unaffected. Defaulting fills `Serving` on every port,
  so `pkg/lwepp`'s endpoint list is unchanged.
* No new validation constraint is added to `targetPorts`, so no already-stored InferencePool can
  become unupdatable. This is verified by the `crdify` CRD-compatibility gate.
* Existing manifests are also unaffected by CRD schema changes elsewhere (`appProtocol`,
  `endpointPickerRef`, etc.) — this is purely an additive field on `Port`.
* No conversion webhook is needed: `v1` is the only served/stored version of `InferencePool`.

## Alternatives considered

* **Plugin-level config only** (status quo via PR #2762's `metricsPort`): rejected as the
  motivating issue — it doesn't generalize to health checking or other consumers, and requires
  every future consumer to reinvent per-plugin port configuration.
* **A separate `metricsPort`/`healthPort` field directly on `InferencePoolSpec`** instead of a
  `Role` enum on `Port`: rejected because it doesn't compose with `TargetPorts`' existing
  uniqueness/cardinality validation and conformance test surface, and can't express "the health
  port is also a serving port" without a second field meaning "same as serving."

## Open questions for maintainers

* Should `pkg/epp` (the full production EPP)'s metrics-data-source plugin be updated in this same
  PR to prefer `PortNumberForRole(Metrics)` over its own `metricsPort` parameter, or is that
  better left as separate follow-up once the API shape itself is approved?
* Conformance: should a new conformance test assert that an implementation's Service/target-port
  wiring honors `Role`, or is that premature before an implementation actually consumes it?
