# Plugin Lifecycle and Stability Levels

Author(s): @hexfusion

Related issues:
- https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/2653
- https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/1405

## Proposal Status
***Draft***

## Summary

GIE's plugin system is growing. Extension points now support
multiple implementations, and more plugin types are coming as
the EPP evolves (data layer sources, parsers, flow control
policies). This growth is healthy, it lets contributors
experiment with new approaches and iterate quickly.

Today there is no mechanism to communicate plugin maturity to
operators. A plugin either exists in the registry or it doesn't.
There is no way to distinguish "this plugin is experimental and
may change" from "this plugin is stable and its config API is
committed." Without a clear support contract, operators can't
make informed deployment decisions, and maintainers can't iterate
on plugin designs without risking silent breakage for users who
adopted them early.

A plugin lifecycle model would let experimentation and stability
coexist: contributors can ship new plugins without the pressure
of immediate stability guarantees, and operators can see exactly
what they're opting into.

## Goals

* Define maturity tiers for EPP plugins (Alpha, Beta, Stable)
  with clear support contracts at each tier
* Gate experimental plugins behind feature flags so they're
  opt-in by default
* Reject removed plugins at config validation time with
  actionable error messages
* Communicate stability to operators at startup via structured
  log messages

## Non-Goals

* Runtime stability negotiation (plugins don't change stability
  while running)
* Out-of-tree plugin certification, conformance testing, or
  governance of stability declarations
* CRD-level stability annotations (this proposal covers compiled
  EPP plugins only)

## Prior Art

kube-scheduler gates alpha plugins via feature flags and
hard-rejects removed plugins at config validation time. Gateway
API uses [Standard/Experimental channels](https://gateway-api.sigs.k8s.io/concepts/versioning/) with
formal graduation criteria. Neither system puts stability
metadata in the plugin interface itself.

## Proposed Design

Stability is managed through the plugin registry, feature gates,
and config validation not through the Plugin interface.

### Stability Levels

Plugin stability uses three maturity tiers: Alpha, Beta, and
Stable. These are plugin-specific labels, not Kubernetes API
versions. There is no separate "Deprecated" level, deprecation
is a signal (a message indicating replacement), not a maturity
tier. The plugin's current level determines its removal timeline.

| Level | Default | Config Contract | Removal Policy |
|-------|---------|-----------------|----------------|
| **Alpha** | Gated off (requires feature gate) | No compatibility guarantee. Config schema may change between releases. | Can be removed any release. |
| **Beta** | Gated on | Config schema is stable. Behavioral changes require release notes. | 2 releases + 6 months after deprecation notice. |
| **Stable** | Always available | Full backward compatibility within config API version. | Not removed within a config API major version. |

**Deprecation** is orthogonal to level. A plugin at any level
can carry a deprecation message signaling that it will be
removed. The level determines how long it must remain available
after that signal. When the policy window expires, the plugin is
removed from the registry entirely. A separate validation
tombstone provides the migration message for stale configs that
still reference it.

**Removal** is not a stability level. Removed plugins are
deleted from the registry. A tombstone map in the validation
layer catches stale configs and returns actionable errors with
migration guidance.

These tiers and their removal policies are defined by this
proposal and are specific to GIE's plugin system. They do not
map to Kubernetes API versions and are independent of the
`EndpointPickerConfig` API version.

### Key Mechanisms

**Registry metadata.** The existing `plugin.Registry` (a
`map[string]FactoryFunc`) is extended to carry stability,
feature gate, and deprecation message alongside the factory
function. This is the single source of truth for plugin
maturity. No changes to the `Plugin` interface are needed.

**Feature gate integration.** Alpha plugins require an explicit
feature gate in `EndpointPickerConfig.FeatureGates`. GIE already
has a `FeatureGates []string` field on the config; this proposal
extends its use to cover per-plugin gating.

**Config validation.** At config load time:
* Alpha plugins without their feature gate enabled are rejected
  with an actionable error
* Removed plugins are rejected with migration guidance
* Plugins with a deprecation message are accepted but log a
  warning with the replacement and removal timeline

**Startup logging.** Every loaded plugin is logged with its
stability level and any deprecation message. This gives
operators immediate visibility into what they're running.

## Implementation

The implementation is scoped to the GIE framework packages. No
changes to the `Plugin` interface or individual plugin code are
required in Phase 1 or 2.

### Current State

Today `plugin.Registry` is a `map[string]FactoryFunc` with no
metadata. Feature gates are phase-level (`prepareDataPlugins`,
`experimentalDatalayer`, `flowControl`), not per-plugin.
Validation checks profile references and gate names but knows
nothing about plugin maturity.

### Phase 1: Registry Metadata + Startup Logging

**Goal:** Every plugin in the registry carries stability
metadata. Operators see stability at startup.

**Changes to `pkg/epp/framework/interface/plugin/registry.go`:**

```go
// StabilityLevel defines the maturity of a registered plugin.
// Three maturity tiers that define the config contract and
// removal policy. These are plugin-specific labels, not
// Kubernetes API versions. Deprecation is orthogonal (a
// message, not a level). Removal means the plugin leaves
// the registry entirely.
type StabilityLevel string

const (
    // Unknown is the zero value. Assigned to plugins registered
    // via the backward-compatible Register() path that have not
    // yet opted into the lifecycle model.
    Unknown StabilityLevel = "Unknown"
    Alpha   StabilityLevel = "Alpha"
    Beta    StabilityLevel = "Beta"
    Stable  StabilityLevel = "Stable"
)

// IsValid returns true if s is a recognized stability level
// that carries a support contract. Unknown is recognized but
// indicates the plugin has not declared its stability.
func (s StabilityLevel) IsValid() bool {
    switch s {
    case Unknown, Alpha, Beta, Stable:
        return true
    }
    return false
}

// RegistryEntry holds a plugin factory and its lifecycle
// metadata.
type RegistryEntry struct {
    // Factory instantiates the plugin.
    Factory FactoryFunc

    // Stability is the maturity level of this plugin.
    // Unknown for plugins registered via Register();
    // Alpha, Beta, or Stable for plugins registered
    // via MustRegister().
    Stability StabilityLevel

    // FeatureGate is the feature gate name required for
    // Alpha plugins. Must be non-empty when Stability is
    // Alpha.
    FeatureGate string

    // DeprecationMessage, if non-empty, signals that this
    // plugin will be removed in a future release. Logged as
    // a warning at startup. The plugin remains fully
    // functional. The removal timeline is determined by the
    // plugin's stability level.
    DeprecationMessage string
}

// Registry is the global plugin registry, keyed by plugin
// type string. All registration must complete before
// LoadRawConfig is called. Concurrent registration is not
// supported.
var Registry = map[string]RegistryEntry{}

// Register adds a plugin factory to the registry without
// stability metadata. Plugins registered this way get Unknown
// stability and will log a warning at startup prompting the
// author to migrate to MustRegister. This preserves backward
// compatibility for out-of-tree plugins that have not yet
// opted into the lifecycle model.
func Register(pluginType string, factory FactoryFunc) {
    Registry[pluginType] = RegistryEntry{
        Factory:   factory,
        Stability: Unknown,
    }
}

// MustRegister adds a plugin factory with explicit lifecycle
// metadata and panics on invalid plugin.
func MustRegister(pluginType string, entry RegistryEntry) {
    if !entry.Stability.IsValid() {
        panic(fmt.Sprintf(
            "plugin %q: invalid stability level %q",
            pluginType, entry.Stability))
    }
    if entry.Stability == Alpha && entry.FeatureGate == "" {
        panic(fmt.Sprintf(
            "plugin %q: alpha plugins must specify a FeatureGate",
            pluginType))
    }
    if entry.Factory == nil {
        panic(fmt.Sprintf(
            "plugin %q: Factory must not be nil",
            pluginType))
    }
    Registry[pluginType] = entry
}
```

**Startup logging** is a separate pass (`logPluginStability`)
that runs after validation but before factory calls. It logs
each plugin's name, type, and stability level. Plugins with a
`DeprecationMessage` get an additional warning.

**Migration path:** Existing `plugin.Register()` calls continue
to work with `Unknown` stability. Plugin authors adopt
`MustRegister()` at their own pace.

### Phase 2: Alpha Gating + Removed Plugin Rejection

**Goal:** Alpha plugins require explicit opt-in. Removed plugins
produce actionable errors. Stability validation runs before
plugin factories are called.

```go
// removedPlugins is a tombstone map for plugins that have been
// deleted from the registry. When an operator's config
// references a removed plugin, validation returns an actionable
// error with migration guidance instead of the generic "not
// registered" error from instantiatePlugins. Tombstones are
// permanent and small.
var removedPlugins = map[string]string{
    // Populated as plugins are removed. Key is the plugin type,
    // value is the migration message. Example:
    // "old-plugin": "Use new-plugin instead. See https://...",
}

func validatePluginStability(
    cfg *configapi.EndpointPickerConfig,
) error {
    enabledGates := sets.New(cfg.FeatureGates...)

    for _, spec := range cfg.Plugins {
        // Check tombstones first -- give a useful migration
        // error instead of the generic "not registered" from
        // instantiatePlugins.
        if msg, ok := removedPlugins[spec.Type]; ok {
            return fmt.Errorf(
                "plugin type '%s' has been removed: %s",
                spec.Type, msg,
            )
        }

        entry, ok := fwkplugin.Registry[spec.Type]
        if !ok {
            continue // Will be caught by instantiatePlugins.
        }

        // Alpha plugins require their feature gate to be
        // explicitly enabled.
        if entry.Stability == fwkplugin.Alpha {
            if !enabledGates.Has(entry.FeatureGate) {
                return fmt.Errorf(
                    "plugin '%s' (type: %s) is alpha and "+
                        "requires feature gate '%s' to be "+
                        "enabled in featureGates",
                    spec.Name, spec.Type, entry.FeatureGate,
                )
            }
        }
    }
    return nil
}
```

**Removed plugins** are deleted from the registry. The
maintainer removes the `MustRegister` call and adds a tombstone
to `removedPlugins`. Tombstones are permanent and small.

**Feature gate registration** for alpha plugins is manual via
`loader.RegisterFeatureGate()`, called alongside
`plugin.MustRegister()`.

## Open Questions

1. Should alpha plugins be completely invisible in the default
   config, or just gated off?
2. Should graduation criteria be GIE-specific, or adopt Gateway
   API's requirements?
3. Where does the stability policy live, `docs/plugin-lifecycle.md`,
   `CONTRIBUTING.md`, or a dedicated proposal?

