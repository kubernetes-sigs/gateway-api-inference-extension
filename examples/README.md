# Scheduling Framework Developer Guide

This directory contains working examples that demonstrate how to extend the
Gateway API Inference Extension (GIE) scheduling framework with custom
out-of-tree plugins and how to build a custom EPP binary.

For a real-world out-of-tree project built on this framework, see
[llm-d-inference-scheduler](https://github.com/llm-d/llm-d-inference-scheduler).

## Directory Layout

```
examples/
├── scheduler/                           # Example 1: Scheduler plugin development
│   ├── main.go                          # Standalone demo — runs locally, no cluster needed
│   └── plugins/
│       ├── register.go                  # RegisterAllPlugins() — central registration
│       ├── filter/
│       │   └── model_affinity.go        # Custom Filter plugin
│       ├── scorer/
│       │   └── least_loaded.go          # Custom Scorer plugin
│       ├── picker/
│       │   └── top_k_random.go          # Custom Picker plugin
│       └── profile/
│           └── logging_profile_handler.go # Custom ProfileHandler plugin
├── custom-epp/                          # Example 2: Custom EPP binary
│   └── main.go                          # Embeds custom plugins into the real EPP
└── README.md                            # This file
```

## Quick Start

**Run the scheduler demo** (no cluster required):

```bash
make run-example EXAMPLE=scheduler
```

**Build a custom EPP binary** (requires a Kubernetes cluster to run):

```bash
make build-example EXAMPLE=custom-epp
```

The binary is output to `bin/custom-epp`.

## Scheduling Architecture

The scheduler uses a **Filter → Score → Pick** pipeline, driven by a
**ProfileHandler**:

```
                    ┌──────────────────────────────────────────────┐
                    │              ProfileHandler                  │
                    │  Pick(): which profiles to run               │
                    │  ProcessResults(): choose primary profile    │
                    └──────────────┬───────────────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────────────┐
                    │           SchedulerProfile                   │
                    │                                              │
                    │  ┌─────────┐  ┌──────────┐  ┌────────────┐  │
                    │  │ Filter  │→ │  Scorer   │→ │   Picker   │  │
                    │  │ (N)     │  │(N,weighted)│  │  (1)       │  │
                    │  └─────────┘  └──────────┘  └────────────┘  │
                    └──────────────────────────────────────────────┘
```

- **Filter** — removes endpoints that should not receive the request.
- **Scorer** — assigns a score in `[0, 1]` to each surviving endpoint.
  Scores are multiplied by the scorer's weight and summed.
- **Picker** — selects the final endpoint(s) from the scored list.
- **ProfileHandler** — controls which profiles run and aggregates results.

## How to Write a Custom Plugin

### Step 1: Implement the Plugin Interface

Every plugin must satisfy `plugin.Plugin` by implementing `TypedName()`:

```go
import plugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"

type MyPlugin struct {
    typedName plugin.TypedName
}

func (p *MyPlugin) TypedName() plugin.TypedName { return p.typedName }
```

### Step 2: Implement the Scheduling Interface

Choose which extension point your plugin targets:

#### Filter

```go
import sched "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

// Drop endpoints that should not serve this request.
func (f *MyFilter) Filter(ctx context.Context, cycleState *sched.CycleState,
    request *sched.LLMRequest, endpoints []sched.Endpoint) []sched.Endpoint {
    // return a subset of endpoints
}
```

See [`plugins/filter/model_affinity.go`](scheduler/plugins/filter/model_affinity.go) for a complete example.

#### Scorer

```go
// Category tells the framework what kind of preference this scorer applies.
func (s *MyScorer) Category() sched.ScorerCategory {
    return sched.Distribution  // or sched.Affinity, sched.Balance
}

// Score returns a value in [0, 1] for each endpoint. Higher is better.
func (s *MyScorer) Score(ctx context.Context, cycleState *sched.CycleState,
    request *sched.LLMRequest, endpoints []sched.Endpoint) map[sched.Endpoint]float64 {
    // return scores
}
```

See [`plugins/scorer/least_loaded.go`](scheduler/plugins/scorer/least_loaded.go) for a complete example.

#### Picker

```go
// Pick selects one or more endpoints from the scored candidates.
func (p *MyPicker) Pick(ctx context.Context, cycleState *sched.CycleState,
    scoredEndpoints []*sched.ScoredEndpoint) *sched.ProfileRunResult {
    // select endpoint(s) and return result
}
```

See [`plugins/picker/top_k_random.go`](scheduler/plugins/picker/top_k_random.go) for a complete example
that sorts by score, keeps the top K candidates, and randomly picks one —
balancing quality with load distribution.

#### ProfileHandler

```go
// Pick selects which profiles to run. Return empty map to stop the loop.
func (h *MyHandler) Pick(ctx context.Context, cycleState *sched.CycleState,
    request *sched.LLMRequest, profiles map[string]sched.SchedulerProfile,
    profileResults map[string]*sched.ProfileRunResult) map[string]sched.SchedulerProfile {
    // return profiles to execute
}

// ProcessResults chooses the primary profile after all profiles finish.
func (h *MyHandler) ProcessResults(ctx context.Context, cycleState *sched.CycleState,
    request *sched.LLMRequest,
    profileResults map[string]*sched.ProfileRunResult) (*sched.SchedulingResult, error) {
    // return final result
}
```

See [`plugins/profile/logging_profile_handler.go`](scheduler/plugins/profile/logging_profile_handler.go) for a complete example.

### Step 3: Provide a FactoryFunc

The config loader uses a `FactoryFunc` to instantiate plugins from YAML:

```go
func Factory(name string, rawParameters json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
    p := New()
    p.typedName.Name = name
    // Optionally parse rawParameters for plugin-specific config
    return p, nil
}
```

### Step 4: Register in RegisterAllPlugins()

Add your plugin to the central registration function:

```go
// plugins/register.go
func RegisterAllPlugins() {
    plugin.Register("my-filter-type",  filter.Factory)
    plugin.Register("my-scorer-type",  scorer.Factory)
    plugin.Register("my-picker-type",  picker.Factory)
    plugin.Register("my-handler-type", profile.Factory)
}
```

## How to Build a Custom EPP

A custom EPP is deployed alongside (or instead of) the default EPP. Each
`InferencePool` independently references whichever EPP it needs via
`endpointPickerRef`.

### How a Request Finds Its EPP

There are two separate flows to understand: the **control plane** (how the
gateway is configured) and the **data plane** (how a request is routed at
runtime).

#### Control Plane

Configuration is **declarative**: you write Gateway API objects, and an
**inference-capable Gateway controller** (for example Envoy Gateway with this
extension, or another implementation that conforms to the
[implementer's guide](https://gateway-api-inference-extension.sigs.k8s.io/guides/implementers/))
turns them into proxy config. The controller typically watches at least
`Gateway`, `HTTPRoute`, and `InferencePool`, and resolves backend endpoints
(for example by tracking Pods that match the pool’s label selector, or via a
shadow Service—both patterns are described in that guide).

The `HTTPRoute` binds traffic to a `Gateway` via `parentRefs` and names an
`InferencePool` as a **backend** via `rules[].backendRefs`. The pool’s
`endpointPickerRef` points at the Kubernetes **Service** that fronts your EPP
Deployment (including the gRPC service port, commonly `9002`). Reconciliation
does not “call” the EPP; it installs static/dynamic proxy configuration so
that, at request time, the data plane knows which EPP address to use for
ext_proc and how to reach pool member pods.

```
                    ┌─────────────────────────┐
                    │        Gateway          │
                    │ (listener / virtual IP) │
                    └───────────▲─────────────┘
                                │  parentRefs
                    ┌───────────┴─────────────┐
                    │       HTTPRoute         │
                    │  backendRefs → pool-a   │
                    └───────────┬─────────────┘
                                │  (ref: group, kind, name)
                                ▼
                    ┌─────────────────────────┐
                    │     InferencePool       │
                    │  name: pool-a           │
                    │  selector + targetPorts │
                    │  endpointPickerRef:     │
                    │    name: epp-a (Service)│
                    │    port: 9002           │
                    └───────────┬─────────────┘
                                │
              ┌─────────────────▼─────────────────┐
              │ Inference Gateway controller      │
              │ reconciles resources → proxy      │
              │ config (route, clusters, ext_proc │
              │ cluster/authority for epp-a:9002) │
              └───────────────────────────────────┘
```

#### Data Plane

At runtime, traffic never flows **through** the `InferencePool` object—it is
only the API that defined the pool’s selector, ports, and EPP reference. The
live path is: **client → proxy → ext_proc round-trip to EPP → chosen
model-server Pod**. The EPP **does not terminate or proxy** the user’s HTTP; it
returns a routing decision on the ext_proc stream.

Envoy’s ext_proc uses **gRPC** (bidirectional streaming). The proxy may pass an
optional candidate subset via metadata (for example
`x-gateway-destination-endpoint-subset`); if omitted, the EPP selects from
endpoints implied by the pool. The EPP must communicate the chosen endpoint
back using both the **`x-gateway-destination-endpoint` response header** and
matching **`dynamic_metadata`** in the ProcessingResponse (see the
[Endpoint Picker protocol](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/docs/proposals/004-endpoint-picker-protocol)).

```
  ① Client HTTP request
           │
           ▼
  ┌────────────────┐
  │ Envoy (proxy)  │  ② Match HTTPRoute; backend is InferencePool →
  └───────┬────────┘     run ext_proc before forwarding
          │
          │ ③ ext_proc gRPC: ProcessingRequest / ProcessingResponse
          ▼
  ┌────────────────┐
  │ EPP (epp-a)    │  ④ Scheduling pipeline: Filter → Score → Pick
  │ :9002          │
  └───────┬────────┘
          │
          │ ⑤ Decision: x-gateway-destination-endpoint + dynamic_metadata
          ▼
  ┌────────────────┐
  │ Envoy (proxy)  │  ⑥ Forward original request to chosen Pod:targetPort
  └───────┬────────┘
          ▼
  ┌────────────────┐
  │ Model server   │
  │ (e.g. vLLM)    │
  └────────────────┘
```

Step by step:

1. **User sends a request** to the Envoy-based Gateway (e.g. `POST /v1/completions`).
2. **Envoy matches** the request via HTTPRoute rules. The Gateway controller
   has already configured Envoy's ext-proc filter based on the `InferencePool`
   and its `endpointPickerRef` (Service + port for the EPP):
   ```yaml
   kind: HTTPRoute
   spec:
     rules:
     - backendRefs:
       - group: inference.networking.k8s.io
         kind: InferencePool
         name: pool-a          # ← which InferencePool
   ```
3. **Envoy calls the EPP** via ext-proc gRPC. The EPP is just a regular
   Kubernetes Service backed by the EPP Deployment.
4. **The EPP runs the scheduling pipeline** (Filter → Score → Pick) against
   the pool's endpoints.
5. **The EPP returns** the chosen endpoint in the `x-gateway-destination-endpoint`
   response header and dynamic metadata. The EPP does **not** proxy the request
   itself.
6. **Envoy forwards** the original request to that specific vLLM pod.

Because the binding path is `HTTPRoute → InferencePool → endpointPickerRef → EPP
Service`, different pools can point to different EPP deployments. This is how
multiple EPPs (upstream and custom) coexist in the same cluster.

**Important:** a **single** HTTP request matches **one** route rule (for a given
listener) and therefore uses **one** `InferencePool` and **one** ext_proc
call to **one** EPP. The data plane below shows **two independent examples**
(side by side), not one request fanning out to both EPPs.

```
  Control plane (declarative; reconciled into proxy config)

        HTTPRoute: /a -> pool-a, /b -> pool-b
                   │                  │
                   ▼                  ▼
     ┌──────────────────────┐  ┌──────────────────────┐
     │ InferencePool        │  │ InferencePool        │
     │ pool-a               │  │ pool-b               │
     │ pickerRef -> epp-a   │  │ pickerRef -> epp-b   │
     └──────────┬───────────┘  └──────────┬───────────┘
                └──────────┬──────────────┘
                           ▼
     ┌─────────────────────────────────────────────────┐
     │ Inference Gateway controller                    │
     │ reconciles -> Envoy config (routes + ext_proc)  │
     └─────────────────────────────────────────────────┘

  Data plane (two independent requests shown side by side)

     Request /a                        Request /b
          │                                 │
          └──────────────┬──────────────────┘
                         ▼
     ┌─────────────────────────────────────────────┐
     │             Envoy (Gateway)                 │
     │   each request: one route, one ext_proc     │
     └───────┬─────────────────────────────┬───────┘
             │ ext_proc                    │ ext_proc
             ▼                             ▼
     ┌──────────────────┐         ┌──────────────────┐
     │ EPP a (upstream) │         │ EPP b (custom)   │
     │ built-in plugins │         │ out-of-tree      │
     └────────┬─────────┘         └────────┬─────────┘
              │ decision                   │ decision
              ▼                            ▼
     ┌─────────────────────────────────────────────┐
     │             Envoy (Gateway)                 │
     │   forwards request to the chosen pod        │
     └───────┬─────────────────────────────┬───────┘
             ▼                             ▼
     ┌──────────────────┐         ┌──────────────────┐
     │ Pod in pool-a    │         │ Pod in pool-b    │
     └──────────────────┘         └──────────────────┘
```

Each `InferencePool` independently references whichever EPP it needs via
`endpointPickerRef`. Requests to `/a` are scheduled by the upstream EPP, while
requests to `/b` are scheduled by the custom EPP with out-of-tree plugins.
In both cases, Envoy calls the EPP to get a routing decision, then Envoy
itself forwards the request to the chosen model server pod.

A custom EPP binary is a standard Go program that registers your plugins and
then starts the upstream EPP Runner. This is the same pattern used by
[llm-d-inference-scheduler](https://github.com/llm-d/llm-d-inference-scheduler/blob/main/cmd/epp/main.go).

### In-tree (this repo)

See [`custom-epp/main.go`](custom-epp/main.go):

```go
import (
    "sigs.k8s.io/gateway-api-inference-extension/cmd/epp/runner"
    "sigs.k8s.io/gateway-api-inference-extension/examples/scheduler/plugins"
)

func main() {
    plugins.RegisterAllPlugins()
    runner.NewRunner().Run(ctrl.SetupSignalHandler())
}
```

### Out-of-tree (your own repo)

Create a new Go module that depends on gateway-api-inference-extension:

```
my-inference-scheduler/
├── go.mod                    # require sigs.k8s.io/gateway-api-inference-extension
├── cmd/epp/main.go           # Your entry point
├── pkg/plugins/
│   ├── register.go           # RegisterAllPlugins()
│   ├── filter/
│   ├── scorer/
│   └── profile/
├── config.yaml               # EndpointPickerConfig referencing your plugins
└── Dockerfile
```

Your `cmd/epp/main.go`:

```go
package main

import (
    "os"
    ctrl "sigs.k8s.io/controller-runtime"
    "sigs.k8s.io/gateway-api-inference-extension/cmd/epp/runner"
    "github.com/my-org/my-inference-scheduler/pkg/plugins"
)

func main() {
    plugins.RegisterAllPlugins()
    if err := runner.NewRunner().Run(ctrl.SetupSignalHandler()); err != nil {
        os.Exit(1)
    }
}
```

Your `go.mod`:

```
module github.com/my-org/my-inference-scheduler

require sigs.k8s.io/gateway-api-inference-extension v1.4.0
```

### YAML Configuration

The custom EPP loads an `EndpointPickerConfig` that can reference both
built-in and custom plugins:

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
  # Custom plugins (registered by your RegisterAllPlugins)
  - type: model-affinity-filter
    name: my-filter
  - type: least-loaded-scorer
    name: my-scorer
  - type: top-k-random-picker
    name: my-picker
    parameters:
      k: 3
  - type: logging-profile-handler
    name: my-handler
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: my-filter
      - pluginRef: my-scorer
        weight: 2
      - pluginRef: my-picker
profileHandler: my-handler
```

### Deploying

Build and deploy to your Kubernetes cluster, then point an `InferencePool`
at your custom EPP:

```yaml
apiVersion: inference.networking.k8s.io/v1
kind: InferencePool
metadata:
  name: my-pool
spec:
  selector:
    matchLabels:
      app: vllm
  targetPorts:
    - number: 8000
  endpointPickerRef:
    name: my-custom-epp   # Your EPP's Service name
    port:
      number: 9002
```

Multiple EPP deployments can coexist in a cluster — each `InferencePool`
independently references whichever EPP it needs via `endpointPickerRef`.
