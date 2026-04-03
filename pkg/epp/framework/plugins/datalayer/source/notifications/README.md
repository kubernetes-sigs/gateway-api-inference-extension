# Notification Data Sources

This package provides two event-driven `DataSource` plugins for the EPP data layer:

- **`k8s-notification-source`** — watches a single Kubernetes GVK (e.g. `Pod`, `Service`) and
  dispatches `NotificationEvent`s to registered `NotificationExtractor`s when objects are
  created, updated, or deleted. Multiple extractors can be notified of GVK changes by a single
  source. Multiple sources are required for different GVKs.
- **`endpoint-notification-source`** — delivers endpoint lifecycle events (add/update, delete)
  to registered `EndpointExtractor`s whenever an endpoint is added to or removed from the
  datastore.

Both sources pass events through to their extractors unchanged. Extractors carry out the
actual work: enriching endpoint attributes, updating caches, triggering side effects, etc.
Multiple extractors can be attached to the notification sources.

## Configuration

Plugins are declared in the top-level `plugins` list and wired into the data layer via the
`data.sources` section. When no `name` is specified the plugin name defaults to its `type`
string, which is the value used in `pluginRef` entries.

### Watching a Kubernetes GVK

```yaml
plugins:
  - type: k8s-notification-source
    parameters:
      group: ""        # empty for core API group
      version: v1
      kind: Pod
  - type: my-pod-extractor   # user-provided extractor plugin

data:
  sources:
    - pluginRef: k8s-notification-source
      extractors:
        - pluginRef: my-pod-extractor
```

### Reacting to endpoint lifecycle events

```yaml
plugins:
  - type: endpoint-notification-source
  - type: my-endpoint-extractor   # user-provided extractor plugin

data:
  sources:
    - pluginRef: endpoint-notification-source
      extractors:
        - pluginRef: my-endpoint-extractor
```

## Writing extractors

### GVK extractor (`NotificationExtractor`)

Implement `NotificationExtractor` to process events from a `k8s-notification-source`.
The framework calls `ExtractNotification` for every add, update, and delete event on
the watched GVK.

```go
package myplugin

import (
	"context"
	"fmt"
	"reflect"

	"k8s.io/apimachinery/pkg/runtime/schema"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// PodEventLogger logs Pod add/remove events.
type PodEventLogger struct{}

func (e *PodEventLogger) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "my-pod-extractor", Name: "my-pod-extractor"}
}

func (e *PodEventLogger) ExpectedInputType() reflect.Type {
	return fwkdl.NotificationEventType
}

// Extract is the base Extractor method — not called for notification extractors.
func (e *PodEventLogger) Extract(_ context.Context, _ any, _ fwkdl.Endpoint) error { return nil }

func (e *PodEventLogger) GVK() schema.GroupVersionKind {
	return schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"}
}

func (e *PodEventLogger) ExtractNotification(_ context.Context, event fwkdl.NotificationEvent) error {
	action := "added/updated"
	if event.Type == fwkdl.EventDelete {
		action = "removed"
	}
	fmt.Printf("pod %s: %s\n", event.Object.GetName(), action) // note: objects are delivered as Unstructured.
	return nil
}
```

### Endpoint extractor (`EndpointExtractor`)

Implement `EndpointExtractor` to process events from an `endpoint-notification-source`.
The framework calls `ExtractEndpoint` for every add/update and delete event on an endpoint.

```go
package myplugin

import (
	"context"
	"fmt"
	"reflect"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// EndpointEventLogger logs endpoint add/remove events.
type EndpointEventLogger struct{}

func (e *EndpointEventLogger) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "my-endpoint-extractor", Name: "my-endpoint-extractor"}
}

func (e *EndpointEventLogger) ExpectedInputType() reflect.Type {
	return fwkdl.EndpointEventReflectType
}

// Extract is the base Extractor method — not called for endpoint extractors.
func (e *EndpointEventLogger) Extract(_ context.Context, _ any, _ fwkdl.Endpoint) error { return nil }

func (e *EndpointEventLogger) ExtractEndpoint(_ context.Context, event fwkdl.EndpointEvent) error {
	action := "added/updated"
	if event.Type == fwkdl.EventDelete {
		action = "removed"
	}
	fmt.Printf("endpoint %s: %s\n", event.Endpoint.GetMetadata().GetNamespacedName(), action)
	return nil
}
```
