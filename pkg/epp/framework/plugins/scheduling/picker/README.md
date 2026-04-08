# Scheduling Pickers

Scheduling Pickers represent the final phase of the scheduling cycle in the Gateway API Inference Extension. After candidate endpoints have been filtered and scored by preceding plugins, the Picker is responsible for selecting the final subset of endpoints (typically just one) to receive the request.

The framework provides three standard picker implementations:
- [Max Score Picker](maxscore/README.md)
- [Random Picker](random/README.md)
- [Weighted Random Picker](weightedrandom/README.md)

All pickers share a common configuration structure and accept the `maxNumOfEndpoints` parameter.

> [!NOTE]
> If `maxNumOfEndpoints` is configured to be greater than `1`, the EPP will join all selected endpoints into a comma-separated string for the routing layer (e.g., assigned to `TargetEndpoint`). However, the framework's internal tracking for post-scheduling plugins (like response handlers) will only reference the **first** endpoint in the list.
