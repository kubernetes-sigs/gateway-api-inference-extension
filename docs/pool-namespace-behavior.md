### Pool Namespace Resolution

When running the Endpoint Picker (EPP), the namespace for the InferencePool can be set in three ways (in order of precedence):

1. **Explicit flag**: If the `--pool-namespace` flag is provided, its value is used.
2. **Environment variable**: If the flag is not set (or set to `default`), the `NAMESPACE` environment variable is checked. If set, its value is used.
3. **Default**: If neither is set, the namespace defaults to `default`.

This allows the EPP to automatically use the namespace it is running in (when the `NAMESPACE` env var is set via Kubernetes Downward API), without requiring explicit configuration.

Example manifest snippet to set the env var from pod metadata:

```yaml
env:
  - name: NAMESPACE
    valueFrom:
      fieldRef:
        fieldPath: metadata.namespace
```

This behavior ensures the EPP works out-of-the-box in most deployment scenarios.
