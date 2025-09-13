# EPP Configuration Flags

This page documents selected configuration flags for the Endpoint Picker (EPP) binary. Most flags are self-explanatory via their `--help` descriptions; only flags with nuanced or non-obvious behavior are detailed here.

## --pool-namespace

**Description:**
Specifies the namespace of the InferencePool this Endpoint Picker is associated with.

**Resolution order:**
1. If `--pool-namespace` is set to a non-empty value, its value is used.
2. If the flag is not set (i.e., left empty), the `NAMESPACE` environment variable is checked. If set, its value is used.
3. If neither is set, the namespace defaults to `default`.

This allows the EPP to automatically use the namespace it is running in (when the `NAMESPACE` env var is set via Kubernetes Downward API), without requiring explicit configuration. If you want to force the use of the default namespace, explicitly set `--pool-namespace=default`. If you want to use the environment variable or fallback, leave the flag unset or set it to an empty string.

**Example manifest snippet to set the env var from pod metadata:**

```yaml
env:
  - name: NAMESPACE
    valueFrom:
      fieldRef:
        fieldPath: metadata.namespace
```

---

For a full list of flags, run:

```
EPP_BINARY --help
```
