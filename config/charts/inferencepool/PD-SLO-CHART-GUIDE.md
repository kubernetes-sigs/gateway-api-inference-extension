# PD-SLO Chart Configuration Guide

Configure the inferencepool Helm chart for PD (Prefill-Decode) disaggregated scheduling with SLO-aware optimization.

## Modes

**Legacy Mode** (default): Single predictor, 1 training + N prediction servers
**PD Mode**: Two predictors (prefill + decode), 4 sidecar containers total

### PD Mode Architecture
```
EPP Pod (5 containers)
├─ EPP Container
└─ Sidecars:
   ├─ Prefill Training Server (port 8000)
   ├─ Prefill Prediction Server (port 8001)
   ├─ Decode Training Server (port 8010)
   └─ Decode Prediction Server (port 8011)

Environment Variables (auto-generated):
- PREFILL_TRAINING_URL=http://localhost:8000
- PREFILL_PREDICTION_URL=http://localhost:8001
- DECODE_TRAINING_URL=http://localhost:8010
- DECODE_PREDICTION_URL=http://localhost:8011
```

## Quick Start

**Minimal Configuration**:
```yaml
inferenceExtension:
  latencyPredictor:
    enabled: true
    pdMode:
      enabled: true
      predictors:
        prefill:
          trainingServer:
            port: 8000
            resources:
              requests: {cpu: "500m", memory: "1Gi"}
          predictionServers:
            count: 1
            startPort: 8001
            resources:
              requests: {cpu: "250m", memory: "512Mi"}
        decode:
          trainingServer:
            port: 8010  # Must differ from prefill!
            resources:
              requests: {cpu: "500m", memory: "1Gi"}
          predictionServers:
            count: 1
            startPort: 8011
            resources:
              requests: {cpu: "250m", memory: "512Mi"}
```

**Deploy**:
```bash
helm install my-pool ./inferencepool -f values-pd-slo.yaml -n llm-d
```

## Configuration Details

### Required Settings

| Component | Setting | Value | Notes |
|-----------|---------|-------|-------|
| Prefill Training | `port` | 8000 | Must differ from decode |
| Prefill Prediction | `startPort` | 8001 | |
| Decode Training | `port` | 8010 | Must differ from prefill |
| Decode Prediction | `startPort` | 8011 | |
| Prediction Count | `count` | 1 | Can increase for production |

### Health Probes (Required)

Both training and prediction servers **must** have `livenessProbe` and `readinessProbe` configured with `httpGet.path` and `port`:

```yaml
prefill:
  trainingServer:
    livenessProbe:
      httpGet: {path: /healthz, port: 8000}
      initialDelaySeconds: 30
    readinessProbe:
      httpGet: {path: /readyz, port: 8000}
      initialDelaySeconds: 45
  predictionServers:
    livenessProbe:
      httpGet: {path: /healthz}
      initialDelaySeconds: 15
    readinessProbe:
      httpGet: {path: /readyz}
      initialDelaySeconds: 10
```

### Images and Resources

**Override per predictor** or **use global defaults** from legacy section (see `values.yaml`).

## Generated Resources

**ConfigMaps** (4 in PD mode):
- `<epp>-latency-predictor-prefill-training`
- `<epp>-latency-predictor-prefill-prediction`
- `<epp>-latency-predictor-decode-training`
- `<epp>-latency-predictor-decode-prediction`

**Environment Variables** (auto-injected into EPP container):
- `PREFILL_TRAINING_URL`, `PREFILL_PREDICTION_URL`
- `DECODE_TRAINING_URL`, `DECODE_PREDICTION_URL`

These are consumed by `llm-d-inference-scheduler`'s `PDPredictorSet` for latency prediction.

## Validation

```bash
# Check 5 containers (1 EPP + 4 sidecars)
kubectl get pods -n llm-d
kubectl describe pod <epp-pod> -n llm-d

# Verify environment variables
kubectl exec <epp-pod> -n llm-d -c epp -- env | grep -E "PREFILL|DECODE"
# Expected: PREFILL_TRAINING_URL=http://localhost:8000, etc.

# Test predictor health
kubectl exec <epp-pod> -n llm-d -c training-server-prefill -- curl http://localhost:8000/healthz
kubectl exec <epp-pod> -n llm-d -c training-server-decode -- curl http://localhost:8010/healthz
```

## Troubleshooting

| Issue | Symptom | Solution |
|-------|---------|----------|
| Port conflict | `address already in use` | Ensure prefill/decode ports differ (8000 vs 8010) |
| Missing env vars | `PREFILL_TRAINING_URL must be set` | Verify `pdMode.enabled=true`, check `kubectl exec <pod> -c epp -- env` |
| ConfigMap missing | `configmap not found` | Check `kubectl get cm -n llm-d \| grep latency-predictor` (should show 4) |
| Pod pending | `Insufficient cpu/memory` | Reduce resource requests (500m/1Gi for training, 250m/512Mi for prediction) |
| Probe failures | Containers restarting | Verify probe paths (`/healthz`, `/readyz`) and ports are configured |

## Important Notes

1. **PD Mode**: Set `pdMode.enabled=true` to enable dual-predictor architecture
2. **Ports**: Training servers must use different ports (prefill: 8000, decode: 8010)
3. **Probes**: Both `livenessProbe` and `readinessProbe` with `httpGet.path` and `port` are required
4. **Resources**: Start with 500m/1Gi (training), 250m/512Mi (prediction) for MVP
5. **Prediction Count**: `count: 1` for MVP, increase for production throughput
6. **Backward Compatibility**: Legacy mode (single predictor) still works when `pdMode.enabled=false`
7. **Joint Optimization**: Currently uses fallback (best pod from each profile). Full joint optimization TBD.

## Reference

- Chart values: `values.yaml`
- Scheduler guide: `llm-d-inference-scheduler/PD-SLO-GUIDE.md`
- Chart template: `templates/_latency-predictor.tpl`
