# Quick Start Guide

## Deploy in 3 Steps

### 1. Update Image Names

Edit `kustomization.yaml` and set your container images:

```yaml
images:
  - name: training-server
    newName: quay.io/your-org/training-server
    newTag: v1.0.0

  - name: prediction-server
    newName: quay.io/your-org/prediction-server
    newTag: v1.0.0
```

### 2. Preview Deployment

```bash
kubectl kustomize .
```

This shows what will be deployed with your image substitutions applied.

### 3. Deploy

```bash
kubectl apply -k .
```

## Verify Deployment

```bash
# Check pods are running
kubectl get pods

# Check services
kubectl get svc

# View logs
kubectl logs -l app=training-server
kubectl logs -l app=prediction-server
```

## Get Prediction Service URL

```bash
# Get external IP
kubectl get svc prediction-service

# Test the service
curl http://<EXTERNAL-IP>/healthz
```

## Update Images

To update to a new version:

1. Edit `kustomization.yaml`:
```yaml
images:
  - name: training-server
    newName: quay.io/your-org/training-server
    newTag: v1.1.0  # New version
```

2. Apply changes:
```bash
kubectl apply -k .
```

Kubernetes will perform a rolling update.

## Common Customizations

### Change Replicas

Edit `kustomization.yaml`:

```yaml
patches:
  - target:
      kind: Deployment
      name: prediction-server-deployment
    patch: |-
      - op: replace
        path: /spec/replicas
        value: 5
```

### Override Config

```yaml
configMapGenerator:
  - name: prediction-server-config
    behavior: merge
    literals:
      - MODEL_SYNC_INTERVAL_SEC=30
      - USE_TREELITE=true
```

### Deploy to Different Namespace

```yaml
namespace: my-namespace
```

## Clean Up

```bash
kubectl delete -k .
```

## Need More Help?

See [README.md](README.md) for detailed documentation.
