# developement

```bash
export MINOR_VERSION=${MINOR_VERSION:-36} # define minor version if you don't already have one
export VERSION="v0.0.${MINOR_VERSION}" # build version tag
kustomize build manifests | kubectl delete -f - -n greg-slo-aware || true # wipe old deployment
docker build . --platform linux/amd64 -f ./Dockerfile-training -t quay.io/grpereir/slo-predicter-trainer:${VERSION} && docker push quay.io/grpereir/slo-predicter-trainer:${VERSION} # build training server image
docker build . --platform linux/amd64 -f ./Dockerfile-test -t quay.io/grpereir/slo-predicter-tester:${VERSION} && docker push quay.io/grpereir/slo-predicter-tester:${VERSION} # build testing job image
docker build . --platform linux/amd64 -f ./Dockerfile-prediction -t quay.io/grpereir/slo-predicter:${VERSION} && docker push quay.io/grpereir/slo-predicter:${VERSION} # build prediction server image
yq eval '.images[].newTag = env(VERSION)' -i manifests/kustomization.yaml # patch the kustomization file with version env var
kustomize build manifests | kubectl apply -f - -n greg-slo-aware # deploy the new setup
k delete job latency-predictor-test || true # this job skips all the code tests unless the pods are ready, so we delete it the first time around
k delete job latency-predictor-functional-test || true
k delete job latency-predictor-stress-test || true
kubectl wait --for=condition=Ready pod -l 'app!=latency-predictor-test' -n greg-slo-aware --timeout=300s # waits for pods to be ready, however THIS HANGS indefinetly, even when the pods come online
kustomize build manifests | kubectl apply -f - -n greg-slo-aware # re-create the job and kick it off
export MINOR_VERSION=$((${MINOR_VERSION} + 1 ))
```

## batch size 1000

Going forward we should drop batch size 1000 from the stress tests, or limit it to a specific QPS different from the other batch sizes. It creates too much overhead for the testing pod. Either migrate to functional tests or drop.

## TreeLite conformal prediction coverage behavior

When using TreeLite mode with conformal prediction, coverage is consistently ~95% instead of the target 90%. This is expected behavior: conformal prediction uses absolute residuals `|actual - predicted|`, and for symmetric (Gaussian) noise distributions, P90 of absolute residuals corresponds to approximately P95 coverage, not P90. This provides conservative (safer) predictions which is desirable for SLO guarantees.

## Test isolation critical for conformal prediction

Unlike native quantile mode (which trains XGBoost with `reg:quantileerror`), TreeLite + conformal mode uses mean regression (`reg:squarederror`) + calibration from test set residuals. This means:

- Native quantile mode: Tolerant of stale data (predictions still reasonable if distribution similar)
- Conformal mode: **Requires clean test data** - calibration computed from current model's residuals on test set. If model wasn't retrained on new data → residuals from wrong distribution → calibration fails → coverage collapses

Tests that add training data must flush afterwards to avoid polluting conformal calibration in subsequent tests.
