# developement

```bash
export MINOR_VERSION=${MINOR_VERSION:-36} # define minor version if you don't already have one
export VERSION="v0.0.${MINOR_VERSION}" # build version tag
kustomize build manifests | kubectl delete -f - -n greg-slo-aware || true # wipe old deployment
docker build . --platform linux/amd64 -f ./training/Dockerfile -t quay.io/grpereir/slo-predicter-trainer:${VERSION} && docker push quay.io/grpereir/slo-predicter-trainer:${VERSION} # build training server image
docker build . --platform linux/amd64 -f ./test/Dockerfile -t quay.io/grpereir/slo-predicter-tester:${VERSION} && docker push quay.io/grpereir/slo-predicter-tester:${VERSION} # build testing job image
docker build . --platform linux/amd64 -f ./prediction/Dockerfile -t quay.io/grpereir/slo-predicter:${VERSION} && docker push quay.io/grpereir/slo-predicter:${VERSION} # build prediction server image
yq eval '.images[].newTag = env(VERSION)' -i manifests/kustomization.yaml # patch the kustomization file with version env var
kustomize build manifests | kubectl apply -f - -n greg-slo-aware # deploy the new setup
k delete job latency-predictor-test || true # this job skips all the code tests unless the pods are ready, so we delete it the first time around
k delete job latency-predictor-functional-test || true
k delete job latency-predictor-stress-test || true
kubectl wait --for=condition=Ready pod -l 'app!=latency-predictor-test' -n greg-slo-aware --timeout=300s # waits for pods to be ready, however THIS HANGS indefinetly, even when the pods come online
kustomize build manifests | kubectl apply -f - -n greg-slo-aware # re-create the job and kick it off
export MINOR_VERSION=$((${MINOR_VERSION} + 1 ))
```

no rebuild required:
```bash
kustomize build manifests | kubectl delete -f - -n greg-slo-aware || true
kustomize build manifests | kubectl apply -f - -n greg-slo-aware
k delete job latency-predictor-test || true
k delete job latency-predictor-functional-test || true
k delete job latency-predictor-stress-test || true
kubectl wait --for=condition=Ready pod -l 'app!=latency-predictor-test' -n greg-slo-aware --timeout=300s
kustomize build manifests | kubectl apply -f - -n greg-slo-aware
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


### OTHER

So XGBoost supports quantile regression which will tell us what our tail latency is, which is how we can be comfortable with our routing decisions because theres a 95% chance the SLO will be met. However, Treelite does not support built in quantile regression, instead we can wrap it with conformal prediction. This will use the training cycle to create an upper bound for the 95 percentile for this wrapper. 

We start with a dataset where we know the actual and system conditions (number of servers, queue depth, request length, etc.). Then we take the model’s prediction and add a margin big enough that, in calibration, it covered ~95% of cases.

challenge: depends on what came before is similar to what will come next


guides:
- https://medium.com/homeday/speeding-up-xgboost-models-with-treelite-d4e1e529ca17
- https://www.kaggle.com/code/code1110/janestreet-faster-inference-by-xgb-with-treelite
- https://stackoverflow.com/questions/50615033/is-it-possible-to-train-an-xgboost-model-in-python-and-deploy-it-run-it-in-c-c?utm_source=chatgpt.com
- https://mlsys.org/Conferences/doc/2018/196.pdf

------------------------------------------------

Start a chat with ezra and folks. From the leads meeting:
    - They don't want to integrate SGLange into every path not because of the competition, but because of maintainability. We are still doing this refactor of helm to kustomize, but while that work is in progress, we don't want ot have to add more support to the helmchart for SGLang because the config would be completely different
    - We still want to show its feasible but not maintain integration into all of it
    - 

