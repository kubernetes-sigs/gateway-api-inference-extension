# InferencePool Rollout
The goal of this guide is to show you how to perform incremental roll out operations,
which gradually deploy new versions of your inference infrastructure.
You can update Inference Pool with minimal service disruption.
This page also provides guidance on traffic splitting and rollbacks to help ensure reliable deployments for InferencePool rollout.

InferencePool rollout is a powerful technique for performing various infrastructure and model updates with minimal disruption and built-in rollback capabilities.
This method allows you to introduce changes incrementally, monitor their impact, and revert to the previous state if necessary.

## Use Cases
Use Cases for InferencePool Rollout:

- Node(compute, accelerator) update roll out
- Base model roll out
- Model server framework rollout

### Node(compute, accelerator) update roll out
Node update roll outs safely migrate inference workloads to new node hardware or accelerator configurations.
This process happens in a controlled manner without interrupting model service.
Use node update roll outs to minimize service disruption during hardware upgrades, driver updates, or security issue resolution.

### Base model roll out
Base model updates roll out in phases to a new base LLM, retaining compatibility with existing LoRA adapters.
You can use base model update roll outs to upgrade to improved model architectures or to address model-specific issues.

### Model server framework rollout
Model server framework rollouts enable the seamless deployment of new versions or entirely different serving frameworks,
like updating from an older vLLM version to a newer one, or even migrating from a custom serving solution to a managed one.
This type of rollout is critical for introducing performance enhancements, new features, or security patches within the serving layer itself,
without requiring changes to the underlying base models or application logic. By incrementally rolling out framework updates,
teams can ensure stability and performance, quickly identifying and reverting any regressions before they impact the entire inference workload.

## How to do InferencePool rollout

1. **Deploy new infrastructure**: Create a new InferencePool configured with the new node(compute/accelerator) / model server / base model that you chose.
1. **Configure traffic splitting**: Use an HTTPRoute to split traffic between the existing InferencePool and the new InferencePool. The `backendRefs.weight` field controls the traffic percentage allocated to each pool.
1. **Maintain InferenceModel integrity**: Retain the existing InferenceModel configuration to ensure uniform model behavior across both node configurations or base model versions or model server versions.
1. **Preserve rollback capability**: Retain the original nodes and InferencePool during the roll out to facilitate a rollback if necessary.

## Example
This is an example of InferencePool rollout with node(compute, accelerator) update roll out

###  Prerequisites
Follow the steps in the [main guide](index.md)

### Deploy new infrastructure
You start with an existing InferencePool named vllm-llama3-8b-instruct.
To replace the original InferencePool, you create a new InferencePool named vllm-llama3-8b-instruct-new along with
InferenceModels and Endpoint Picker Extension configured with the updated node specifications of `nvidia-h100-80gb` accelerator type,

```yaml
kubectl apply -f - <<EOF
---
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: InferenceModel
metadata:
  name: food-review-new
spec:
  modelName: food-review
  criticality: Standard
  poolRef:
    name: vllm-llama3-8b-instruct-new
  targetModels:
    - name: food-review-1
      weight: 100
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama3-8b-instruct-new
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vllm-llama3-8b-instruct-new
  template:
    metadata:
      labels:
        app: vllm-llama3-8b-instruct-new
    spec:
      containers:
        - name: vllm
          image: "vllm/vllm-openai:latest"
          imagePullPolicy: Always
          command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
          args:
            - "--model"
            - "meta-llama/Llama-3.1-8B-Instruct"
            - "--tensor-parallel-size"
            - "1"
            - "--port"
            - "8000"
            - "--max-num-seq"
            - "1024"
            - "--compilation-config"
            - "3"
            - "--enable-lora"
            - "--max-loras"
            - "2"
            - "--max-lora-rank"
            - "8"
            - "--max-cpu-loras"
            - "12"
          env:
            - name: VLLM_USE_V1
              value: "1"
            - name: PORT
              value: "8000"
            - name: HUGGING_FACE_HUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token
                  key: token
            - name: VLLM_ALLOW_RUNTIME_LORA_UPDATING
              value: "true"
          ports:
            - containerPort: 8000
              name: http
              protocol: TCP
          lifecycle:
            preStop:
              sleep:
                seconds: 30
          livenessProbe:
            httpGet:
              path: /health
              port: http
              scheme: HTTP
            periodSeconds: 1
            successThreshold: 1
            failureThreshold: 5
            timeoutSeconds: 1
          readinessProbe:
            httpGet:
              path: /health
              port: http
              scheme: HTTP
            periodSeconds: 1
            successThreshold: 1
            failureThreshold: 1
            timeoutSeconds: 1
          startupProbe:
            failureThreshold: 600
            initialDelaySeconds: 2
            periodSeconds: 1
            httpGet:
              path: /health
              port: http
              scheme: HTTP
          resources:
            limits:
              nvidia.com/gpu: 1
            requests:
              nvidia.com/gpu: 1
          volumeMounts:
            - mountPath: /data
              name: data
            - mountPath: /dev/shm
              name: shm
            - name: adapters
              mountPath: "/adapters"
      initContainers:
        - name: lora-adapter-syncer
          tty: true
          stdin: true
          image: us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/lora-syncer:main
          restartPolicy: Always
          imagePullPolicy: Always
          env:
            - name: DYNAMIC_LORA_ROLLOUT_CONFIG
              value: "/config/configmap.yaml"
          volumeMounts: # DO NOT USE subPath, dynamic configmap updates don't work on subPaths
            - name: config-volume
              mountPath:  /config
      restartPolicy: Always
      enableServiceLinks: false
      terminationGracePeriodSeconds: 130
      nodeSelector:
        cloud.google.com/gke-accelerator: "nvidia-h100-80gb"
      volumes:
        - name: data
          emptyDir: {}
        - name: shm
          emptyDir:
            medium: Memory
        - name: adapters
          emptyDir: {}
        - name: config-volume
          configMap:
            name: vllm-llama3-8b-instruct-adapters-new
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: vllm-llama3-8b-instruct-adapters-new
data:
  configmap.yaml: |
    vLLMLoRAConfig:
      name: vllm-llama3-8b-instruct-adapters-new
      port: 8000
      defaultBaseModel: meta-llama/Llama-3.1-8B-Instruct
      ensureExist:
        models:
        - id: food-review-1
          source: Kawon/llama3.1-food-finetune_v14_r8
---
apiVersion: inference.networking.k8s.io/v1
kind: InferencePool
metadata:
  name: vllm-llama3-8b-instruct-new
spec:
  targetPortNumber: 8000
  selector:
    app: vllm-llama3-8b-instruct-new
  extensionRef:
    name: vllm-llama3-8b-instruct-epp-new
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-llama3-8b-instruct-epp-new
  namespace: default
spec:
  selector:
    app: vllm-llama3-8b-instruct-epp-new
  ports:
    - protocol: TCP
      port: 9002
      targetPort: 9002
      appProtocol: http2
  type: ClusterIP
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-llama3-8b-instruct-epp-new
  namespace: default
  labels:
    app: vllm-llama3-8b-instruct-epp-new
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-llama3-8b-instruct-epp-new
  template:
    metadata:
      labels:
        app: vllm-llama3-8b-instruct-epp-new
    spec:
      terminationGracePeriodSeconds: 130
      containers:
      - name: epp
        image: us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/epp:main
        imagePullPolicy: Always
        args:
        - --pool-name
        - "vllm-llama3-8b-instruct-new"
        - --pool-namespace
        - "default"
        - --v
        - "4"
        - --zap-encoder
        - "json"
        - --grpc-port
        - "9002"
        - --grpc-health-port
        - "9003"
        - --config-file
        - "/config/default-plugins.yaml"
        ports:
        - containerPort: 9002
          name: grpc
        - containerPort: 9003
          name: grpc-health
        - containerPort: 9090
          name: metrics
        livenessProbe:
          grpc:
            port: 9003
            service: inference-extension
          initialDelaySeconds: 5
          periodSeconds: 10
        readinessProbe:
          grpc:
            port: 9003
            service: inference-extension
          initialDelaySeconds: 5
          periodSeconds: 10
        volumeMounts:
        - name: plugins-config-volume
          mountPath: /config
      volumes:
      - name: plugins-config-volume
        configMap:
          name: plugins-config
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: plugins-config
  namespace: default
data:
  default-plugins.yaml: |
    apiVersion: inference.networking.x-k8s.io/v1alpha1
    kind: EndpointPickerConfig
    plugins:
    - type: low-queue-filter
      parameters:
        threshold: 128
    - type: lora-affinity-filter
      parameters:
        threshold: 0.999
    - type: least-queue-filter
    - type: least-kv-cache-filter
    - type: decision-tree-filter
      name: low-latency-filter
      parameters:
        current:
          pluginRef: low-queue-filter
        nextOnSuccess:
          decisionTree:
            current:
              pluginRef: lora-affinity-filter
            nextOnSuccessOrFailure:
              decisionTree:
                current:
                  pluginRef: least-queue-filter
                nextOnSuccessOrFailure:
                  decisionTree:
                    current:
                      pluginRef: least-kv-cache-filter
        nextOnFailure:
          decisionTree:
            current:
              pluginRef: least-queue-filter
            nextOnSuccessOrFailure:
              decisionTree:
                current:
                  pluginRef: lora-affinity-filter
                nextOnSuccessOrFailure:
                  decisionTree:
                    current:
                      pluginRef: least-kv-cache-filter
    - type: random-picker
      parameters:
        maxNumOfEndpoints: 1
    - type: single-profile-handler
    schedulingProfiles:
    - name: default
      plugins:
      - pluginRef: low-latency-filter
      - pluginRef: random-picker
  plugins-v2.yaml: |
    apiVersion: inference.networking.x-k8s.io/v1alpha1
    kind: EndpointPickerConfig
    plugins:
    - type: queue-scorer
    - type: kv-cache-scorer
    - type: prefix-cache-scorer
      parameters:
        hashBlockSize: 64
        maxPrefixBlocksToMatch: 256
        lruCapacityPerServer: 31250
    - type: max-score-picker
      parameters:
        maxNumOfEndpoints: 1
    - type: single-profile-handler
    schedulingProfiles:
    - name: default
      plugins:
      - pluginRef: queue-scorer
        weight: 1
      - pluginRef: kv-cache-scorer
        weight: 1
      - pluginRef: prefix-cache-scorer
        weight: 1
      - pluginRef: max-score-picker
EOF
```

### Direct traffic to the new inference pool
By configuring an **HTTPRoute**, as shown below, you can incrementally split traffic between the original `vllm-llama3-8b-instruct` and new `vllm-llama3-8b-instruct-new`.

```bash
kubectl edit httproute llm-route
```

Change the backendRefs list in HTTPRoute to match the following:


```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: llm-route
spec:
  parentRefs:
    - group: gateway.networking.k8s.io
      kind: Gateway
      name: inference-gateway
  rules:
    - backendRefs:
        - group: inference.networking.k8s.io
          kind: InferencePool
          name: vllm-llama3-8b-instruct
          weight: 90
        - group: inference.networking.k8s.io
          kind: InferencePool
          name: vllm-llama3-8b-instruct-new
          weight: 10
      matches:
        - path:
            type: PathPrefix
            value: /
```

The above configuration means one in every ten requests should be sent to the new version. Try it out:

1. Get the gateway IP:
```bash
IP=$(kubectl get gateway/inference-gateway -o jsonpath='{.status.addresses[0].value}'); PORT=80
```

2. Send a few requests as follows:
```bash
curl -i ${IP}:${PORT}/v1/completions -H 'Content-Type: application/json' -d '{
"model": "food-review",
"prompt": "Write as if you were a critic: San Francisco",
"max_tokens": 100,
"temperature": 0
}'
```

### Finish the rollout


Modify the HTTPRoute to direct 100% of the traffic to the latest version of the InferencePool.

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: llm-route
spec:
  parentRefs:
    - group: gateway.networking.k8s.io
      kind: Gateway
      name: inference-gateway
  rules:
    - backendRefs:
        - group: inference.networking.k8s.io
          kind: InferencePool
          name: vllm-llama3-8b-instruct-new
          weight: 100
      matches:
        - path:
            type: PathPrefix
            value: /
```

### Delete old version of InferencePool, InferenceModel and Endpoint Picker Extension
```shell
kubectl delete InferenceModel food-review --ignore-not-found
kubectl delete Deployment vllm-llama3-8b-instruct --ignore-not-found
kubectl delete ConfigMap vllm-llama3-8b-instruct-adapters --ignore-not-found
kubectl delete InferencePool vllm-llama3-8b-instruct --ignore-not-found
kubectl delete Deployment vllm-llama3-8b-instruct-epp --ignore-not-found
kubectl delete Service vllm-llama3-8b-instruct-epp --ignore-not-found
```

With this, all requests should be served by the new Inference Pool.
