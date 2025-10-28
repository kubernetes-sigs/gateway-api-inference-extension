# Configuring Plugins via YAML

The set of lifecycle hooks (plugins) that are used by the Inference Gateway (IGW) is determined by how
it is configured. The IGW is primarily configured via a configuration file.

The YAML file can either be specified as a path to a file or in-line as a parameter. The configuration defines the set of
plugins to be instantiated along with their parameters. Each plugin can also be given a name, enabling
the same plugin type to be instantiated multiple times, if needed (such as when configuring multiple scheduling profiles).

Also defined is a set of SchedulingProfiles, which determine the set of plugins to be used when scheduling a request.
If no scheduling profile is specified, a default profile, named `default` will be added and will reference all of the
instantiated plugins.

The set of plugins instantiated can include a Profile Handler, which determines which SchedulingProfiles
will be used for a particular request. A Profile Handler must be specified, unless the configuration only
contains one profile, in which case the `SingleProfileHandler` will be used.

In addition, the set of instantiated plugins can also include a picker, which chooses the actual pod to which
the request is scheduled after filtering and scoring. If one is not referenced in a SchedulingProfile, an
instance of `MaxScorePicker` will be added to the SchedulingProfile in question.

***NOTE***: While the configuration text looks like a Kubernetes CRD, it is
**NOT** a Kubernetes CRD. Specifically, the config is not reconciled upon, and is only read on startup.
This behavior is intentional, as augmenting the scheduling config without redeploying the EPP is not supported.

The configuration text has the following form:
```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- ....
- ....
schedulingProfiles:
- ....
- ....
```

The first two lines of the configuration are constant and must appear as is.

The plugins section defines the set of plugins that will be instantiated and their parameters.
Each entry in this section has the following form:

```yaml
- name: aName
  type: a-type
  parameters:
    parm1: val1
    parm2: val2
```

The fields in a plugin entry are:

- *name* which is optional, provides a name by which the plugin instance can be referenced. If this
field is omitted, the plugin's type will be used as its name.
- *type* specifies the type of the plugin to be instantiated.
- *parameters* which is optional, defines the set of parameters used to configure the plugin in question.
The actual set of parameters varies from plugin to plugin.

The schedulingProfiles section defines the set of scheduling profiles that can be used in scheduling
requests to pods. The number of scheduling profiles one defines, depends on the use case. For simple
serving of requests, one is enough. For disaggregated prefill, two profiles are required. Each entry
in this section has the following form:

```yaml
- name: aName
  plugins:
  - pluginRef: plugin1
  - pluginRef: plugin2
    weight: 50
```

The fields in a schedulingProfile entry are:

- *name* specifies the scheduling profile's name.
- *plugins* specifies the set of plugins to be used when this scheduling profile is chosen for a request.
Each entry in the schedulingProfile's plugins section has the following fields:
  - *pluginRef* is a reference to the name of the plugin instance to be used
  - *weight* is the weight to be used if the referenced plugin is a scorer. If omitted, a weight of one
    will be used.

A complete configuration might look like this:
```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: prefix-cache-scorer
  parameters:
    blockSize: 5
    maxPrefixBlocksToMatch: 256
    lruCapacityPerServer: 31250
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: prefix-cache-scorer
    weight: 50
```

If the configuration is in a file, the EPP command line argument `--config-file`
should be used to specify the full path of the file in question. For example:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${EPP_NAME}
  ...
spec:
  ...
  template:
    ...
    spec:
      ...
      containers:
      - name: epp
        image: ghcr.io/llm-d/llm-d-inference-scheduler:latest
        imagePullPolicy: IfNotPresent
        args:
        - --pool-name
        - "${POOL_NAME}"
        ...
        - --config-file
        - "/etc/epp/epp-config.yaml"
```

If the configuration is passed as in-line text the EPP command line argument `--config-text`
should be used. For example:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${EPP_NAME}
  ...
spec:
  ...
  template:
    ...
    spec:
      ...
      containers:
      - name: epp
        image: ghcr.io/llm-d/llm-d-inference-scheduler:latest
        imagePullPolicy: IfNotPresent
        args:
        - --pool-name
        - "${POOL_NAME}"
        ...
        - --config-text
        - |
          apiVersion: inference.networking.x-k8s.io/v1alpha1
          kind: EndpointPickerConfig
          plugins:
          - type: prefix-cache-scorer
            parameters:
              blockSize: 5
              maxPrefixBlocksToMatch: 256
              lruCapacityPerServer: 31250
          schedulingProfiles:
          - name: default
            plugins:
            - pluginRef: prefix-cache-scorer
              weight: 50
```

The EPP configuration in the above two examples are equivalent to the following configuration after
defaults have been applied:

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: prefix-cache-scorer
  parameters:
    blockSize: 5
    maxPrefixBlocksToMatch: 256
    lruCapacityPerServer: 31250
- type: single-profile-handler
- type: max-score-picker
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: prefix-cache-scorer
    weight: 50
  -pluginRef: max-score-picker
```

## Plugin Configuration

This section describes how to setup the various plugins that are available with the IGW.

#### **SingleProfileHandler**

Selects a single profile which is always the primary profile.

- *Type*: single-profile-handler
- *Parameters*: none

#### **PrefixCacheScorer**

Scores pods based on the amount of the prompt is believed to be in the pod's KvCache.

- *Type*: prefix-cache-scorer
- *Parameters*:
  - `blockSize` specified the size of the blocks to break up the input prompt when
    calculating the block hashes. If not specified defaults to `64`
  - `maxPrefixBlocksToMatch` specifies the maximum number of prefix blocks to match. If
   not specified defaults to `256`
  - `lruCapacityPerServer` specifies the capacity of the LRU indexer in number of entries
    per server (pod). If not specified defaults to `31250`

#### **LoRAAffinityScorer**

Scores pods based on whether the requested LoRA adapter is already loaded in the pod's HBM, or if
the pod is ready to load the LoRA on demand.

- *Type*: lora-affinity-scorer
- *Parameters*: none

#### **MaxScorePicker**

Picks the pod with the maximum score from the list of candidates. This is the default picker plugin
if not specified.

- *Type*: max-score-picker
- *Parameters*: 
  - `maxNumOfEndpoints`: Maximum number of endpoints to pick from the list of candidates, based on
    the scores of those endpoints. If not specified defaults to `1`.

#### **RandomPicker**

Picks a random pod from the list of candidates.

- *Type*: random-picker
- *Parameters*: 
  - `maxNumOfEndpoints`: Maximum number of endpoints to pick from the list of candidates. If not
    specified defaults to `1`.

#### **WeightedRandomPicker**

Picks pod(s) from the list of candidates based on weighted random sampling using A-Res algorithm.

- *Type*: weighted-random-picker
- *Parameters*:
  - `maxNumOfEndpoints`: Maximum number of endpoints to pick from the list of candidates. If not
    specified defaults to `1`.

#### **KvCacheScorer**

Scores the candidate pods based on their KV cache utilization.

- *Type*: kv-cache-utilization-scorer
- *Parameters*: none

#### **QueueScorer**

Scores list of candidate pods based on the pod's waiting queue size. The lower the
waiting queue size the pod has, the higher the score it will get (since it's more
available to serve new request).

- *Type*: queue-scorer
- *Parameters*: none


#### **LoraAffinityScorer**

Scores list of candidate pods based on the LoRA adapters loaded on the pod. 
Pods with the adapter already loaded or able to be actively loaded will be 
scored higher (since it's more available to serve new request).

- *Type*: lora-affinity-scorer
- *Parameters*: none
