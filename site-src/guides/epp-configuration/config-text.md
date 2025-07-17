# Configuring Plugins via text

The set of lifecycle hooks (plugins) that are used by the Inference Gateway (IGW) is determined by how 
it is configured. The IGW can be configured in several ways, either by code or via text.

If configured by code either a set of predetermined environment variables must be used or one must
fork the IGW and change code.

A simpler way to congigure the IGW is to use a text based configuration. This text is in YAML format
and can either be in a file or specified in-line as a parameter. The configuration defines the set of
plugins to be instantiated along with their parameters. Each plugin can also be given a name, enabling
the same plugin type to be instantiated multiple times, if needed. Also defined is a set of
SchedulingProfiles, which determine the set of plugins to be used when scheduling a request. The set
of plugins instantiated must also include a Profile Handler, which determines which SchedulingProfiles
will be used for a particular request.

It should be noted that while the configuration text looks like a Kubernetes Custom Resource, it is
**NOT** a Kubernetes Custom Resource. Kubernetes infrastructure is used to load the configuration
text and in the future will also help in versioning the text.

It should also be noted that even when the configuration text is loaded from a file, it is loaded at
the Endpoint-Picker's (EPP) startup and changes to the file at runtime are ignored.

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

The plugins section defines the set of plugins that will be instantiated and their parameters. Each entry in this section
has the following form:
```yaml
- name: aName
  type: a-type
  parameters:
    parm1: val1
    parm2: val2
```
The fields in a plugin entry are:
- *name* which is optional, provides a name by which the plugin instance can be referenced. If this
field is omitted, the plugin's type will be used as its name.<br>
- *type* specifies the type of the plugin to be instantiated.<br>
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
  - *weight* is the weight to be used if the referenced plugin is a scorer.

A complete configuration might look like this:
```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: prefix-cache
  parameters:
    hashBlockSize: 5
    maxPrefixBlocksToMatch: 256
    lruCapacityPerServer: 31250
- type: max-score
- type: single-profile
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: max-score
  - pluginRef: prefix-cache
    weight: 50
```

If the configuration is in a file, the EPP command line argument `--configFile`
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
        - -poolName
        - "${POOL_NAME}"
        ...
        - --configFile
        - "/etc/epp/epp-config.yaml"
```

If the configuration is passed as in-line text the EPP command line argument `--configText`
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
        - -poolName
        - "${POOL_NAME}"
        ...
        - --configText
        - |
          apiVersion: inference.networking.x-k8s.io/v1alpha1
          kind: EndpointPickerConfig
          plugins:
          - type: prefix-cache
            parameters:
              hashBlockSize: 5
              maxPrefixBlocksToMatch: 256
              lruCapacityPerServer: 31250
          - type: max-score
          - type: single-profile
          schedulingProfiles:
          - name: default
            plugins:
            - pluginRef: max-score
            - pluginRef: prefix-cache
              weight: 50
```

## Plugin Configuration

This section describes how to setup the various plugins that are available with the IGW.

**SingleProfileHandler**<br>
Selects a single profile which is always the primary profile.<br>
*Type*: single-profile<br>
*Parameters*: none<br>

**LeastKVCacheFilter**<br>
Finds the max and min KV cache of all pods, divides the whole range (max-min) by the
number of pods, and finds the pods that fall into the first range.<br>
*Type*: least-KV-cache<br>
*Parameters*: none<br>

**LeastQueueFilter**<br>
Finds the max and min queue size of all pods, divides the whole range (max-min) by the
number of pods, and finds the pods that fall into the first range.<br>
*Type*: least-queue<br>
*Parameters*: none<br>

**LoraAffinityFilter**<br>
Implements a pod selection strategy that when the use of a LoRA adapter is requested, prioritizes pods 
that are believed to have the specific LoRA adapter loaded. It also allows for load balancing through
some randomization.<br>
*Type*: lora-affinity<br>
*Parameters*:<br>
\- `threshold` a probability threshold to sometimes select pods that don't seem to have the LoRA
    adapter loaded to enable load balancing. If not specified defaults to `0.999`<br>

**LowQueueFilter**<br>
Filters out pods who's waiting queue size is greater than the specified theshold.<br>
*Type*: low-queue<br>
*Parameters*:<br>
\- `threshold` the waiting queue threshold. If not specified defaults to `128`<br>

**PrefixCachePlugin**<br>
Scores pods based on the amount of the prompt is believed to be in the pod's KvCache.<br>
*Type*: prefix-cache<br>
*Parameters*:<br>
\- `hashBlockSize` specified the size of the blocks to break up the input prompt when
   calculating the block hashes. If not specified defaults to `64`<br>
\- `maxPrefixBlocksToMatch` specifies the maximum number of prefix blocks to match. If
   not specified defaults to `256`<br>
\- `lruCapacityPerServer` specifies the capacity of the LRU indexer in number of entries
   per server (pod). If not specified defaults to `31250`<br>

**MaxScorePicker**<br>
Picks the pod with the maximum score from the list of candidates.<br>
*Type*: max-score<br>
*Parameters*: none<br>

**RandomPicker**<br>
Picks a random pod from the list of candidates.<br>
*Type*: random<br>
*Parameters*: none<br>

**KvCacheScorer**<br>
Scores the candidate pods based on their KV cache utilization.<br>
*Type*: kv-cache<br>
*Parameters*: none<br>

**QueueScorer**<br>
Scores list of candidate pods based on the pod's waiting queue size. The lower the
waiting queue size the pod has, the higher the score it will get (since it's more
available to serve new request).<br>
*Type*: queue<br>
*Parameters*: none<br>