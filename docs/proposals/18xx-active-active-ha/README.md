# Active-Active HA Deployment Architecture Proposal
Author(s): @delavet

## Proposal Status
_**Draft**_

## Summary
This proposal addresses the need for [Active-Active High Availability (HA) deployment](https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/1406) in the Gateway API Inference Extension (GAIE) project. The goal is to enable multiple EPP (Endpoint Picker) instances to run concurrently in an active-active configuration without external dependencies like Redis or etcd, while ensuring fast convergence and eventual consistency of distributed state across the cluster.

Key challenges that need to be addressed include:

+ Synchronizing distributed state without external dependencies (like Redis/etcd)
+ Ensuring fast convergence and eventual consistency
+ Managing state data like prefix-cache state across multiple instances
+ Providing graceful failure recovery and state re-convergence

## Goals
+ Enable state data sharing among multiple EPP instances
+ Achieve fast convergence with eventual consistency using CRDT-based state synchronization
+ Ensure no additional latency from external state stores like Redis
+ Maintain state consistency for critical components like prefix cache and queueing
+ Enable automatic discovery and joining of EPP cluster nodes

## Non-Goals
+ Implementing strong/linearizable consistency protocols (like Raft/etcd)
+ Achieving full accuracy during brief periods of inconsistency (soft drift acceptable)

## Proposal
### Overview
The Active-Active HA deployment solution introduces a distributed state synchronization layer using memberlist and CRDT (Conflict-Free Replicated Data Types) to maintain eventual consistency across multiple EPP instances. The architecture consists of several key components:

```plain
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   EPP Node 1    │    │   EPP Node 2    │    │   EPP Node N    │
│                 │    │                 │    │                 │
│  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │
│  │ Prefix    │  │    │  │ Prefix    │  │    │  │ Prefix    │  │
│  │ Cache     │  │    │  │ Cache     │  │    │  │ Cache     │  │
│  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │
│                 │    │                 │    │                 │
│  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │
│  │ Flow      │  │    │  │ Flow      │  │    │  │ Flow      │  │
│  │ Control   │  │    │  │ Control   │  │    │  │ Control   │  │
│  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │
│                 │    │                 │    │                 │
│  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │
│  │ Distributed│ │    │  │ Distributed│ │    │  │ Distributed│ │
│  │ State     │  │◄───┼──┤ State     │  │◄───┼──┤ State     │  │
│  │ Layer     │  │    │  │ Layer     │  │    │  │ Layer     │  │
│  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │
│        │        │    │        │        │    │        │        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                    ┌─────────────────┐
                    │ Memberlist      │
                    │ Gossip Protocol │
                    └─────────────────┘
```

The solution consists of three main architectural components:

1. **Distributed State Store**: A CRDT-based data store that maintains consistent state across all EPP nodes
2. **Memberlist with Gossip Protocol**: Enables peer-to-peer communication between EPP instances for state synchronization
3. **Kubernetes Service Discovery**: Enables automatic discovery of EPP instances in the kubernetes cluster

### Component Detail
#### 1. Distributed State Store
The distributed state store implements a CRDT (Conflict-Free Replicated Data Type) based approach to handle concurrent updates across multiple nodes. It will implement following data interface:

```go
type StateDataType string

const (
	HierarchicalMap StateDataType = "hierarchical-map"
	Counter         StateDataType = "counter"
)

type StateStore interface {
	RegisterState(namespace string, dataType StateDataType, callback func(op StateOps)) error
	// for HierarchicalMap state
	Get(namespace string, fields ...string) (any, bool, error)
	Set(namespace string, value any, fields ...string) error
	Delete(namespace string, fields ...string) error
	// for Counter state
	Increment(namespace string)
	Decrement(namespace string)
	Value(namespace string) int64
}
```

The StateStore divides different state data into different namespaces. State data comes in various types, and when using it, you can first register a namespace and its corresponding state data type with the StateStore via `RegisterState` (optionally passing in a callback to perform custom actions when the state data changes). Afterward, you can perform different kinds of operations based on the corresponding namespace and the type of state data.

+ If the type is `HierarchicalMap`, it can perform operations similar to those provided by `unstructured.Unstructured`.
+ If the type is `Counter`, you can increment or decrement the value of the counter and retrieve the corresponding counter value.

In a single-replica environment, the StateStore can easily implement a local version, so this will not be elaborated further. In an Active-Active HA deployment, the corresponding distributed implementation uses appropriate CRDT (Conflict-free Replicated Data Type) data structures:

**a) Multi-LWWRegister (Multiple Last-Write-Wins Register)**

Multi-LWWRegister Resolves conflicts between concurrent updates by timestamp and node ID，Provides a key-value store interface using LWWRegister for each stored value.

The LWWRegister is a CRDT that stores a value along with its timestamp and node ID to resolve concurrent update conflicts. When two nodes update the same key concurrently, the value with the later timestamp is chosen. If timestamps are the same, the node ID serves as a tie-breaker.

```go
type LWWRegister struct {
    Value     any    `json:"value"`     // Stores the actual value
    Timestamp int64  `json:"timestamp"` // Timestamp, used for comparing older/newer values
    NodeID    string `json:"node_id"`   // Node ID, used as tie breaker
}
```

Multi-LWWRegister is constructed by storing LWWRegisters in a hierarchical map, forming a distributed multi-layer nested map. It can provide the following data interfaces:

```go
type HierarchicalMap interface {
    Get(fields ...string) (any, bool, error)
    Set(value any, fields ...string) error
    Delete(fields ...string) error
}
```

The interface supports hierarchical key structures for organizing state data. For example: `prefix-cache/hash/podName` for prefix cache state. This interface is like `unstructured.Unstructured`, and should meet the requirements of most components that need to store and retrieve state data.



**b) PNCounter**

PNCounter can implement a distributed counter by combining two GCOUNTERs. GCounter, on the other hand, achieves distributed consistency by counting each node separately.

```go
type GCounter struct {
	values map[string]int
}

type PNCounter struct {
    positives, negatives *GCounter
}
```

PNCounter can implement the following data interface:

```go
type Counter interface {
    Increment()
    Decrement()
    Value() int64
}
```

This can support some state data with countable properties, such as queue length, etc.

#### 2. Memberlist with Gossip Protocol
The [memberlist](https://github.com/hashicorp/memberlist) handles:

+ Node discovery and cluster membership
+ Failure detection and handling
+ Message broadcasting using TransmitLimitedQueue for synchronizing states between nodes

Node configuration includes:

+ Bind address and port for gossip communication
+ Kubernetes label selector to find other EPP nodes
+ Cluster namespace for service discovery

#### 3. Integration with Existing Components
The distributed state store integrates with key EPP components which relies on stateful data:

+ **Prefix Cache Plugin**: Uses distributed state to track which model server instances have cached specific prompt prefixes
+ **Flow Control**: Uses distributed state to track the state of request flows



### Implementation Plan
#### Phase 1: Core Distributed State Infrastructure
+ Introduce memberlist dependency
+ Implement  CRDT structures such as LWWRegister / PNCounter
+ Implement memberlist gossip protocol for node communication
+ Create distributed state store with above interface
+ Implement Kubernetes-based node discovery

#### Phase 2: Integration with Critical Components
+ Refactor prefix cache plugin and flow control layer to use distributed state store
+ Implement performance benchmarks to compare single vs multi-node performance

#### Phase 3: Testing and Observability
+ Add end-to-end tests for multi-node scenarios
+ Implement observability features