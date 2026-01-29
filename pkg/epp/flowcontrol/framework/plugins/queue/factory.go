/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package queue provides the factory and registration mechanism for all SafeQueue implementations.
// It allows new queues to be added to the system and instantiated by name.
package queue

import (
	"fmt"
	"sync"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
)

// RegisteredQueueName is the unique name under which a queue is registered.
type RegisteredQueueName string

// QueueConstructor defines the function signature for creating a SafeQueue.
type QueueConstructor func(policy flowcontrol.OrderingPolicy) (contracts.SafeQueue, error)

var (
	// mu guards the registration map.
	mu sync.RWMutex
	// RegisteredQueues stores the constructors for all registered queues.
	RegisteredQueues = make(map[RegisteredQueueName]QueueConstructor)
)

// MustRegisterQueue registers a queue constructor, and panics if the name is already registered.
// This is intended to be called from init() functions.
func MustRegisterQueue(name RegisteredQueueName, constructor QueueConstructor) {
	mu.Lock()
	defer mu.Unlock()
	if _, ok := RegisteredQueues[name]; ok {
		panic(fmt.Sprintf("SafeQueue already registered with name %q", name))
	}
	RegisteredQueues[name] = constructor
}

// NewQueueFromName creates a new SafeQueue given its registered name and the OrderingPolicy that will be optionally
// used to configure the queue (provided it declares CapabilityPriorityConfigurable).
// This is called by the FlowRegistry during initialization of a flow's ManagedQueue.
func NewQueueFromName(name RegisteredQueueName, policy flowcontrol.OrderingPolicy) (contracts.SafeQueue, error) {
	mu.RLock()
	defer mu.RUnlock()
	constructor, ok := RegisteredQueues[name]
	if !ok {
		return nil, fmt.Errorf("no SafeQueue registered with name %q", name)
	}
	return constructor(policy)
}
