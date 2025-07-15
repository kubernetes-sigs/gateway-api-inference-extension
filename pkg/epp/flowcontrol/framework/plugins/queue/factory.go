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

// Package queue defines interfaces and implementations for various queue data structures used by the FlowController.
package queue

import (
	"fmt"
	"sync"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
)

type RegisteredQueueName string

// QueueConstructor defines the function signature for creating a `framework.SafeQueue`.
type QueueConstructor func(comparator framework.ItemComparator) (framework.SafeQueue, error)

var (
	// mu guards the registration maps.
	mu sync.RWMutex
	// RegisteredQueues stores the constructors for all registered queues.
	RegisteredQueues = make(map[RegisteredQueueName]QueueConstructor)
)

// MustRegisterQueue registers a queue constructor, and panics if the name is
// already registered.
// This is intended to be called from init() functions.
func MustRegisterQueue(name RegisteredQueueName, constructor QueueConstructor) {
	mu.Lock()
	defer mu.Unlock()
	if _, ok := RegisteredQueues[name]; ok {
		panic(fmt.Sprintf("framework.SafeQueue already registered with name %q", name))
	}
	RegisteredQueues[name] = constructor
}

// NewQueueFromName creates a new SafeQueue given its registered name and the `framework.ItemComparator` that will be
// optionally used to configure the queue (provided it declares `framework.CapabilityPriorityConfigurable`).
// This is called by the `registry.FlowRegistry` during initialization of a flow's `ports.ManagedQueue`.
func NewQueueFromName(name RegisteredQueueName, comparator framework.ItemComparator) (framework.SafeQueue, error) {
	mu.RLock()
	defer mu.RUnlock()
	constructor, ok := RegisteredQueues[name]
	if !ok {
		return nil, fmt.Errorf("no framework.SafeQueue registered with name %q", name)
	}
	return constructor(comparator)
}
