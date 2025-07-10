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

// Package dispatch provides the factory and registration mechanism for all `framework.IntraFlowDispatchPolicy`
// implementations.
// It allows new policies to be added to the system and instantiated by name.
package dispatch

import (
	"fmt"
	"sync"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
)

// RegisteredPolicyName is the unique name under which a policy is registered.
type RegisteredPolicyName string

// PolicyConstructor defines the function signature for creating a `framework.IntraFlowDispatchPolicy`.
type PolicyConstructor func() (framework.IntraFlowDispatchPolicy, error)

var (
	// mu guards the registration map.
	mu sync.RWMutex
	// RegisteredPolicies stores the constructors for all registered policies.
	RegisteredPolicies = make(map[RegisteredPolicyName]PolicyConstructor)
)

// MustRegisterPolicy registers a policy constructor, and panics if the name is already registered.
// This is intended to be called from the `init()` function of a policy implementation.
func MustRegisterPolicy(name RegisteredPolicyName, constructor PolicyConstructor) {
	mu.Lock()
	defer mu.Unlock()
	if _, ok := RegisteredPolicies[name]; ok {
		panic(fmt.Sprintf("IntraFlowDispatchPolicy already registered with name %q", name))
	}
	RegisteredPolicies[name] = constructor
}

// NewPolicyFromName creates a new `IntraFlowDispatchPolicy` given its registered name.
// This is called by the `registry.FlowRegistry` when configuring a flow.
func NewPolicyFromName(name RegisteredPolicyName) (framework.IntraFlowDispatchPolicy, error) {
	mu.RLock()
	defer mu.RUnlock()
	constructor, ok := RegisteredPolicies[name]
	if !ok {
		return nil, fmt.Errorf("no IntraFlowDispatchPolicy registered with name %q", name)
	}
	return constructor()
}
