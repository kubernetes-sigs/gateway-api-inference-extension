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

package datalayer

import (
	"errors"
	"fmt"
)

var errInvalidRuntimeState = errors.New("invalid datalayer runtime state")

// RuntimeState represents the lifecycle state of Runtime.
type RuntimeState int64

const (
	// StateInitial is the initial state when Runtime is created.
	StateInitial RuntimeState = iota
	// StateConfiguring indicates Runtime is currently being configured.
	StateConfiguring
	// StateConfigured indicates Runtime has been configured with sources.
	StateConfigured
	// StateStarting indicates Runtime is currently starting.
	StateStarting
	// StateStarted indicates Runtime is actively managing collectors and reconcilers.
	StateStarted
	// StateStopping indicates Runtime is currently stopping.
	StateStopping
	// StateStopped indicates Runtime has been stopped.
	StateStopped
	// StateError indicates Runtime is in an error state.
	StateError
)

// String returns the string representation of RuntimeState.
func (s RuntimeState) String() string {
	switch s {
	case StateInitial:
		return "initial"
	case StateConfiguring:
		return "configuring"
	case StateConfigured:
		return "configured"
	case StateStarting:
		return "starting"
	case StateStarted:
		return "started"
	case StateStopping:
		return "stopping"
	case StateStopped:
		return "stopped"
	case StateError:
		return "error"
	default:
		return "unknown"
	}
}

// transition attempts to atomically transition from expectedState to newState.
// Returns nil on success, or an error if the transition fails.
// If the transition fails, sets the state to StateError.
func (r *Runtime) transition(expectedState, newState RuntimeState) error {
	if r.state.CompareAndSwap(int64(expectedState), int64(newState)) {
		return nil
	}
	r.setError()
	current := r.State()
	return fmt.Errorf("%w: cannot transition from %s to %s", errInvalidRuntimeState, current, newState)
}

// setError sets the state to StateError atomically.
func (r *Runtime) setError() {
	r.state.Store(int64(StateError))
}
