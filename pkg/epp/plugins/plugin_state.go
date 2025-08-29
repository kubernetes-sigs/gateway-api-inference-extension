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

package plugins

import (
	"context"
	"fmt"
	"sync"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	// stalenessThreshold defines the threshold for considering data as stale.
	// if data of a request hasn't been read/write in the last "stalenessThreshold", it is considered as stale data
	// and will be cleaned in the next cleanup cycle.
	stalenessThreshold = time.Minute * 5
	// cleanupInterval defines the periodic interval that the cleanup go routine uses to check for stale data.
	cleanupInterval = time.Minute
)

// NewPluginState initializes a new PluginState and returns its pointer.
func NewPluginState(ctx context.Context) *PluginState {
	pluginState := &PluginState{}
	go pluginState.cleanup(ctx)
	return pluginState
}

// PluginState provides a mechanism for plugins to store and retrieve arbitrary data by multiple extension points.
// Data stored by the plugin in one extension point can be written, read or altered by another extension point.
// The data stored in PluginState is always stored in the context of a given request.
// If the data hasn't been accessed during "stalenessThreshold", it is cleaned by a cleanup internal mechanism.
//
// Note: PluginState uses a sync.Map to back the storage, because it is thread safe.
// It's aimed to optimize for the "write once and read many times" scenarios.
type PluginState struct {
	// key: RequestID, value: sync.Map[StateKey]StateData
	storage sync.Map
	// key: RequestID, value: time.Time
	requestToLastAccessTime sync.Map
}

// Read retrieves data with the given "key" in the context of "requestID" from PluginState.
// If the key is not present, ErrNotFound is returned.
func (s *PluginState) Read(requestID string, key StateKey) (StateData, error) {
	s.requestToLastAccessTime.Store(requestID, time.Now())
	stateMap, ok := s.storage.Load(requestID)
	if !ok {
		return nil, ErrNotFound
	}

	stateData := stateMap.(*sync.Map)
	if value, ok := stateData.Load(key); ok {
		return value.(StateData), nil
	}

	return nil, ErrNotFound
}

// Write stores the given "val" in PluginState with the given "key" in the context of the given "requestID".
func (s *PluginState) Write(requestID string, key StateKey, val StateData) {
	s.requestToLastAccessTime.Store(requestID, time.Now())
	var stateData *sync.Map
	stateMap, ok := s.storage.Load(requestID)
	if ok {
		stateData = stateMap.(*sync.Map)
	} else {
		stateData = &sync.Map{}
	}

	stateData.Store(key, val)

	s.storage.Store(requestID, stateData)
}

// Delete deletes data associated with the given requestID.
// It is possible to call Delete explicitly when the handling of a request is completed
// or alternatively, if the request failed during its processing, a cleanup goroutine will
// clean data of stale requests.
func (s *PluginState) Delete(requestID string) {
	s.storage.Delete(requestID)
	s.requestToLastAccessTime.Delete(requestID)
}

// cleanup periodically deletes data associated with the given requestID.
func (s *PluginState) cleanup(ctx context.Context) {
	ticker := time.NewTicker(cleanupInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			log.FromContext(ctx).V(logutil.DEFAULT).Info("Shutting down plugin state cleanup")
			return
		case <-ticker.C:
			s.cleanStaleRequests()
		}
	}
}

// cleanStaleRequests iterates through all requests and removes those that haven't been
// accessed for longer than stalenessThreshold. This operation is safe to run concurrently
// with other operations on the PluginState.
func (s *PluginState) cleanStaleRequests() {
	s.requestToLastAccessTime.Range(func(k, v any) bool {
		requestID := k.(string)
		lastAccessTime := v.(time.Time)
		if time.Since(lastAccessTime) > stalenessThreshold {
			s.Delete(requestID) // cleanup stale requests (this is safe in sync.Map)
		}
		return true
	})
}

// ReadPluginStateKey retrieves data with the given key from PluginState and asserts it to type T.
// Returns an error if the key is not found or the type assertion fails.
func ReadPluginStateKey[T StateData](state *PluginState, requestID string, key StateKey) (T, error) {
	var zero T

	raw, err := state.Read(requestID, key)
	if err != nil {
		return zero, err
	}

	val, ok := raw.(T)
	if !ok {
		return zero, fmt.Errorf("unexpected type for key %q: got %T", key, raw)
	}

	return val, nil
}
