/*
Copyright 2026 The Kubernetes Authors.

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

package eviction

import "sync"

// EvictionRegistry is a shared registry that maps request IDs to eviction channels.
// It bridges the eviction plugin (which decides what to evict) and the ext_proc Process()
// goroutine (which owns the stream needed to send ImmediateResponse).
//
// Lifecycle:
//   - PreRequest: plugin creates an eviction channel and registers it via Register().
//   - Process(): after HandleRequest returns, looks up the channel via Get() and selects on it.
//   - EvictN: evictor closes the channel via the EvictionItem.EvictCh reference.
//   - Process() defer: removes the channel via Deregister().
//
// All methods are goroutine-safe.
type EvictionRegistry struct {
	mu       sync.RWMutex
	channels map[string]chan struct{} // requestID → eviction channel
}

// NewEvictionRegistry creates a new EvictionRegistry.
func NewEvictionRegistry() *EvictionRegistry {
	return &EvictionRegistry{
		channels: make(map[string]chan struct{}),
	}
}

// Register stores an eviction channel for the given request ID.
func (r *EvictionRegistry) Register(requestID string, ch chan struct{}) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.channels[requestID] = ch
}

// Get returns the eviction channel for the given request ID, or nil if not found.
func (r *EvictionRegistry) Get(requestID string) chan struct{} {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.channels[requestID]
}

// Deregister removes the eviction channel for the given request ID.
func (r *EvictionRegistry) Deregister(requestID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.channels, requestID)
}
