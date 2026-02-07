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

package saturation

import (
	"context"
	"slices"
	"sync"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

const (
	// BatchEvictorType is the type of the batch evictor plugin.
	BatchEvictorType = "batch-evictor"
)

// evictionCandidate represents a single item scheduled for potential eviction.
type evictionCandidate struct {
	queue      flowcontrol.EvictableQueue
	item       flowcontrol.QueueItemAccessor
	priority   int
	usageLimit float64
}

// BatchEvictor collects eviction candidates during the dispatch cycle and evicts them in batch
// at the end of the cycle. Only negative-priority items are evicted.
type BatchEvictor struct {
	name string
	mu   sync.Mutex
	// candidates stores items scheduled for eviction during the current dispatch cycle
	candidates []evictionCandidate
}

var _ flowcontrol.Evictor = &BatchEvictor{}

// NewBatchEvictor creates a new batch evictor that only evicts negative-priority items.
func NewBatchEvictor() *BatchEvictor {
	return &BatchEvictor{
		name:       BatchEvictorType,
		candidates: make([]evictionCandidate, 0, 100),
	}
}

// TypedName returns the type and name tuple of this plugin instance.
func (e *BatchEvictor) TypedName() plugin.TypedName {
	return plugin.TypedName{
		Type: BatchEvictorType,
		Name: e.name,
	}
}

// ScheduleEvictionCandidate registers a gated item as a candidate for eviction.
// Only items with negative priority (priority < 0) are scheduled.
func (e *BatchEvictor) ScheduleEvictionCandidate(
	ctx context.Context,
	candidate flowcontrol.QueueItemAccessor,
	queue flowcontrol.EvictableQueue,
	priority int,
	usageLimit float64,
) {
	// Only schedule negative-priority items for eviction
	if priority >= 0 {
		return
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	e.candidates = append(e.candidates, evictionCandidate{
		queue:      queue,
		item:       candidate,
		priority:   priority,
		usageLimit: usageLimit,
	})
}

// ProcessScheduled evicts all scheduled negative-priority candidates.
// Candidates are evicted in LIFO order (most recently scheduled first) within each priority level.
func (e *BatchEvictor) ProcessScheduled(ctx context.Context) (int, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	evicted := 0

	// Evict in reverse order
	for _, candidate := range slices.Backward(e.candidates) {
		// Attempt to remove the item from its queue
		if _, err := candidate.queue.Remove(candidate.item.Handle()); err != nil {
			// Item may have already been dispatched or removed - not an error
			continue
		}
		evicted++
	}

	// Clear candidates for next cycle
	e.candidates = e.candidates[:0]

	return evicted, nil
}
