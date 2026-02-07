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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

const (
	// NoOpEvictorType is the type of the no-op evictor plugin.
	NoOpEvictorType = "noop-evictor"
)

// NoOpEvictor is a stub implementation that performs no eviction.
// This is useful as a default/fallback or when eviction is disabled.
type NoOpEvictor struct {
	name string
}

var _ flowcontrol.Evictor = &NoOpEvictor{}

// NewNoOpEvictor creates a new no-op evictor.
func NewNoOpEvictor() *NoOpEvictor {
	return &NoOpEvictor{name: NoOpEvictorType}
}

// TypedName returns the type and name tuple of this plugin instance.
func (e *NoOpEvictor) TypedName() plugin.TypedName {
	return plugin.TypedName{
		Type: NoOpEvictorType,
		Name: e.name,
	}
}

// ScheduleEviction does nothing (no-op).
func (e *NoOpEvictor) ScheduleEvictionCandidate(ctx context.Context, candidate flowcontrol.QueueItemAccessor, queue flowcontrol.EvictableQueue, priority int, usageLimit float64) {
	// No-op: don't schedule anything for eviction
}

// ProcessScheduled always returns 0 (no requests evicted).
func (e *NoOpEvictor) ProcessScheduled(ctx context.Context) (int, error) {
	return 0, nil
}
