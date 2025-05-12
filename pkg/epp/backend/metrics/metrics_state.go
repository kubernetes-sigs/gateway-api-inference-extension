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

package metrics

import (
	"fmt"
	"time"
)

// newMetricsState initializes a new MetricsState and returns its pointer.
func newMetricsState() *MetricsState {
	return &MetricsState{
		ActiveModels:  make(map[string]int),
		WaitingModels: make(map[string]int),
	}
}

// MetricsState holds the latest state of the metrics that were scraped from a pod.
type MetricsState struct {
	// ActiveModels is a set of models(including LoRA adapters) that are currently cached to GPU.
	ActiveModels  map[string]int
	WaitingModels map[string]int
	// MaxActiveModels is the maximum number of models that can be loaded to GPU.
	MaxActiveModels         int
	RunningQueueSize        int
	WaitingQueueSize        int
	KVCacheUsagePercent     float64
	KvCacheMaxTokenCapacity int

	// UpdateTime record the last time when the metrics were updated.
	UpdateTime time.Time
}

// String returns a string with all MetricState information
func (s *MetricsState) String() string {
	if s == nil {
		return ""
	}
	return fmt.Sprintf("%+v", *s)
}

// Clone creates a copy of MetricsState and returns its pointer.
// Clone returns nil if the object being cloned is nil.
func (s *MetricsState) Clone() *MetricsState {
	if s == nil {
		return nil
	}
	activeModels := make(map[string]int, len(s.ActiveModels))
	for key, value := range s.ActiveModels {
		activeModels[key] = value
	}
	waitingModels := make(map[string]int, len(s.WaitingModels))
	for key, value := range s.WaitingModels {
		waitingModels[key] = value
	}
	return &MetricsState{
		ActiveModels:            activeModels,
		WaitingModels:           waitingModels,
		MaxActiveModels:         s.MaxActiveModels,
		RunningQueueSize:        s.RunningQueueSize,
		WaitingQueueSize:        s.WaitingQueueSize,
		KVCacheUsagePercent:     s.KVCacheUsagePercent,
		KvCacheMaxTokenCapacity: s.KvCacheMaxTokenCapacity,
		UpdateTime:              s.UpdateTime,
	}
}
