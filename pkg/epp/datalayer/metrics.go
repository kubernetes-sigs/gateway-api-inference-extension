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
	"fmt"
	"time"
)

// Metrics holds the latest metrics snapshot scraped from a pod.
type Metrics struct {
	// ActiveModels is a set of models(including LoRA adapters) that are currently cached to GPU.
	ActiveModels  map[string]int
	WaitingModels map[string]int
	// MaxActiveModels is the maximum number of models that can be loaded to GPU.
	MaxActiveModels         int
	RunningQueueSize        int
	WaitingQueueSize        int
	KVCacheUsagePercent     float64
	KvCacheMaxTokenCapacity int
	CacheBlockSize          int

	// UpdateTime records the last time when the metrics were updated.
	UpdateTime time.Time
}

// NewMetrics initializes a new empty Metrics object.
func NewMetrics() *Metrics {
	return &Metrics{
		ActiveModels:  make(map[string]int),
		WaitingModels: make(map[string]int),
	}
}

// String returns a string with all Metric information
func (m *Metrics) String() string {
	if m == nil {
		return ""
	}
	return fmt.Sprintf("%+v", *m)
}

// Clone creates a copy of Metrics and returns its pointer.
// Clone returns nil if the object being cloned is nil.
func (m *Metrics) Clone() *Metrics {
	if m == nil {
		return nil
	}
	activeModels := make(map[string]int, len(m.ActiveModels))
	for key, value := range m.ActiveModels {
		activeModels[key] = value
	}
	waitingModels := make(map[string]int, len(m.WaitingModels))
	for key, value := range m.WaitingModels {
		waitingModels[key] = value
	}
	return &Metrics{
		ActiveModels:            activeModels,
		WaitingModels:           waitingModels,
		MaxActiveModels:         m.MaxActiveModels,
		RunningQueueSize:        m.RunningQueueSize,
		WaitingQueueSize:        m.WaitingQueueSize,
		KVCacheUsagePercent:     m.KVCacheUsagePercent,
		KvCacheMaxTokenCapacity: m.KvCacheMaxTokenCapacity,
		CacheBlockSize:          m.CacheBlockSize,
		UpdateTime:              m.UpdateTime,
	}
}
