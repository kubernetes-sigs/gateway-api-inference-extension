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

// Package metrics is a library to interact with backend metrics.
package metrics

import (
	"fmt"
	"time"

	podinfo "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/pod-info"
)

const (
	MetricsDataKey = "metrics"
)

var _ podinfo.ScrapedData = &MetricsData{}

func newMetrics() *MetricsData {
	return &MetricsData{
		ActiveModels:  make(map[string]int),
		WaitingModels: make(map[string]int),
	}
}

type MetricsData struct {
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

func (m *MetricsData) String() string {
	if m == nil {
		return ""
	}
	return fmt.Sprintf("%+v", *m)
}

func (m *MetricsData) Clone() podinfo.ScrapedData {
	if m == nil {
		return nil
	}
	cm := make(map[string]int, len(m.ActiveModels))
	for k, v := range m.ActiveModels {
		cm[k] = v
	}
	wm := make(map[string]int, len(m.WaitingModels))
	for k, v := range m.WaitingModels {
		wm[k] = v
	}
	return &MetricsData{
		ActiveModels:            cm,
		WaitingModels:           wm,
		MaxActiveModels:         m.MaxActiveModels,
		RunningQueueSize:        m.RunningQueueSize,
		WaitingQueueSize:        m.WaitingQueueSize,
		KVCacheUsagePercent:     m.KVCacheUsagePercent,
		KvCacheMaxTokenCapacity: m.KvCacheMaxTokenCapacity,
		UpdateTime:              m.UpdateTime,
	}
}
