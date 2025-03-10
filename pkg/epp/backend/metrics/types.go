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
	"context"
	"fmt"
	"sync"
	"time"
	"unsafe"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

func NewPodMetricsFactory(pmc PodMetricsClient, refreshMetricsInterval time.Duration) *PodMetricsFactory {
	return &PodMetricsFactory{
		pmc:                    pmc,
		refreshMetricsInterval: refreshMetricsInterval,
	}
}

type PodMetricsFactory struct {
	pmc                    PodMetricsClient
	refreshMetricsInterval time.Duration
}

func (f *PodMetricsFactory) NewPodMetrics(parentCtx context.Context, in *corev1.Pod, ds Datastore) PodMetrics {
	pm := &podMetrics{
		pod:       unsafe.Pointer(toInternalPod(in)),
		metrics:   unsafe.Pointer(newMetrics()),
		pmc:       f.pmc,
		ds:        ds,
		interval:  f.refreshMetricsInterval,
		parentCtx: parentCtx,
		once:      sync.Once{},
		done:      make(chan struct{}),
		logger:    log.FromContext(parentCtx),
	}
	pm.startRefreshLoop()
	return pm
}

type PodMetrics interface {
	GetPod() *Pod
	GetMetrics() *Metrics
	UpdatePod(*corev1.Pod)
	StopRefreshLoop()
}

type Pod struct {
	NamespacedName types.NamespacedName
	Address        string
}

type Metrics struct {
	// ActiveModels is a set of models(including LoRA adapters) that are currently cached to GPU.
	ActiveModels map[string]int
	// MaxActiveModels is the maximum number of models that can be loaded to GPU.
	MaxActiveModels         int
	RunningQueueSize        int
	WaitingQueueSize        int
	KVCacheUsagePercent     float64
	KvCacheMaxTokenCapacity int

	// UpdateTime record the last time when the metrics were updated.
	UpdateTime time.Time
}

func newMetrics() *Metrics {
	return &Metrics{
		ActiveModels: make(map[string]int),
	}
}

func (m *Metrics) String() string {
	if m == nil {
		return ""
	}
	return fmt.Sprintf("%+v", *m)
}

func (m *Metrics) Clone() *Metrics {
	cm := make(map[string]int, len(m.ActiveModels))
	for k, v := range m.ActiveModels {
		cm[k] = v
	}
	clone := &Metrics{
		ActiveModels:            cm,
		MaxActiveModels:         m.MaxActiveModels,
		RunningQueueSize:        m.RunningQueueSize,
		WaitingQueueSize:        m.WaitingQueueSize,
		KVCacheUsagePercent:     m.KVCacheUsagePercent,
		KvCacheMaxTokenCapacity: m.KvCacheMaxTokenCapacity,
		UpdateTime:              m.UpdateTime,
	}
	return clone
}
