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
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
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
	pod := toInternalPod(in)
	pm := &podMetrics{
		pmc:       f.pmc,
		ds:        ds,
		interval:  f.refreshMetricsInterval,
		startOnce: sync.Once{},
		stopOnce:  sync.Once{},
		done:      make(chan struct{}),
		logger:    log.FromContext(parentCtx).WithValues("pod", pod.NamespacedName),
	}
	pm.pod.Store(pod)
	pm.metrics.Store(newMetricsState())

	pm.startRefreshLoop(parentCtx)
	return pm
}

type PodMetrics interface {
	GetPod() *backend.Pod
	GetMetrics() *MetricsState
	UpdatePod(*corev1.Pod)
	StopRefreshLoop()
	String() string
}
