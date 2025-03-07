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

// Package datastore is a library to interact with backend model servers such as probing metrics.
package datastore

import (
	"fmt"

	"k8s.io/apimachinery/pkg/types"
)

type Pod struct {
	NamespacedName types.NamespacedName
	Address        string
}

type Metrics struct {
	// ActiveModels is a set of models(including LoRA adapters) that are currently cached to GPU.
	ActiveModels map[string]int
	// MaxActiveModels is the maximum number of models that can be loaded to GPU.
	MaxActiveModels     int
	WaitingQueueSize    int
	KVCacheUsagePercent float64
}

type PodMetrics struct {
	Pod
	Metrics
}

func (pm *PodMetrics) String() string {
	return fmt.Sprintf("Pod: %+v; Address: %+v; Metrics: %+v", pm.NamespacedName, pm.Address, pm.Metrics)
}

func (pm *PodMetrics) Clone() *PodMetrics {
	cm := make(map[string]int, len(pm.ActiveModels))
	for k, v := range pm.ActiveModels {
		cm[k] = v
	}
	clone := &PodMetrics{
		Pod: Pod{
			NamespacedName: pm.NamespacedName,
			Address:        pm.Address,
		},
		Metrics: Metrics{
			ActiveModels:        cm,
			MaxActiveModels:     pm.MaxActiveModels,
			WaitingQueueSize:    pm.WaitingQueueSize,
			KVCacheUsagePercent: pm.KVCacheUsagePercent,
		},
	}
	return clone
}
