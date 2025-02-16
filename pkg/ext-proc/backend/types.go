// Package backend is a library to interact with backend model servers such as probing metrics.
package backend

import (
	"fmt"

	"k8s.io/apimachinery/pkg/types"
)

type Metrics struct {
	// ActiveModels is a set of models(including LoRA adapters) that are currently cached to GPU.
	ActiveModels map[string]int
	// MaxActiveModels is the maximum number of models that can be loaded to GPU.
	MaxActiveModels         int
	RunningQueueSize        int
	WaitingQueueSize        int
	KVCacheUsagePercent     float64
	KvCacheMaxTokenCapacity int
}

type PodMetrics struct {
	NamespacedName types.NamespacedName
	Address        string
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
		NamespacedName: pm.NamespacedName,
		Address:        pm.Address,
		Metrics: Metrics{
			ActiveModels:            cm,
			RunningQueueSize:        pm.RunningQueueSize,
			WaitingQueueSize:        pm.WaitingQueueSize,
			KVCacheUsagePercent:     pm.KVCacheUsagePercent,
			KvCacheMaxTokenCapacity: pm.KvCacheMaxTokenCapacity,
		},
	}
	return clone
}
