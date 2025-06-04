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

package filter

import (
	"context"
	"encoding/json"

	"sigs.k8s.io/controller-runtime/pkg/log"
	commonconfig "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/common/config"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/config"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

const sheddableCapacityFilterName = "sheddable-capacity"

type sheddableCapacityFilterParameters struct {
	QueueThreshold   int     `json:"queue-threshold"`
	KvCacheThreshold float64 `json:"kvcache-threshold"`
}

func init() {
	framework.Register(sheddableCapacityFilterName, sheddableCapacityFilterFcatory)
}

func sheddableCapacityFilterFcatory(rawParameters json.RawMessage) (framework.Plugin, error) {
	// Use a default logger for plugin creation
	baseLogger := log.Log.WithName("sheddable-capacity-filter-factory")

	parameters := sheddableCapacityFilterParameters{
		QueueThreshold:   commonconfig.DefaultQueueThresholdCritical,
		KvCacheThreshold: commonconfig.DefaultKVCacheThreshold,
	}
	if err := json.Unmarshal(rawParameters, &parameters); err != nil {
		baseLogger.Error(err,
			"failed to parse the parameters of the "+sheddableCapacityFilterName+" filter")
		return nil, err
	}

	return &SheddableCapacityFilter{
		queueThreshold:   parameters.QueueThreshold,
		kvCacheThreshold: parameters.KvCacheThreshold,
	}, nil
}

// compile-time type validation
var _ framework.Filter = &SheddableCapacityFilter{}

// NewSheddableCapacityFilter initializes a new SheddableCapacityFilter and returns its pointer.
func NewSheddableCapacityFilter() *SheddableCapacityFilter {
	return &SheddableCapacityFilter{
		queueThreshold:   config.Conf.QueueThresholdCritical,
		kvCacheThreshold: config.Conf.KVCacheThreshold,
	}
}

// SheddableCapacityFilter filters only pods that has capacity for sheddable requests.
type SheddableCapacityFilter struct {
	queueThreshold   int
	kvCacheThreshold float64
}

// Name returns the name of the filter.
func (f *SheddableCapacityFilter) Name() string {
	return sheddableCapacityFilterName
}

// Filter filters out pods that doesn't meet the filter criteria.
func (f *SheddableCapacityFilter) Filter(_ context.Context, request *types.LLMRequest, _ *types.CycleState, pods []types.Pod) []types.Pod {
	if request.Critical {
		return pods // // Allow all pods to passthrough if the request is critical, even if all pods reach their capacity.
	}

	filteredPods := []types.Pod{}

	for _, pod := range pods {
		if pod.GetMetrics().WaitingQueueSize <= f.queueThreshold && pod.GetMetrics().KVCacheUsagePercent <= f.kvCacheThreshold {
			filteredPods = append(filteredPods, pod)
		}
	}

	return filteredPods
}
