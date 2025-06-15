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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/config"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

const LowQueueFilterName = "low-queue"

type lowQueueFilterParameters struct {
	Threshold int `json:"threshold"`
}

// compile-time type validation
var _ framework.Filter = &LowQueueFilter{}

// LowQueueFilterFactory is the factory function for the LowQueue filter
func LowQueueFilterFactory(rawParameters json.RawMessage) (plugins.Plugin, error) {
	// Use a default logger for plugin creation
	baseLogger := log.Log.WithName("low-queue-filter-factory")

	parameters := lowQueueFilterParameters{Threshold: config.DefaultQueueingThresholdLoRA}
	if err := json.Unmarshal(rawParameters, &parameters); err != nil {
		baseLogger.Error(err,
			"failed to parse the parameters of the "+LowQueueFilterName+" filter")
		return nil, err
	}

	return &LowQueueFilter{queueingThresholdLoRA: parameters.Threshold}, nil
}

// NewLowQueueFilter initializes a new LowQueueFilter and returns its pointer.
func NewLowQueueFilter() *LowQueueFilter {
	return &LowQueueFilter{
		queueingThresholdLoRA: config.Conf.QueueingThresholdLoRA,
	}
}

// LowQueueFilter returns pods that their waiting queue size is less than a configured threshold
type LowQueueFilter struct {
	queueingThresholdLoRA int
}

// Name returns the name of the filter.
func (f *LowQueueFilter) Name() string {
	return LowQueueFilterName
}

// Filter filters out pods that doesn't meet the filter criteria.
func (f *LowQueueFilter) Filter(_ context.Context, _ *types.LLMRequest, _ *types.CycleState, pods []types.Pod) []types.Pod {
	filteredPods := []types.Pod{}

	for _, pod := range pods {
		if pod.GetMetrics().WaitingQueueSize <= f.queueingThresholdLoRA {
			filteredPods = append(filteredPods, pod)
		}
	}

	return filteredPods
}
