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
	"fmt"

	"encoding/json"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/config"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

const (
	LowQueueFilterType = "low-queue"
)

type lowQueueFilterParameters struct {
	Threshold int `json:"threshold"`
}

// compile-time type validation
var _ framework.Filter = &LowQueueFilter{}

// LowQueueFilterFactory defines the factory function for LowQueueFilter.
func LowQueueFilterFactory(name string, rawParameters json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	parameters := lowQueueFilterParameters{Threshold: config.DefaultQueueingThresholdLoRA}
	if err := json.Unmarshal(rawParameters, &parameters); err != nil {
		return nil, fmt.Errorf("failed to parse the parameters of the '%s' filter - %w", LowQueueFilterType, err)
	}

	return &LowQueueFilter{
		name:                  name,
		queueingThresholdLoRA: parameters.Threshold}, nil
}

// NewLowQueueFilter initializes a new LowQueueFilter and returns its pointer.
func NewLowQueueFilter(threshold int) *LowQueueFilter {
	return &LowQueueFilter{
		queueingThresholdLoRA: threshold,
	}
}

// LowQueueFilter returns pods that their waiting queue size is less than a configured threshold
type LowQueueFilter struct {
	name                  string
	queueingThresholdLoRA int
}

// Type returns the type of the filter.
func (f *LowQueueFilter) Type() string {
	return LowQueueFilterType
}

// Name returns the name of the filter.
func (f *LowQueueFilter) Name() string {
	return f.name
}

// Filter filters out pods that doesn't meet the filter criteria.
func (f *LowQueueFilter) Filter(_ context.Context, _ *types.CycleState, _ *types.LLMRequest, pods []types.Pod) []types.Pod {
	filteredPods := []types.Pod{}

	for _, pod := range pods {
		if pod.GetMetrics().WaitingQueueSize <= f.queueingThresholdLoRA {
			filteredPods = append(filteredPods, pod)
		}
	}

	return filteredPods
}
