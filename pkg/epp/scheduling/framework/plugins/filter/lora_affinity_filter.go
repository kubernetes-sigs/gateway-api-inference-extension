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
	"fmt"
	"math/rand"
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/config"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

const (
	LoraAffinityFilterType = "lora-affinity"
)

type loraAffinityFilterParameters struct {
	Threshold float64 `json:"threshold"`
}

// compile-time type validation
var _ framework.Filter = &LoraAffinityFilter{}

// LoraAffinityFilterFactory defines the factory function for LoraAffinityFilter.
func LoraAffinityFilterFactory(name string, rawParameters json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	parameters := loraAffinityFilterParameters{Threshold: config.DefaultLoraAffinityThreshold}
	if err := json.Unmarshal(rawParameters, &parameters); err != nil {
		return nil, fmt.Errorf("failed to parse the parameters of the '%s' filter - %w", LoraAffinityFilterType, err)
	}
	return NewLoraAffinityFilter(parameters.Threshold).WithName(name), nil
}

// NewLoraAffinityFilter initializes a new LoraAffinityFilter and returns its pointer.
func NewLoraAffinityFilter(threshold float64) *LoraAffinityFilter {
	return &LoraAffinityFilter{
		tn:                    plugins.TypedName{Type: LoraAffinityFilterType, Name: LoraAffinityFilterType},
		loraAffinityThreshold: threshold,
	}
}

// LoraAffinityFilter implements a pod selection strategy that prioritizes pods
// with existing LoRA model affinity while allowing for load balancing through randomization.
//
// The function works by:
// 1. Separating pods into two groups: those with target model affinity and those with available capacity
// 2. Using a probability threshold to sometimes select from non-affinity pods to enable load balancing
// 3. Falling back to whatever group has pods if one group is empty
type LoraAffinityFilter struct {
	tn                    plugins.TypedName
	loraAffinityThreshold float64
}

// TypedName returns the type and name tuple of this plugin instance.
func (f *LoraAffinityFilter) TypedName() plugins.TypedName {
	return f.tn
}

// WithName sets the type of the filter.
func (f *LoraAffinityFilter) WithName(name string) *LoraAffinityFilter {
	f.tn.Name = name
	return f
}

// Filter filters out pods that doesn't meet the filter criteria.
func (f *LoraAffinityFilter) Filter(_ context.Context, _ *types.CycleState, request *types.LLMRequest, pods []types.Pod) []types.Pod {
	// Pre-allocate slices with estimated capacity
	filtered_affinity := make([]types.Pod, 0, len(pods))
	filtered_available := make([]types.Pod, 0, len(pods))

	// Categorize pods based on affinity and availability
	for _, pod := range pods {
		_, active := pod.GetMetrics().ActiveModels[request.TargetModel]
		_, waiting := pod.GetMetrics().WaitingModels[request.TargetModel]

		if active || waiting {
			filtered_affinity = append(filtered_affinity, pod)
		} else if len(pod.GetMetrics().ActiveModels)+len(pod.GetMetrics().WaitingModels) < pod.GetMetrics().MaxActiveModels {
			filtered_available = append(filtered_available, pod)
		}
	}

	// Use crypto/rand for better randomization in production environments
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	// If both groups have pods, use probability to select which group to return
	if len(filtered_affinity) > 0 && len(filtered_available) > 0 {
		if randGen.Float64() < f.loraAffinityThreshold {
			return filtered_affinity
		}
		return filtered_available
	}

	// Return whichever group has pods
	if len(filtered_affinity) > 0 {
		return filtered_affinity
	}

	return filtered_available
}
