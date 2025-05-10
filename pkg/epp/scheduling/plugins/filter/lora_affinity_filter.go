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
	"math/rand"
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/config"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

// compile-time type validation
var _ plugins.Filter = &LoraAffinityFilter{}

// NewLoraAffinityFilter returns a new LoraAffinityFilter.
func NewLoraAffinityFilter() *LoraAffinityFilter {
	return &LoraAffinityFilter{
		loraAffinityThreshold: config.Conf.LoraAffinityThreshold,
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
	loraAffinityThreshold float64
}

// Name returns the name of the filter.
func (f *LoraAffinityFilter) Name() string {
	return "lora-affinity"
}

// Filter filters out pods that doesn't meet the filter criteria.
func (f *LoraAffinityFilter) Filter(ctx *types.SchedulingContext, pods []types.Pod) []types.Pod {
	// Pre-allocate slices with estimated capacity
	filtered_affinity := make([]types.Pod, 0, len(pods))
	filtered_available := make([]types.Pod, 0, len(pods))

	// Categorize pods based on affinity and availability
	for _, pod := range pods {
		_, active := pod.GetMetrics().ActiveModels[ctx.Req.TargetModel]
		_, waiting := pod.GetMetrics().WaitingModels[ctx.Req.TargetModel]

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
