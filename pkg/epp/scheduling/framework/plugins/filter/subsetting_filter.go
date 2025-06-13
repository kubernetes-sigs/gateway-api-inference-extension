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
	"strings"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

const (
	subsetHintNamespace = "envoy.lb.subset_hint"
	subsetHintKey       = "x-gateway-destination-endpoint-subset"
)

// compile-time type assertion
var _ framework.Filter = &SubsetFilter{}

// NewSubsetFilter initializes a new SubsetFilter.
func NewSubsetFilter() *SubsetFilter {
	return &SubsetFilter{}
}

// SubsetFilter filters Pods based on the subset hint provided by the proxy via filterMetadata.
type SubsetFilter struct{}

// Name returns the name of the filter.
func (f *SubsetFilter) Name() string {
	return "subset-hint"
}

// Filter filters out pods that are not in the subset provided in filterMetadata.
func (f *SubsetFilter) Filter(_ context.Context, request *types.LLMRequest, _ *types.CycleState, pods []types.Pod) []types.Pod {
	// Check if envoy.lb.subset_hint is present in the metadata map
	subsetMap, found := request.GetMetadata()[subsetHintNamespace].(map[string]any)
	if !found {
		return pods
	}

	// Check if endpoint key is present in the subset map and ensure there is at least one value
	endpointSubsetList, found := subsetMap[subsetHintKey].([]interface{})
	if !found || len(endpointSubsetList) == 0 {
		return pods
	}

	// Create a map of endpoint addys for easy lookup
	endpoints := make(map[string]bool)
	for _, endpoint := range endpointSubsetList {
		epStr := strings.Split(endpoint.(string), ":")[0]
		endpoints[epStr] = true
	}

	// Filter based on address
	filteredPods := []types.Pod{}
	for _, pod := range pods {
		if _, found := endpoints[pod.GetPod().Address]; found {
			filteredPods = append(filteredPods, pod)
		}
	}

	return filteredPods
}
