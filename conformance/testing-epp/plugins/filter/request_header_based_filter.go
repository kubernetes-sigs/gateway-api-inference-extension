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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

const (
	headerTestEppEndPointSelectionKey = "test-epp-endpoint-selection"
)

// compile-time type assertion
var _ framework.Filter = &HeaderBasedTestingFilter{}

// NewHeaderBasedTestingFilter initializes a new HeaderBasedTestingFilter and returns its pointer.
// This should be only used in testing purpose.
func NewHeaderBasedTestingFilter() *HeaderBasedTestingFilter {
	return &HeaderBasedTestingFilter{}
}

// HeaderBasedTestingFilter filters Pods based on an address specified in the "test-epp-endpoint-selection" request header.
type HeaderBasedTestingFilter struct{}

// Name returns the name of the filter.
func (f *HeaderBasedTestingFilter) Name() string {
	return "test-header-based"
}

// Filter filters out pods that doesn't meet the filter criteria.
func (f *HeaderBasedTestingFilter) Filter(_ context.Context, request *types.LLMRequest, _ *types.CycleState, pods []types.Pod) []types.Pod {
	filteredPods := []types.Pod{}

	endPointInReqeust, found := request.Headers[headerTestEppEndPointSelectionKey]
	if !found {
		return filteredPods
	}

	for _, pod := range pods {
		if pod.GetPod().Address == endPointInReqeust {
			filteredPods = append(filteredPods, pod)
		}
	}
	return filteredPods
}
