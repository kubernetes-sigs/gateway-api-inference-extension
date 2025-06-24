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
	"math"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

const (
	LeastKVCacheFilterType = "least-KV-cache"
)

// compile-time type validation
var _ framework.Filter = &LeastKVCacheFilter{}

// LeastKVCacheFilterFactory defines the factory function for LeastKVCacheFilter.
func LeastKVCacheFilterFactory(name string, _ json.RawMessage, _ plugins.Handle) (plugins.Plugin, error) {
	return NewLeastKVCacheFilter().WithName(name), nil
}

// NewLeastKVCacheFilter initializes a new LeastKVCacheFilter and returns its pointer.
func NewLeastKVCacheFilter() *LeastKVCacheFilter {
	f := &LeastKVCacheFilter{}
	return f.WithName("")
}

// LeastKVCacheFilter finds the max and min KV cache of all pods, divides the whole range
// (max-min) by the number of pods, and finds the pods that fall into the first range.
// The intuition is that if there are multiple pods that share similar KV cache in the low range, we
// should consider them all instead of the absolute minimum one. This worked better than picking the
// least one as it gives more choices for the next filter, which on aggregate gave better results.
type LeastKVCacheFilter struct {
	name string
}

// Type returns the type of the filter.
func (f *LeastKVCacheFilter) Type() string {
	return LeastKVCacheFilterType
}

// Name returns the name of the filter.
func (f *LeastKVCacheFilter) Name() string {
	return f.name
}

// WithName sets the name of the filter.
func (f *LeastKVCacheFilter) WithName(name string) *LeastKVCacheFilter {
	if name == "" {
		name = f.Type()
	}
	f.name = name
	return f
}

// Filter filters out pods that doesn't meet the filter criteria.
func (f *LeastKVCacheFilter) Filter(_ context.Context, _ *types.CycleState, _ *types.LLMRequest, pods []types.Pod) []types.Pod {
	filteredPods := []types.Pod{}

	min := math.MaxFloat64
	var max float64 = 0

	for _, pod := range pods {
		if pod.GetMetrics().KVCacheUsagePercent <= min {
			min = pod.GetMetrics().KVCacheUsagePercent
		}
		if pod.GetMetrics().KVCacheUsagePercent >= max {
			max = pod.GetMetrics().KVCacheUsagePercent
		}
	}

	for _, pod := range pods {
		if pod.GetMetrics().KVCacheUsagePercent >= min && pod.GetMetrics().KVCacheUsagePercent <= min+(max-min)/float64(len(pods)) {
			filteredPods = append(filteredPods, pod)
		}
	}
	return filteredPods
}
