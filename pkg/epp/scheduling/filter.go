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

package scheduling

import (
	"errors"
	"math"
	"math/rand"
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

type Filter interface {
	Name() string
	Filter(ctx *types.Context, pods []*types.PodMetrics) ([]*types.PodMetrics, error)
}

type basicFilter struct {
	name   string
	filter filterFunc
}

func (bf *basicFilter) Name() string {
	if bf == nil {
		return "nil"
	}
	return bf.name
}

func (bf *basicFilter) Filter(ctx *types.Context, pods []*types.PodMetrics) ([]*types.PodMetrics, error) {
	loggerTrace := ctx.Logger.V(logutil.TRACE)
	loggerTrace.Info("Running a filter", "name", bf.Name(), "podCount", len(pods))

	return bf.filter(ctx, pods)
}

// decisionTreeFilter applies current filterFunc, and then recursively applies next filters
// depending success or failure of the current filter.
// It can be used to construct a flow chart algorithm.
type decisionTreeFilter struct {
	current Filter
	// nextOnSuccess filter will be applied after successfully applying the current filter.
	// The filtered results will be passed to the next filter.
	nextOnSuccess Filter
	// nextOnFailure filter will be applied if current filter fails.
	// The original input will be passed to the next filter.
	nextOnFailure Filter
	// nextOnSuccessOrFailure is a convenience field to configure the next filter regardless of the
	// success or failure of the current filter.
	// NOTE: When using nextOnSuccessOrFailure, both nextOnSuccess and nextOnFailure SHOULD be nil.
	// However if that's not the case, nextOnSuccess and nextOnFailure will be used, instead of
	// nextOnSuccessOrFailure,  in the success and failure scenarios, respectively.
	nextOnSuccessOrFailure Filter
}

func (f *decisionTreeFilter) Name() string {
	if f == nil {
		return "nil"
	}
	return f.current.Name()
}

func (f *decisionTreeFilter) Filter(ctx *types.Context, pods []*types.PodMetrics) ([]*types.PodMetrics, error) {
	loggerTrace := ctx.Logger.V(logutil.TRACE)
	filtered, err := f.current.Filter(ctx, pods)

	next := f.nextOnSuccessOrFailure
	if err == nil && len(filtered) > 0 {
		if f.nextOnSuccess == nil && f.nextOnSuccessOrFailure == nil {
			// No succeeding filters to run, return.
			return filtered, err
		}
		if f.nextOnSuccess != nil {
			next = f.nextOnSuccess
		}
		loggerTrace.Info("Filter succeeded", "filter", f.Name(), "next", next.Name(), "filteredPodCount", len(filtered))
		// On success, pass the filtered result to the next filter.
		return next.Filter(ctx, filtered)
	} else {
		if f.nextOnFailure == nil && f.nextOnSuccessOrFailure == nil {
			// No succeeding filters to run, return.
			return filtered, err
		}
		if f.nextOnFailure != nil {
			next = f.nextOnFailure
		}
		loggerTrace.Info("Filter failed", "filter", f.Name(), "next", next.Name())
		// On failure, pass the initial set of pods to the next filter.
		return next.Filter(ctx, pods)
	}
}

// filterFunc filters a set of input pods to a subset.
type filterFunc func(ctx *types.Context, pods []*types.PodMetrics) ([]*types.PodMetrics, error)

// toFilterFunc is a helper function to convert a per pod filter func to the FilterFunc.
func toFilterFunc(pp podPredicate) filterFunc {
	return func(ctx *types.Context, pods []*types.PodMetrics) ([]*types.PodMetrics, error) {
		filtered := []*types.PodMetrics{}
		for _, pod := range pods {
			pass := pp(ctx.Req, pod)
			if pass {
				filtered = append(filtered, pod)
			}
		}
		if len(filtered) == 0 {
			return nil, errors.New("no pods left")
		}
		return filtered, nil
	}
}

var leastQueueFilter = &basicFilter{
	name:   "least queuing",
	filter: leastQueuingFilterFunc,
}

// leastQueuingFilterFunc finds the max and min queue size of all pods, divides the whole range
// (max-min) by the number of pods, and finds the pods that fall into the first range.
// The intuition is that if there are multiple pods that share similar queue size in the low range,
// we should consider them all instead of the absolute minimum one. This worked better than picking
// the least one as it gives more choices for the next filter, which on aggregate gave better
// results.
// TODO: Compare this strategy with other strategies such as top K.
func leastQueuingFilterFunc(ctx *types.Context, pods []*types.PodMetrics) ([]*types.PodMetrics, error) {
	min := math.MaxInt
	max := 0
	filtered := []*types.PodMetrics{}

	for _, pod := range pods {
		if pod.WaitingQueueSize <= min {
			min = pod.WaitingQueueSize
		}
		if pod.WaitingQueueSize >= max {
			max = pod.WaitingQueueSize
		}
	}

	for _, pod := range pods {
		if pod.WaitingQueueSize >= min && pod.WaitingQueueSize <= min+(max-min)/len(pods) {
			filtered = append(filtered, pod)
		}
	}
	return filtered, nil
}

var lowQueueFilter = &basicFilter{
	name:   "low queueing filter",
	filter: toFilterFunc((queueThresholdPredicate(config.QueueingThresholdLoRA))),
}

var leastKVCacheFilter = &basicFilter{
	name:   "least KV cache percent",
	filter: leastKVCacheFilterFunc,
}

// leastKVCacheFilterFunc finds the max and min KV cache of all pods, divides the whole range
// (max-min) by the number of pods, and finds the pods that fall into the first range.
// The intuition is that if there are multiple pods that share similar KV cache in the low range, we
// should consider them all instead of the absolute minimum one. This worked better than picking the
// least one as it gives more choices for the next filter, which on aggregate gave better results.
// TODO: Compare this strategy with other strategies such as top K.
func leastKVCacheFilterFunc(ctx *types.Context, pods []*types.PodMetrics) ([]*types.PodMetrics, error) {
	min := math.MaxFloat64
	var max float64 = 0
	filtered := []*types.PodMetrics{}

	for _, pod := range pods {
		if pod.KVCacheUsagePercent <= min {
			min = pod.KVCacheUsagePercent
		}
		if pod.KVCacheUsagePercent >= max {
			max = pod.KVCacheUsagePercent
		}
	}

	for _, pod := range pods {
		if pod.KVCacheUsagePercent >= min && pod.KVCacheUsagePercent <= min+(max-min)/float64(len(pods)) {
			filtered = append(filtered, pod)
		}
	}
	return filtered, nil
}

var loRAAffinityFilter = &basicFilter{
	name:   "affinity LoRA",
	filter: loRASoftAffinityFilterFunc,
}

// loRASoftAffinityPredicate implements a pod selection strategy that prioritizes pods
// with existing LoRA model affinity while allowing for load balancing through randomization.
//
// The function works by:
// 1. Separating pods into two groups: those with target model affinity and those with available capacity
// 2. Using a probability threshold to sometimes select from non-affinity pods to enable load balancing
// 3. Falling back to whatever group has pods if one group is empty
//
// Parameters:
//   - logger: Logger interface for diagnostic output
//   - req: LLM request containing the resolved target model
//   - pods: Slice of pod metrics to filter
//
// Returns:
//   - Filtered slice of pod metrics based on affinity and availability
//   - Error if any issues occur during filtering
func loRASoftAffinityFilterFunc(ctx *types.Context, pods []*types.PodMetrics) ([]*types.PodMetrics, error) {

	// Pre-allocate slices with estimated capacity
	filtered_affinity := make([]*types.PodMetrics, 0, len(pods))
	filtered_available := make([]*types.PodMetrics, 0, len(pods))

	// Categorize pods based on affinity and availability
	for _, pod := range pods {
		_, active := pod.ActiveModels[ctx.Req.ResolvedTargetModel]
		_, waiting := pod.WaitingModels[ctx.Req.ResolvedTargetModel]

		if active || waiting {
			filtered_affinity = append(filtered_affinity, pod)
		} else if len(pod.ActiveModels)+len(pod.WaitingModels) < pod.MaxActiveModels {
			filtered_available = append(filtered_available, pod)
		}
	}

	// Use crypto/rand for better randomization in production environments
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	// If both groups have pods, use probability to select which group to return
	if len(filtered_affinity) > 0 && len(filtered_available) > 0 {
		if randGen.Float64() < config.LoraAffinityThreshold {
			return filtered_affinity, nil
		}
		return filtered_available, nil
	}

	// Return whichever group has pods
	if len(filtered_affinity) > 0 {
		return filtered_affinity, nil
	}

	return filtered_available, nil
}

// podPredicate is a filter function to check whether a pod is desired.
type podPredicate func(req *types.LLMRequest, pod *types.PodMetrics) bool

func queueThresholdPredicate(queueThreshold int) podPredicate {
	return func(req *types.LLMRequest, pod *types.PodMetrics) bool {
		return pod.WaitingQueueSize <= queueThreshold
	}
}

func kvCacheThresholdPredicate(kvCacheThreshold float64) podPredicate {
	return func(req *types.LLMRequest, pod *types.PodMetrics) bool {
		return pod.KVCacheUsagePercent <= kvCacheThreshold
	}
}

func (pp podPredicate) and(another podPredicate) podPredicate {
	return func(req *types.LLMRequest, pod *types.PodMetrics) bool {
		return pp(req, pod) && another(req, pod)
	}
}
