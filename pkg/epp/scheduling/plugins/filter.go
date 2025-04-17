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

package plugins

import (
	"errors"
	"math"
	"math/rand"
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/config"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

type Filter struct {
	name   string
	filter filterFunc
}

func (bf *Filter) Name() string {
	if bf == nil {
		return "nil"
	}
	return bf.name
}

func (bf *Filter) Filter(ctx *types.Context, pods []types.Pod) ([]types.Pod, error) {
	loggerTrace := ctx.Logger.V(logutil.TRACE)
	loggerTrace.Info("Running a filter", "name", bf.Name(), "podCount", len(pods))

	return bf.filter(ctx, pods)
}

// DecisionTreeFilter applies current filterFunc, and then recursively applies next filters
// depending success or failure of the current filter.
// It can be used to construct a flow chart algorithm.
type DecisionTreeFilter struct {
	Current types.Filter
	// NextOnSuccess filter will be applied after successfully applying the current filter.
	// The filtered results will be passed to the next filter.
	NextOnSuccess types.Filter
	// NextOnFailure filter will be applied if current filter fails.
	// The original input will be passed to the next filter.
	NextOnFailure types.Filter
	// NextOnSuccessOrFailure is a convenience field to configure the next filter regardless of the
	// success or failure of the current filter.
	// NOTE: When using NextOnSuccessOrFailure, both nextOnSuccess and nextOnFailure SHOULD be nil.
	// However if that's not the case, nextOnSuccess and nextOnFailure will be used, instead of
	// NextOnSuccessOrFailure,  in the success and failure scenarios, respectively.
	NextOnSuccessOrFailure types.Filter
}

func (f *DecisionTreeFilter) Name() string {
	if f == nil {
		return "nil"
	}
	return f.Current.Name()
}

func (f *DecisionTreeFilter) Filter(ctx *types.Context, pods []types.Pod) ([]types.Pod, error) {
	loggerTrace := ctx.Logger.V(logutil.TRACE)
	filtered, err := f.Current.Filter(ctx, pods)

	next := f.NextOnSuccessOrFailure
	if err == nil && len(filtered) > 0 {
		if f.NextOnSuccess == nil && f.NextOnSuccessOrFailure == nil {
			// No succeeding filters to run, return.
			return filtered, err
		}
		if f.NextOnSuccess != nil {
			next = f.NextOnSuccess
		}
		loggerTrace.Info("Filter succeeded", "filter", f.Name(), "next", next.Name(), "filteredPodCount", len(filtered))
		// On success, pass the filtered result to the next filter.
		return next.Filter(ctx, filtered)
	} else {
		if f.NextOnFailure == nil && f.NextOnSuccessOrFailure == nil {
			// No succeeding filters to run, return.
			return filtered, err
		}
		if f.NextOnFailure != nil {
			next = f.NextOnFailure
		}
		loggerTrace.Info("Filter failed", "filter", f.Name(), "next", next.Name())
		// On failure, pass the initial set of pods to the next filter.
		return next.Filter(ctx, pods)
	}
}

// filterFunc filters a set of input pods to a subset.
type filterFunc func(ctx *types.Context, pods []types.Pod) ([]types.Pod, error)

// toFilterFunc is a helper function to convert a per pod filter func to the FilterFunc.
func toFilterFunc(pp podPredicate) filterFunc {
	return func(ctx *types.Context, pods []types.Pod) ([]types.Pod, error) {
		filtered := []types.Pod{}
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

var LeastQueueFilter = &Filter{
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
func leastQueuingFilterFunc(ctx *types.Context, pods []types.Pod) ([]types.Pod, error) {
	min := math.MaxInt
	max := 0
	filtered := []types.Pod{}

	for _, pod := range pods {
		if pod.GetMetrics().WaitingQueueSize <= min {
			min = pod.GetMetrics().WaitingQueueSize
		}
		if pod.GetMetrics().WaitingQueueSize >= max {
			max = pod.GetMetrics().WaitingQueueSize
		}
	}

	for _, pod := range pods {
		if pod.GetMetrics().WaitingQueueSize >= min && pod.GetMetrics().WaitingQueueSize <= min+(max-min)/len(pods) {
			filtered = append(filtered, pod)
		}
	}
	return filtered, nil
}

var LowQueueFilter = &Filter{
	name:   "low queueing filter",
	filter: toFilterFunc((queueThresholdPredicate(config.Conf.QueueingThresholdLoRA))),
}

var LeastKVCacheFilter = &Filter{
	name:   "least KV cache percent",
	filter: leastKVCacheFilterFunc,
}

// leastKVCacheFilterFunc finds the max and min KV cache of all pods, divides the whole range
// (max-min) by the number of pods, and finds the pods that fall into the first range.
// The intuition is that if there are multiple pods that share similar KV cache in the low range, we
// should consider them all instead of the absolute minimum one. This worked better than picking the
// least one as it gives more choices for the next filter, which on aggregate gave better results.
// TODO: Compare this strategy with other strategies such as top K.
func leastKVCacheFilterFunc(ctx *types.Context, pods []types.Pod) ([]types.Pod, error) {
	min := math.MaxFloat64
	var max float64 = 0
	filtered := []types.Pod{}

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
			filtered = append(filtered, pod)
		}
	}
	return filtered, nil
}

var LoRAAffinityFilter = &Filter{
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
func loRASoftAffinityFilterFunc(ctx *types.Context, pods []types.Pod) ([]types.Pod, error) {

	// Pre-allocate slices with estimated capacity
	filtered_affinity := make([]types.Pod, 0, len(pods))
	filtered_available := make([]types.Pod, 0, len(pods))

	// Categorize pods based on affinity and availability
	for _, pod := range pods {
		_, active := pod.GetMetrics().ActiveModels[ctx.Req.ResolvedTargetModel]
		_, waiting := pod.GetMetrics().WaitingModels[ctx.Req.ResolvedTargetModel]

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
		if randGen.Float64() < config.Conf.LoraAffinityThreshold {
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

var HasCapacityFilter = &Filter{
	name:   "has capacity for sheddable requests",
	filter: toFilterFunc(queueThresholdPredicate(config.Conf.QueueThresholdCritical).and(kvCacheThresholdPredicate(config.Conf.KVCacheThreshold))),
}

var DropRequestFilter = &Filter{
	name: "drop request",
	filter: func(ctx *types.Context, pods []types.Pod) ([]types.Pod, error) {
		ctx.Logger.V(logutil.DEFAULT).Info("Request dropped", "request", ctx.Req)
		return []types.Pod{}, errutil.Error{
			Code: errutil.InferencePoolResourceExhausted, Msg: "dropping request due to limited backend resources",
		}
	},
}

// podPredicate is a filter function to check whether a pod is desired.
type podPredicate func(req *types.LLMRequest, pod types.Pod) bool

func queueThresholdPredicate(queueThreshold int) podPredicate {
	return func(req *types.LLMRequest, pod types.Pod) bool {
		return pod.GetMetrics().WaitingQueueSize <= queueThreshold
	}
}

func kvCacheThresholdPredicate(kvCacheThreshold float64) podPredicate {
	return func(req *types.LLMRequest, pod types.Pod) bool {
		return pod.GetMetrics().KVCacheUsagePercent <= kvCacheThreshold
	}
}

func (pp podPredicate) and(another podPredicate) podPredicate {
	return func(req *types.LLMRequest, pod types.Pod) bool {
		return pp(req, pod) && another(req, pod)
	}
}
