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

	"github.com/go-logr/logr"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

type Filter interface {
	Name() string
	Filter(logger logr.Logger, req *LLMRequest, pods []backendmetrics.PodMetrics) ([]backendmetrics.PodMetrics, error)
}

// filter applies current filterFunc, and then recursively applies next filters depending success or
// failure of the current filterFunc.
// It can be used to construct a flow chart algorithm.
type filter struct {
	name   string
	filter filterFunc
	// nextOnSuccess filter will be applied after successfully applying the current filter.
	// The filtered results will be passed to the next filter.
	nextOnSuccess *filter
	// nextOnFailure filter will be applied if current filter fails.
	// The original input will be passed to the next filter.
	nextOnFailure *filter
	// nextOnSuccessOrFailure is a convenience field to configure the next filter regardless of the
	// success or failure of the current filter.
	// NOTE: When using nextOnSuccessOrFailure, both nextOnSuccess and nextOnFailure SHOULD be nil.
	// However if that's not the case, nextOnSuccess and nextOnFailure will be used, instead of
	// nextOnSuccessOrFailure,  in the success and failure scenarios, respectively.
	nextOnSuccessOrFailure *filter
}

func (f *filter) Name() string {
	if f == nil {
		return "nil"
	}
	return f.name
}

func (f *filter) Filter(logger logr.Logger, req *LLMRequest, pods []backendmetrics.PodMetrics) ([]backendmetrics.PodMetrics, error) {
	loggerTrace := logger.V(logutil.TRACE)
	loggerTrace.Info("Running a filter", "name", f.Name(), "podCount", len(pods))

	filtered, err := f.filter(logger, req, pods)

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
		return next.Filter(logger, req, filtered)
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
		return next.Filter(logger, req, pods)
	}
}

// filterFunc filters a set of input pods to a subset.
type filterFunc func(logger logr.Logger, req *LLMRequest, pods []backendmetrics.PodMetrics) ([]backendmetrics.PodMetrics, error)

// toFilterFunc is a helper function to convert a per pod filter func to the FilterFunc.
func toFilterFunc(pp podPredicate) filterFunc {
	return func(logger logr.Logger, req *LLMRequest, pods []backendmetrics.PodMetrics) ([]backendmetrics.PodMetrics, error) {
		filtered := []backendmetrics.PodMetrics{}
		for _, pod := range pods {
			pass := pp(req, pod)
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

// leastQueuingFilterFunc finds the max and min queue size of all pods, divides the whole range
// (max-min) by the number of pods, and finds the pods that fall into the first range.
// The intuition is that if there are multiple pods that share similar queue size in the low range,
// we should consider them all instead of the absolute minimum one. This worked better than picking
// the least one as it gives more choices for the next filter, which on aggregate gave better
// results.
// TODO: Compare this strategy with other strategies such as top K.
func leastQueuingFilterFunc(logger logr.Logger, req *LLMRequest, pods []backendmetrics.PodMetrics) ([]backendmetrics.PodMetrics, error) {
	min := math.MaxInt
	max := 0
	filtered := []backendmetrics.PodMetrics{}

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

func lowQueueingPodPredicate(_ *LLMRequest, pod backendmetrics.PodMetrics) bool {
	return pod.GetMetrics().WaitingQueueSize < queueingThresholdLoRA
}

// leastKVCacheFilterFunc finds the max and min KV cache of all pods, divides the whole range
// (max-min) by the number of pods, and finds the pods that fall into the first range.
// The intuition is that if there are multiple pods that share similar KV cache in the low range, we
// should consider them all instead of the absolute minimum one. This worked better than picking the
// least one as it gives more choices for the next filter, which on aggregate gave better results.
// TODO: Compare this strategy with other strategies such as top K.
func leastKVCacheFilterFunc(logger logr.Logger, req *LLMRequest, pods []backendmetrics.PodMetrics) ([]backendmetrics.PodMetrics, error) {
	min := math.MaxFloat64
	var max float64 = 0
	filtered := []backendmetrics.PodMetrics{}

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

// podPredicate is a filter function to check whether a pod is desired.
type podPredicate func(req *LLMRequest, pod backendmetrics.PodMetrics) bool

// We consider serving an adapter low cost it the adapter is active in the model server, or the
// model server has room to load the adapter. The lowLoRACostPredicate ensures weak affinity by
// spreading the load of a LoRA adapter across multiple pods, avoiding "pinning" all requests to
// a single pod. This gave good performance in our initial benchmarking results in the scenario
// where # of lora slots > # of lora adapters.
func lowLoRACostPredicate(req *LLMRequest, pod backendmetrics.PodMetrics) bool {
	_, ok := pod.GetMetrics().ActiveModels[req.ResolvedTargetModel]
	return ok || len(pod.GetMetrics().ActiveModels) < pod.GetMetrics().MaxActiveModels
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
func loRASoftAffinityFilter(logger logr.Logger, req *LLMRequest, pods []backendmetrics.PodMetrics) ([]backendmetrics.PodMetrics, error) {

	// Pre-allocate slices with estimated capacity
	filtered_affinity := make([]backendmetrics.PodMetrics, 0, len(pods))
	filtered_available := make([]backendmetrics.PodMetrics, 0, len(pods))

	// Categorize pods based on affinity and availability
	for _, pod := range pods {

		if _, exists := pod.GetMetrics().ActiveModels[req.ResolvedTargetModel]; exists {
			filtered_affinity = append(filtered_affinity, pod)
		} else if len(pod.GetMetrics().ActiveModels) < pod.GetMetrics().MaxActiveModels {
			filtered_available = append(filtered_available, pod)
		}
	}

	// Use crypto/rand for better randomization in production environments
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	// If both groups have pods, use probability to select which group to return
	if len(filtered_affinity) > 0 && len(filtered_available) > 0 {
		if randGen.Float64() < loraAffinityThreshold {
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

func criticalRequestPredicate(req *LLMRequest, _ backendmetrics.PodMetrics) bool {
	return req.Critical
}

func noQueueAndLessThanKVCacheThresholdPredicate(queueThreshold int, kvCacheThreshold float64) podPredicate {
	return func(req *LLMRequest, pod backendmetrics.PodMetrics) bool {
		return pod.GetMetrics().WaitingQueueSize <= queueThreshold && pod.GetMetrics().KVCacheUsagePercent <= kvCacheThreshold
	}
}
