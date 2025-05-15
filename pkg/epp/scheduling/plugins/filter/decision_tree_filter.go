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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// DecisionTreeFilter applies current fitler, and then recursively applies next filters
// depending success or failure of the current filter.
// It can be used to construct a flow chart algorithm.
type DecisionTreeFilter struct {
	Current plugins.Filter
	// NextOnSuccess filter will be applied after successfully applying the current filter.
	// The filtered results will be passed to the next filter.
	NextOnSuccess plugins.Filter
	// NextOnFailure filter will be applied if current filter results in no pods.
	// The original input will be passed to the next filter.
	NextOnFailure plugins.Filter
	// NextOnSuccessOrFailure is a convenience field to configure the next filter regardless of the
	// success or failure of the current filter.
	// NOTE: When using NextOnSuccessOrFailure, both nextOnSuccess and nextOnFailure SHOULD be nil.
	// However if that's not the case, nextOnSuccess and nextOnFailure will be used, instead of
	// NextOnSuccessOrFailure, in the success and failure scenarios, respectively.
	NextOnSuccessOrFailure plugins.Filter
}

// Name returns the name of the filter.
func (f *DecisionTreeFilter) Name() string {
	if f == nil {
		return "nil"
	}
	return f.Current.Name()
}

// Filter filters out pods that doesn't meet the filter criteria.
func (f *DecisionTreeFilter) Filter(ctx *types.SchedulingContext, pods []types.Pod) []types.Pod {
	loggerTrace := ctx.Logger.V(logutil.TRACE)
	filteredPod := f.Current.Filter(ctx, pods)

	next := f.NextOnSuccessOrFailure
	if len(filteredPod) > 0 {
		if f.NextOnSuccess == nil && f.NextOnSuccessOrFailure == nil {
			// No succeeding filters to run, return.
			return filteredPod
		}
		if f.NextOnSuccess != nil {
			next = f.NextOnSuccess
		}
		loggerTrace.Info("Filter succeeded", "filter", f.Name(), "next", next.Name(), "filteredPodCount", len(filteredPod))
		// On success, pass the filtered result to the next filter.
		return next.Filter(ctx, filteredPod)
	} else {
		if f.NextOnFailure == nil && f.NextOnSuccessOrFailure == nil {
			// No succeeding filters to run, return.
			return filteredPod
		}
		if f.NextOnFailure != nil {
			next = f.NextOnFailure
		}
		loggerTrace.Info("Filter failed", "filter", f.Name(), "next", next.Name())
		// On failure, pass the initial set of pods to the next filter.
		return next.Filter(ctx, pods)
	}
}
