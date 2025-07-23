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
	"errors"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	DecisionTreeFilterType = "decision-tree-filter"
)

// compile-time type assertion
var _ framework.Filter = &DecisionTreeFilter{}

// DecisionTreeFilter applies current filter, and then recursively applies next filters
// depending success or failure of the current filter.
// It can be used to construct a flow chart algorithm.
// Since a DecisionTreeFilter takes on the type and name of the current filter,
// it is not embedding a fixed plugins.TypeName.
type DecisionTreeFilter struct {
	Current framework.Filter
	// NextOnSuccess filter will be applied after successfully applying the current filter.
	// The filtered results will be passed to the next filter.
	NextOnSuccess framework.Filter
	// NextOnFailure filter will be applied if current filter results in no pods.
	// The original input will be passed to the next filter.
	NextOnFailure framework.Filter
	// NextOnSuccessOrFailure is a convenience field to configure the next filter regardless of the
	// success or failure of the current filter.
	// NOTE: When using NextOnSuccessOrFailure, both nextOnSuccess and nextOnFailure SHOULD be nil.
	// However if that's not the case, nextOnSuccess and nextOnFailure will be used, instead of
	// NextOnSuccessOrFailure, in the success and failure scenarios, respectively.
	NextOnSuccessOrFailure framework.Filter
}

type decisionTreeFilterParameters struct {
	Current                *decisionTreeFilterEntry `json:"current"`
	NextOnSuccess          *decisionTreeFilterEntry `json:"nextOnSuccess"`
	NextOnFailure          *decisionTreeFilterEntry `json:"nextOnFailure"`
	NextOnSuccessOrFailure *decisionTreeFilterEntry `json:"nextOnSuccessOrFailure"`
}

type decisionTreeFilterEntry struct {
	PluginRef    *string                       `json:"pluginRef"`
	DecisionTree *decisionTreeFilterParameters `json:"decisionTree"`
}

func DecisionTreeFilterFactory(name string, rawParameters json.RawMessage, handle plugins.Handle) (plugins.Plugin, error) {
	parameters := decisionTreeFilterParameters{}
	if err := json.Unmarshal(rawParameters, &parameters); err != nil {
		return nil, fmt.Errorf("failed to parse the parameters of the '%s' filter - %w", name, err)
	}
	return loadDecisionTree(&parameters, handle)
}

func loadDecisionTree(parameters *decisionTreeFilterParameters, handle plugins.Handle) (*DecisionTreeFilter, error) {
	result := &DecisionTreeFilter{}
	var err error

	if parameters.Current == nil {
		return nil, errors.New("a current filter must be specified")
	}
	result.Current, err = loadDecisionTreeEntry(parameters.Current, handle)
	if err != nil {
		return nil, err
	}

	if parameters.NextOnSuccess != nil {
		result.NextOnSuccess, err = loadDecisionTreeEntry(parameters.NextOnSuccess, handle)
		if err != nil {
			return nil, err
		}
	}

	if parameters.NextOnFailure != nil {
		result.NextOnFailure, err = loadDecisionTreeEntry(parameters.NextOnFailure, handle)
		if err != nil {
			return nil, err
		}
	}

	if parameters.NextOnSuccessOrFailure != nil {
		result.NextOnSuccessOrFailure, err = loadDecisionTreeEntry(parameters.NextOnSuccessOrFailure, handle)
		if err != nil {
			return nil, err
		}
	}

	return result, nil
}

func loadDecisionTreeEntry(entry *decisionTreeFilterEntry, handle plugins.Handle) (framework.Filter, error) {
	if entry.PluginRef != nil && entry.DecisionTree != nil {
		return nil, errors.New("both pluginRef and decisionTree may not be specified")
	}

	if entry.PluginRef != nil {
		instance := handle.Plugin(*entry.PluginRef)
		if instance == nil {
			return nil, errors.New(*entry.PluginRef + " is a reference to an undefined Plugin")
		}
		if theFilter, ok := instance.(framework.Filter); ok {
			return theFilter, nil
		}
		return nil, errors.New(*entry.PluginRef + " is not a filter")
	} else if entry.DecisionTree != nil {
		return loadDecisionTree(entry.DecisionTree, handle)
	}
	return nil, errors.New("either pluginRef or decisionTree must be specified")
}

func (f *DecisionTreeFilter) TypedName() plugins.TypedName {
	if f == nil {
		// TODO: this keeps the previous behavior ("nil"/"") - not sure
		// why done this way.
		// Change to empty TypedName or some more meaningful values?
		return plugins.TypedName{Type: "nil", Name: ""}
	}
	return f.Current.TypedName()
}

// Filter filters out pods that doesn't meet the filter criteria.
func (f *DecisionTreeFilter) Filter(ctx context.Context, cycleState *types.CycleState, request *types.LLMRequest, pods []types.Pod) []types.Pod {
	loggerTrace := log.FromContext(ctx).V(logutil.TRACE)
	filteredPod := f.Current.Filter(ctx, cycleState, request, pods)

	next := f.NextOnSuccessOrFailure
	if len(filteredPod) > 0 {
		if f.NextOnSuccess == nil && f.NextOnSuccessOrFailure == nil {
			// No succeeding filters to run, return.
			return filteredPod
		}
		if f.NextOnSuccess != nil {
			next = f.NextOnSuccess
		}
		loggerTrace.Info("Filter succeeded", "filter", f.TypedName(), "next", next.TypedName(), "filteredPodCount", len(filteredPod))
		// On success, pass the filtered result to the next filter.
		return next.Filter(ctx, cycleState, request, filteredPod)
	} else {
		if f.NextOnFailure == nil && f.NextOnSuccessOrFailure == nil {
			// No succeeding filters to run, return.
			return filteredPod
		}
		if f.NextOnFailure != nil {
			next = f.NextOnFailure
		}
		loggerTrace.Info("Filter failed", "filter", f.TypedName(), "next", next.TypedName())
		// On failure, pass the initial set of pods to the next filter.
		return next.Filter(ctx, cycleState, request, pods)
	}
}
