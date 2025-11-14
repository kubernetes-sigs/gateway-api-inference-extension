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

package framework

import (
	"context"
	"fmt"
	"strings"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// NewSchedulerProfile creates a new SchedulerProfile object and returns its pointer.
func NewSchedulerProfile() *SchedulerProfile {
	return &SchedulerProfile{
		filters: []Filter{},
		scorers: []*WeightedScorer{},
		// picker remains nil since profile doesn't support multiple pickers
	}
}

// SchedulerProfile provides a profile configuration for the scheduler which influence routing decisions.
type SchedulerProfile struct {
	filters []Filter
	scorers []*WeightedScorer
	picker  Picker
}

// WithFilters sets the given filter plugins as the Filter plugins.
// if the SchedulerProfile has Filter plugins, this call replaces the existing plugins with the given ones.
func (p *SchedulerProfile) WithFilters(filters ...Filter) *SchedulerProfile {
	p.filters = filters
	return p
}

// WithScorers sets the given scorer plugins as the Scorer plugins.
// if the SchedulerProfile has Scorer plugins, this call replaces the existing plugins with the given ones.
func (p *SchedulerProfile) WithScorers(scorers ...*WeightedScorer) *SchedulerProfile {
	p.scorers = scorers
	return p
}

// WithPicker sets the given picker plugins as the Picker plugin.
// if the SchedulerProfile has Picker plugin, this call replaces the existing plugin with the given one.
func (p *SchedulerProfile) WithPicker(picker Picker) *SchedulerProfile {
	p.picker = picker
	return p
}

// AddPlugins adds the given plugins to all scheduler plugins according to the interfaces each plugin implements.
// A plugin may implement more than one scheduler plugin interface.
// Special Case: In order to add a scorer, one must use the scorer.NewWeightedScorer function in order to provide a weight.
// if a scorer implements more than one interface, supplying a WeightedScorer is sufficient. The function will take the internal
// scorer object and register it to all interfaces it implements.
func (p *SchedulerProfile) AddPlugins(pluginObjects ...plugins.Plugin) error {
	for _, plugin := range pluginObjects {
		if weightedScorer, ok := plugin.(*WeightedScorer); ok {
			p.scorers = append(p.scorers, weightedScorer)
			plugin = weightedScorer.Scorer // if we got WeightedScorer, unwrap the plugin
		} else if scorer, ok := plugin.(Scorer); ok { // if we got a Scorer instead of WeightedScorer that's an error.
			return fmt.Errorf("failed to register scorer '%s' without a weight. follow function documentation to register a scorer", scorer.TypedName())
		}
		if filter, ok := plugin.(Filter); ok {
			p.filters = append(p.filters, filter)
		}
		if picker, ok := plugin.(Picker); ok {
			if p.picker != nil {
				return fmt.Errorf("failed to set '%s' as picker, already have a registered picker plugin '%s'", picker.TypedName(), p.picker.TypedName())
			}
			p.picker = picker
		}
	}
	return nil
}

func (p *SchedulerProfile) String() string {
	filterNames := make([]string, len(p.filters))
	for i, filter := range p.filters {
		filterNames[i] = filter.TypedName().String()
	}
	scorerNames := make([]string, len(p.scorers))
	for i, scorer := range p.scorers {
		scorerNames[i] = fmt.Sprintf("%s: %d", scorer.TypedName(), scorer.Weight())
	}

	return fmt.Sprintf(
		"{Filters: [%s], Scorers: [%s], Picker: %s}",
		strings.Join(filterNames, ", "),
		strings.Join(scorerNames, ", "),
		p.picker.TypedName(),
	)
}

// Run runs a SchedulerProfile. It invokes all the SchedulerProfile plugins for the given request in this
// order - Filters, Scorers, Picker. After completing all, it returns the result.
func (p *SchedulerProfile) Run(ctx context.Context, request *types.LLMRequest, cycleState *types.CycleState, candidatePods []types.Pod) (*types.ProfileRunResult, error) {
	pods := p.runFilterPlugins(ctx, request, cycleState, candidatePods)
	if len(pods) == 0 {
		return nil, errutil.Error{Code: errutil.Internal, Msg: "no pods available for the given request"}
	}
	// if we got here, there is at least one pod to score
	weightedScorePerPod := p.runScorerPlugins(ctx, request, cycleState, pods)

	result := p.runPickerPlugin(ctx, cycleState, weightedScorePerPod)

	return result, nil
}

func (p *SchedulerProfile) runFilterPlugins(ctx context.Context, request *types.LLMRequest, cycleState *types.CycleState, pods []types.Pod) []types.Pod {
	logger := log.FromContext(ctx)
	filteredPods := pods
	logger.V(logutil.DEBUG).Info("Before running filter plugins", "pods", filteredPods)

	for _, filter := range p.filters {
		logger.V(logutil.VERBOSE).Info("Running filter plugin", "plugin", filter.TypedName())
		before := time.Now()
		filteredPods = filter.Filter(ctx, cycleState, request, filteredPods)
		metrics.RecordPluginProcessingLatency(FilterExtensionPoint, filter.TypedName().Type, filter.TypedName().Name, time.Since(before))
		logger.V(logutil.DEBUG).Info("Completed running filter plugin successfully", "plugin", filter.TypedName(), "pods", filteredPods)
		if len(filteredPods) == 0 {
			break
		}
	}
	logger.V(logutil.VERBOSE).Info("Completed running filter plugins successfully")

	return filteredPods
}

func (p *SchedulerProfile) runScorerPlugins(ctx context.Context, request *types.LLMRequest, cycleState *types.CycleState, pods []types.Pod) map[types.Pod]float64 {
	logger := log.FromContext(ctx)
	logger.V(logutil.DEBUG).Info("Before running scorer plugins", "pods", pods)

	weightedScorePerPod := make(map[types.Pod]float64, len(pods))
	for _, pod := range pods {
		weightedScorePerPod[pod] = float64(0) // initialize weighted score per pod with 0 value
	}
	// Iterate through each scorer in the chain and accumulate the weighted scores.
	for _, scorer := range p.scorers {
		logger.V(logutil.VERBOSE).Info("Running scorer plugin", "plugin", scorer.TypedName())
		before := time.Now()
		scores := scorer.Score(ctx, cycleState, request, pods)
		metrics.RecordPluginProcessingLatency(ScorerExtensionPoint, scorer.TypedName().Type, scorer.TypedName().Name, time.Since(before))
		for pod, score := range scores { // weight is relative to the sum of weights
			logger.V(logutil.DEBUG).Info("Calculated score", "plugin", scorer.TypedName(), "endpoint", pod.GetPod().NamespacedName, "score", score)
			weightedScorePerPod[pod] += enforceScoreRange(score) * float64(scorer.Weight())
		}
		logger.V(logutil.DEBUG).Info("Completed running scorer plugin successfully", "plugin", scorer.TypedName())
	}
	logger.V(logutil.VERBOSE).Info("Completed running scorer plugins successfully")

	return weightedScorePerPod
}

func (p *SchedulerProfile) runPickerPlugin(ctx context.Context, cycleState *types.CycleState, weightedScorePerPod map[types.Pod]float64) *types.ProfileRunResult {
	logger := log.FromContext(ctx)
	scoredPods := make([]*types.ScoredPod, len(weightedScorePerPod))
	i := 0
	for pod, score := range weightedScorePerPod {
		scoredPods[i] = &types.ScoredPod{Pod: pod, Score: score}
		i++
	}
	logger.V(logutil.VERBOSE).Info("Running picker plugin", "plugin", p.picker.TypedName())
	logger.V(logutil.DEBUG).Info("Candidate pods for picking", "pods-weighted-score", scoredPods)
	before := time.Now()
	result := p.picker.Pick(ctx, cycleState, scoredPods)
	metrics.RecordPluginProcessingLatency(PickerExtensionPoint, p.picker.TypedName().Type, p.picker.TypedName().Name, time.Since(before))
	logger.V(logutil.DEBUG).Info("Completed running picker plugin successfully", "plugin", p.picker.TypedName(), "result", result)

	return result
}

func enforceScoreRange(score float64) float64 {
	if score < 0 {
		return 0
	}
	if score > 1 {
		return 1
	}
	return score
}
