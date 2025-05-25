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
	"fmt"
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// NewSchedulerProfile creates a new SchedulerProfile object and returns its pointer.
func NewSchedulerProfile(name string) *SchedulerProfile {
	return &SchedulerProfile{
		name:                name,
		preCyclePlugins:     []PreCycle{},
		filters:             []Filter{},
		scorers:             []*WeightedScorer{},
		postCyclePlugins:    []PostCycle{},
		PostResponsePlugins: []PostResponse{},
		// picker remains nil since profile doesn't support multiple pickers
	}
}

// SchedulerProfile provides a profile configuration for the scheduler which influence routing decisions.
type SchedulerProfile struct {
	name                string
	preCyclePlugins     []PreCycle
	filters             []Filter
	scorers             []*WeightedScorer
	picker              Picker
	postCyclePlugins    []PostCycle
	PostResponsePlugins []PostResponse // TODO this field should get out of the scheduler
}

// GetName returns profile name.
func (p *SchedulerProfile) GetName() string {
	return p.name
}

// WithPreCyclePlugins sets the given plugins as the PreCycle plugins.
// If the SchedulerProfile has PreCycle plugins, this call replaces the existing plugins with the given ones.
func (p *SchedulerProfile) WithPreCyclePlugins(plugins ...PreCycle) *SchedulerProfile {
	p.preCyclePlugins = plugins
	return p
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

// WithPostCyclePlugins sets the given plugins as the PostCycle plugins.
// If the SchedulerProfile has PostCycle plugins, this call replaces the existing plugins with the given ones.
func (p *SchedulerProfile) WithPostCyclePlugins(plugins ...PostCycle) *SchedulerProfile {
	p.postCyclePlugins = plugins
	return p
}

// WithPostResposePlugins sets the given plugins as the PostResponse plugins.
// If the SchedulerProfile has PostResponse plugins, this call replaces the existing plugins with the given ones.
// TODO remove PostResponse from SchedulerProfile, including this function
func (p *SchedulerProfile) WithPostResponsePlugins(plugins ...PostResponse) *SchedulerProfile {
	p.PostResponsePlugins = plugins
	return p
}

// AddPlugins adds the given plugins to all scheduler plugins according to the interfaces each plugin implements.
// A plugin may implement more than one scheduler plugin interface.
// Special Case: In order to add a scorer, one must use the scorer.NewWeightedScorer function in order to provide a weight.
// if a scorer implements more than one interface, supplying a WeightedScorer is sufficient. The function will take the internal
// scorer object and register it to all interfaces it implements.
func (p *SchedulerProfile) AddPlugins(pluginObjects ...Plugin) error {
	for _, plugin := range pluginObjects {
		if weightedScorer, ok := plugin.(*WeightedScorer); ok {
			p.scorers = append(p.scorers, weightedScorer)
			plugin = weightedScorer.Scorer // if we got WeightedScorer, unwrap the plugin
		} else if scorer, ok := plugin.(Scorer); ok { // if we got a Scorer instead of WeightedScorer that's an error.
			return fmt.Errorf("failed to register scorer '%s' without a weight. follow function documentation to register a scorer", scorer.Name())
		}
		if preCyclePlugin, ok := plugin.(PreCycle); ok {
			p.preCyclePlugins = append(p.preCyclePlugins, preCyclePlugin)
		}
		if filter, ok := plugin.(Filter); ok {
			p.filters = append(p.filters, filter)
		}
		if picker, ok := plugin.(Picker); ok {
			if p.picker != nil {
				return fmt.Errorf("failed to set '%s' as picker, already have a registered picker plugin '%s'", picker.Name(), p.picker.Name())
			}
			p.picker = picker
		}
		if postCyclePlugin, ok := plugin.(PostCycle); ok {
			p.postCyclePlugins = append(p.postCyclePlugins, postCyclePlugin)
		}
		if postResponsePlugin, ok := plugin.(PostResponse); ok {
			p.PostResponsePlugins = append(p.PostResponsePlugins, postResponsePlugin)
		}
	}
	return nil
}

// RunCycle runs a SchedulerProfile cycle. In other words, it invokes all the SchedulerProfile plugins in this
// order - PreCyclePlugins, Filters, Scorers, Picker, PostCyclePlugins. After completing all, it returns the result.
func (p *SchedulerProfile) RunCycle(ctx *types.SchedulingContext) (*types.Result, error) {
	p.runPreCyclePlugins(ctx)

	pods := p.runFilterPlugins(ctx)
	if len(pods) == 0 {
		return nil, errutil.Error{Code: errutil.Internal, Msg: "no pods available for the given request"}
	}
	// if we got here, there is at least one pod to score
	weightedScorePerPod := p.runScorerPlugins(ctx, pods)

	result := p.runPickerPlugin(ctx, weightedScorePerPod)

	p.runPostCyclePlugins(ctx, result)

	return result, nil
}

func (p *SchedulerProfile) runPreCyclePlugins(ctx *types.SchedulingContext) {
	for _, plugin := range p.preCyclePlugins {
		ctx.Logger.V(logutil.DEBUG).Info("Running pre-cycle plugin", "plugin", plugin.Name())
		before := time.Now()
		plugin.PreCycle(ctx)
		metrics.RecordSchedulerPluginProcessingLatency(PreCyclePluginType, plugin.Name(), time.Since(before))
	}
}

func (p *SchedulerProfile) runFilterPlugins(ctx *types.SchedulingContext) []types.Pod {
	loggerDebug := ctx.Logger.V(logutil.DEBUG)
	filteredPods := ctx.PodsSnapshot
	loggerDebug.Info("Before running filter plugins", "pods", filteredPods)

	for _, filter := range p.filters {
		loggerDebug.Info("Running filter plugin", "plugin", filter.Name())
		before := time.Now()
		filteredPods = filter.Filter(ctx, filteredPods)
		metrics.RecordSchedulerPluginProcessingLatency(FilterPluginType, filter.Name(), time.Since(before))
		loggerDebug.Info("Filter plugin result", "plugin", filter.Name(), "pods", filteredPods)
		if len(filteredPods) == 0 {
			break
		}
	}
	loggerDebug.Info("After running filter plugins")

	return filteredPods
}

func (p *SchedulerProfile) runScorerPlugins(ctx *types.SchedulingContext, pods []types.Pod) map[types.Pod]float64 {
	loggerDebug := ctx.Logger.V(logutil.DEBUG)
	loggerDebug.Info("Before running scorer plugins", "pods", pods)

	weightedScorePerPod := make(map[types.Pod]float64, len(pods))
	for _, pod := range pods {
		weightedScorePerPod[pod] = float64(0) // initialize weighted score per pod with 0 value
	}
	// Iterate through each scorer in the chain and accumulate the weighted scores.
	for _, scorer := range p.scorers {
		loggerDebug.Info("Running scorer", "scorer", scorer.Name())
		before := time.Now()
		scores := scorer.Score(ctx, pods)
		metrics.RecordSchedulerPluginProcessingLatency(ScorerPluginType, scorer.Name(), time.Since(before))
		for pod, score := range scores { // weight is relative to the sum of weights
			weightedScorePerPod[pod] += score * float64(scorer.Weight())
		}
		loggerDebug.Info("After running scorer", "scorer", scorer.Name())
	}
	loggerDebug.Info("After running scorer plugins")

	return weightedScorePerPod
}

func (p *SchedulerProfile) runPickerPlugin(ctx *types.SchedulingContext, weightedScorePerPod map[types.Pod]float64) *types.Result {
	loggerDebug := ctx.Logger.V(logutil.DEBUG)
	scoredPods := make([]*types.ScoredPod, len(weightedScorePerPod))
	i := 0
	for pod, score := range weightedScorePerPod {
		scoredPods[i] = &types.ScoredPod{Pod: pod, Score: score}
		i++
	}

	loggerDebug.Info("Before running picker plugin", "pods weighted score", fmt.Sprint(weightedScorePerPod))
	before := time.Now()
	result := p.picker.Pick(ctx, scoredPods)
	metrics.RecordSchedulerPluginProcessingLatency(PickerPluginType, p.picker.Name(), time.Since(before))
	loggerDebug.Info("After running picker plugin", "result", result)

	return result
}

func (p *SchedulerProfile) runPostCyclePlugins(ctx *types.SchedulingContext, res *types.Result) {
	for _, plugin := range p.postCyclePlugins {
		ctx.Logger.V(logutil.DEBUG).Info("Running post-cycle plugin", "plugin", plugin.Name())
		before := time.Now()
		plugin.PostCycle(ctx, res)
		metrics.RecordSchedulerPluginProcessingLatency(PostCyclePluginType, plugin.Name(), time.Since(before))
	}
}
