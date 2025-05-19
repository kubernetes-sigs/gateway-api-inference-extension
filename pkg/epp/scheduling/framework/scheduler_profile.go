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
)

// NewSchedulerProfile creates a new SchedulerProfile object and returns its pointer.
func NewSchedulerProfile() *SchedulerProfile {
	return &SchedulerProfile{
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
	preCyclePlugins     []PreCycle
	filters             []Filter
	scorers             []*WeightedScorer
	picker              Picker
	postCyclePlugins    []PostCycle
	PostResponsePlugins []PostResponse // TODO this field should get out of the scheduler
}

// PreCyclePlugins returns the SchedulerProfile's PreCycle plugins.
func (p *SchedulerProfile) PreCyclePlugins() []PreCycle {
	return p.preCyclePlugins
}

// Filters returns the SchedulerProfile's Filter plugins.
func (p *SchedulerProfile) Filters() []Filter {
	return p.filters
}

// Scorers returns the SchedulerProfile's Scorer plugins.
func (p *SchedulerProfile) Scorers() []*WeightedScorer {
	return p.scorers
}

// Picker returns the SchedulerProfile's Picker plugin.
func (p *SchedulerProfile) Picker() Picker {
	return p.picker
}

// PostCyclePlugins returns the SchedulerProfile's PostCycle plugins.
func (p *SchedulerProfile) PostCyclePlugins() []PostCycle {
	return p.postCyclePlugins
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
