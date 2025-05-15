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
	"fmt"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins/scorer"
)

// NewSchedulerConfig creates a new SchedulerConfig object and returns its pointer.
func NewSchedulerConfig() *SchedulerConfig {
	return &SchedulerConfig{
		preSchedulePlugins:  []plugins.PreSchedule{},
		filters:             []plugins.Filter{},
		scorers:             map[plugins.Scorer]int{},
		postSchedulePlugins: []plugins.PostSchedule{},
		postResponsePlugins: []plugins.PostResponse{},
		// picker remains nil since config doesn't support multiple pickers
	}
}

// SchedulerConfig provides a configuration for the scheduler which influence routing decisions.
type SchedulerConfig struct {
	preSchedulePlugins  []plugins.PreSchedule
	filters             []plugins.Filter
	scorers             map[plugins.Scorer]int // map from scorer to weight
	picker              plugins.Picker
	postSchedulePlugins []plugins.PostSchedule
	postResponsePlugins []plugins.PostResponse
}

// WithPreSchedulePlugins sets the given plugins as the PreSchedule plugins.
// If the SchedulerConfig has PreSchedule plugins, this call replaces the existing plugins with the given ones.
func (c *SchedulerConfig) WithPreSchedulePlugins(plugins ...plugins.PreSchedule) *SchedulerConfig {
	c.preSchedulePlugins = plugins
	return c
}

// WithFilters sets the given filter plugins as the Filter plugins.
// if the SchedulerConfig has Filter plugins, this call replaces the existing plugins with the given ones.
func (c *SchedulerConfig) WithFilters(filters ...plugins.Filter) *SchedulerConfig {
	c.filters = filters
	return c
}

// WithScorers sets the given scorer plugins as the Scorer plugins.
// if the SchedulerConfig has Scorer plugins, this call replaces the existing plugins with the given ones.
func (c *SchedulerConfig) WithScorers(scorers ...*scorer.WeightedScorer) *SchedulerConfig {
	scorersMap := make(map[plugins.Scorer]int, len(scorers))
	for _, scorer := range scorers {
		scorersMap[scorer.Scorer] = scorer.Weight()
	}
	c.scorers = scorersMap
	return c
}

// WithPicker sets the given picker plugins as the Picker plugin.
// if the SchedulerConfig has Picker plugin, this call replaces the existing plugin with the given one.
func (c *SchedulerConfig) WithPicker(picker plugins.Picker) *SchedulerConfig {
	c.picker = picker
	return c
}

// WithPostSchedulePlugins sets the given plugins as the PostSchedule plugins.
// If the SchedulerConfig has PostSchedule plugins, this call replaces the existing plugins with the given ones.
func (c *SchedulerConfig) WithPostSchedulePlugins(plugins ...plugins.PostSchedule) *SchedulerConfig {
	c.postSchedulePlugins = plugins
	return c
}

// WithPostResponsePlugins sets the given plugins as the PostResponse plugins.
// If the SchedulerConfig has PostResponse plugins, this call replaces the existing plugins with the given ones.
func (c *SchedulerConfig) WithPostResponsePlugins(plugins ...plugins.PostResponse) *SchedulerConfig {
	c.postResponsePlugins = plugins
	return c
}

// AddPlugins adds the given plugins to all scheduler plugins according to the interfaces each plugin implements.
// A plugin may implement more than one scheduler plugin interface.
func (c *SchedulerConfig) AddPlugins(pluginObjects ...plugins.Plugin) error {
	for _, plugin := range pluginObjects {
		if weightedScorer, ok := plugin.(*scorer.WeightedScorer); ok {
			c.scorers[weightedScorer.Scorer] = weightedScorer.Weight()
			plugin = weightedScorer.Scorer // if we got WeightedScorer, unwrap the plugin
		}
		if preSchedulePlugin, ok := plugin.(plugins.PreSchedule); ok {
			c.preSchedulePlugins = append(c.preSchedulePlugins, preSchedulePlugin)
		}
		if filter, ok := plugin.(plugins.Filter); ok {
			c.filters = append(c.filters, filter)
		}
		if picker, ok := plugin.(plugins.Picker); ok {
			if c.picker != nil {
				return fmt.Errorf("failed to set '%s' as picker, already have a registered picker plugin '%s'", picker.Name(), c.picker.Name())
			}
			c.picker = picker
		}
		if postSchedulePlugin, ok := plugin.(plugins.PostSchedule); ok {
			c.postSchedulePlugins = append(c.postSchedulePlugins, postSchedulePlugin)
		}
		if postResponsePlugin, ok := plugin.(plugins.PostResponse); ok {
			c.postResponsePlugins = append(c.postResponsePlugins, postResponsePlugin)
		}
	}
	return nil
}
