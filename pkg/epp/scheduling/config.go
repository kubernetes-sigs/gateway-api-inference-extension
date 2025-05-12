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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins/prefix"
)

// NewSchedulerConfig creates a new SchedulerConfig object with the given plugins.
func NewSchedulerConfig(preSchedulePlugins []plugins.PreSchedule, filters []plugins.Filter, scorers map[plugins.Scorer]int,
	picker plugins.Picker, postSchedulePlugins []plugins.PostSchedule, postResponsePlugins []plugins.PostResponse, opts ...ConfigOption) *SchedulerConfig {
	config := &SchedulerConfig{
		preSchedulePlugins:  preSchedulePlugins,
		filters:             filters,
		scorers:             scorers,
		picker:              picker,
		postSchedulePlugins: postSchedulePlugins,
		postResponsePlugins: postResponsePlugins,
	}
	for _, opt := range opts {
		opt(config)
	}
	return config
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

type ConfigOption func(*SchedulerConfig)

// TODO(https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/813): Replace this
// with a more generic way to add plugins.
func AddPrefixPlugin(prefixConfig prefix.Config, weight int) ConfigOption {
	return func(cfg *SchedulerConfig) {
		prefixPlugin := prefix.New(prefixConfig)
		cfg.preSchedulePlugins = append(cfg.preSchedulePlugins, prefixPlugin)
		cfg.postSchedulePlugins = append(cfg.postSchedulePlugins, prefixPlugin)
		cfg.scorers[prefixPlugin] = weight
	}
}
