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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins/picker"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins/prefix"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins/scorer"
)

// NewSchedulerConfig creates a new SchedulerConfig object with the given plugins.
func NewSchedulerConfig(preSchedulePlugins []plugins.PreSchedule, filters []plugins.Filter, scorers map[plugins.Scorer]int,
	picker plugins.Picker, postSchedulePlugins []plugins.PostSchedule) *SchedulerConfig {
	return &SchedulerConfig{
		preSchedulePlugins:  preSchedulePlugins,
		filters:             filters,
		scorers:             scorers,
		picker:              picker,
		postSchedulePlugins: postSchedulePlugins,
	}
}

// SchedulerConfig provides a configuration for the scheduler which influence routing decisions.
type SchedulerConfig struct {
	preSchedulePlugins  []plugins.PreSchedule
	filters             []plugins.Filter
	scorers             map[plugins.Scorer]int // map from scorer to weight
	picker              plugins.Picker
	postSchedulePlugins []plugins.PostSchedule
}

var defPlugin = &defaultPlugin{}

// When the scheduler is initialized with NewScheduler function, this config will be used as default.
// it's possible to call NewSchedulerWithConfig to pass a different argument.

// For build time plugins changes, it's recommended to change the defaultConfig variable in this file.
var defaultConfig = &SchedulerConfig{
	preSchedulePlugins:  []plugins.PreSchedule{},
	filters:             []plugins.Filter{defPlugin},
	scorers:             map[plugins.Scorer]int{},
	picker:              defPlugin,
	postSchedulePlugins: []plugins.PostSchedule{},
}

func CreateConfig(opts ...ConfigOption) *SchedulerConfig {
	config := &SchedulerConfig{
		preSchedulePlugins:  []plugins.PreSchedule{},
		postSchedulePlugins: []plugins.PostSchedule{},
		scorers:             map[plugins.Scorer]int{},
		filters:             []plugins.Filter{&sheddableRequestFilterV2{}},
		picker:              &picker.MaxScorePicker{},
	}
	for _, opt := range opts {
		opt(config)
	}
	return config
}

type ConfigOption func(*SchedulerConfig)

func WithPrefixPlugin(prefixConfig prefix.Config) ConfigOption {
	return func(cfg *SchedulerConfig) {
		prefixPlugin := prefix.New(prefixConfig)
		cfg.preSchedulePlugins = append(cfg.preSchedulePlugins, prefixPlugin)
		cfg.postSchedulePlugins = append(cfg.postSchedulePlugins, prefixPlugin)
		cfg.scorers[prefixPlugin] = prefixConfig.Weight
	}
}

func WithQueuePlugin(queueConfig scorer.QueueScorerConfig) ConfigOption {
	return func(cfg *SchedulerConfig) {
		queuePlugin := &scorer.QueueScorer{}
		cfg.scorers[queuePlugin] = queueConfig.Weight
	}
}

func WithKVCachePlugin(kvCacheConfig scorer.KVCacheScorerConfig) ConfigOption {
	return func(cfg *SchedulerConfig) {
		kvCachePlugin := &scorer.KVCacheScorer{}
		cfg.scorers[kvCachePlugin] = kvCacheConfig.Weight
	}
}
