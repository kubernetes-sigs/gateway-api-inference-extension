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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins/filter"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins/picker"
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

// When the scheduler is initialized with NewScheduler function, this config will be used as default.
// it's possible to call NewSchedulerWithConfig to pass a different argument.

// For build time plugins changes, it's recommended to change the defaultConfig variable in this file.
var defaultConfig = &SchedulerConfig{
	preSchedulePlugins:  []plugins.PreSchedule{},
	filters:             []plugins.Filter{&filter.SheddableRequestFilter{}, filter.LowLatencyFilter},
	scorers:             map[plugins.Scorer]int{},
	picker:              &picker.RandomPicker{},
	postSchedulePlugins: []plugins.PostSchedule{},
}
