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

import "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins"

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
