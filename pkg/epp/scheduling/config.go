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

// SchedulerConfig provides a configuration for the scheduler which includes
// items like filters, scorers, etc that influence routing decisions.
//
// This is not threadsafe and the machinery here does not support dynamically
// changing this at runtime, so this should be set once on startup and not
// changed thereafter.
type SchedulerConfig struct {
	PreSchedulePlugins  []plugins.PreSchedule
	Filters             []plugins.Filter
	Scorers             map[plugins.Scorer]int // map from scorer to weight
	Picker              plugins.Picker
	PostSchedulePlugins []plugins.PostSchedule
}

var defPlugin = &defaultPlugin{}

// When the scheduler is initialized with NewScheduler function, this config will be used as default.
// it's possible to call NewSchedulerWithConfig to pass a different argument.

// For build time plugins changes, it's recommended to change the defaultConfig variable in this file.
var defaultConfig = &SchedulerConfig{
	PreSchedulePlugins:  []plugins.PreSchedule{},
	Filters:             []plugins.Filter{defPlugin},
	Scorers:             map[plugins.Scorer]int{},
	Picker:              defPlugin,
	PostSchedulePlugins: []plugins.PostSchedule{},
}
