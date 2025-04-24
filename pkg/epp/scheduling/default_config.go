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
)

var defPlugin = &defaultPlugin{}

var defaultConfig = &SchedulerConfig{
	preSchedulePlugins:  []plugins.PreSchedule{},
	scorers:             []plugins.Scorer{},
	filters:             []plugins.Filter{defPlugin},
	postSchedulePlugins: []plugins.PostSchedule{},
	picker:              defPlugin,
}
