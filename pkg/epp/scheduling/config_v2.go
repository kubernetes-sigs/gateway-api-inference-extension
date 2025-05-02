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

// Package scheduling implements request scheduling algorithms.
package scheduling

import (
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins/filter"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins/picker"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins/prefix"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/plugins/scorer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

func CreateConfig(weights ScorerWeights, prefixConfig prefix.Config) *SchedulerConfig {
	prefixPlugin := prefix.New(prefixConfig)
	queuePlugin := &scorer.QueueScorer{}
	kvCachePlugin := &scorer.KVCacheScorer{}
	config := &SchedulerConfig{
		PreSchedulePlugins:  []plugins.PreSchedule{prefixPlugin},
		PostSchedulePlugins: []plugins.PostSchedule{prefixPlugin},
		Scorers: map[plugins.Scorer]int{
			prefixPlugin:  weights.Prefix,
			queuePlugin:   weights.Queue,
			kvCachePlugin: weights.KVCache,
		},
		Filters: []plugins.Filter{&sheddableRequestFilterV2{}},
		Picker:  &picker.MaxScorePicker{},
	}
	return config
}

type ScorerWeights struct {
	Prefix  int
	Queue   int
	KVCache int
}

type sheddableRequestFilterV2 struct{}

func (p *sheddableRequestFilterV2) Name() string {
	return "sheddableRequestFilterV2"
}

func (p *sheddableRequestFilterV2) Filter(ctx *types.SchedulingContext, pods []types.Pod) []types.Pod {
	if ctx.Req.Critical {
		// Allow all pods to pass through if the request is critical, even if all pods reach their capacity.
		return pods
	}

	// Only allow pods that have enough capacity to handle the request.
	return filter.HasCapacityFilter.Filter(ctx, pods)
}
