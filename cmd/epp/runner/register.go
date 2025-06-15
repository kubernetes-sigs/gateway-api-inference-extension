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

package runner

import (
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/registry"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/filter"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/multi/prefix"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/picker"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/profile"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/scorer"
)

// RegisterAllPlgugins registers the factory functions of all known plugins
func RegisterAllPlgugins() {
	plugins := map[string]registry.Factory{
		filter.LeastKVCacheFilterName:    filter.LeastKVCacheFilterFactory,
		filter.LeastQueueFilterName:      filter.LeastQueueFilterFactory,
		filter.LoraAffinityFilterName:    filter.LoraAffinityFilterFactory,
		filter.LowQueueFilterName:        filter.LowQueueFilterFactory,
		prefix.PrefixCachePluginName:     prefix.PrefixCachePluginFactory,
		picker.MaxScorePickerName:        picker.MaxScorePickerFactory,
		picker.RandomPickerName:          picker.RandomPickerFactory,
		profile.SingleProfileHandlerName: profile.SingleProfileHandlerFactory,
		scorer.KvCacheScorerName:         scorer.KvCacheScorerFactory,
		scorer.QueueScorerName:           scorer.QueueScorerFactory,
	}
	for name, factory := range plugins {
		registry.Register(name, factory)
	}
}
