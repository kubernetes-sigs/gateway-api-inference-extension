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
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/filter"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/multi/prefix"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/picker"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/profile"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/scorer"
)

// RegisterAllPlgugins registers the factory functions of all known plugins
func RegisterAllPlgugins() {
	plugins.Register(filter.LeastKVCacheFilterType, filter.LeastKVCacheFilterFactory)
	plugins.Register(filter.LeastQueueFilterType, filter.LeastQueueFilterFactory)
	plugins.Register(filter.LoraAffinityFilterType, filter.LoraAffinityFilterFactory)
	plugins.Register(filter.LowQueueFilterType, filter.LowQueueFilterFactory)
	plugins.Register(prefix.PrefixCachePluginType, prefix.PrefixCachePluginFactory)
	plugins.Register(picker.MaxScorePickerType, picker.MaxScorePickerFactory)
	plugins.Register(picker.RandomPickerType, picker.RandomPickerFactory)
	plugins.Register(profile.SingleProfileHandlerType, profile.SingleProfileHandlerFactory)
	plugins.Register(scorer.KvCacheScorerType, scorer.KvCacheScorerFactory)
	plugins.Register(scorer.QueueScorerType, scorer.QueueScorerFactory)
}

// eppHandle is a temporary implementation of the interface plugins.Handle
type eppHandle struct {
}
