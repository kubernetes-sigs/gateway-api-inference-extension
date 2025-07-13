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
	"context"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/filter"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/multi/prefix"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/picker"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/profile"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/scorer"
	testfilter "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework/plugins/test/filter"
)

// RegisterAllPlugins registers the factory functions of all known plugins
func RegisterAllPlugins() {
	plugins.Register(filter.DecisionTreeFilterType, filter.DecisionTreeFilterFactory)
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
	// register filter for test purpose only (used in conformance tests)
	plugins.Register(testfilter.HeaderBasedTestingFilterType, testfilter.HeaderBasedTestingFilterFactory)
}

// eppHandle is an implementation of the interface plugins.Handle
type eppHandle struct {
	ctx context.Context
	plugins.HandlePlugins
}

// Context returns a context the plugins can use, if they need one
func (h *eppHandle) Context() context.Context {
	return h.ctx
}

// eppHandlePlugins implements the set of APIs to work with instantiated plugins
type eppHandlePlugins struct {
	plugins map[string]plugins.Plugin
}

// Plugin returns the named plugin instance
func (h *eppHandlePlugins) Plugin(name string) plugins.Plugin {
	return h.plugins[name]
}

// AddPlugin adds a plugin to the set of known plugin instances
func (h *eppHandlePlugins) AddPlugin(name string, plugin plugins.Plugin) {
	h.plugins[name] = plugin
}

// GetAllPlugins returns all of the known plugins
func (h *eppHandlePlugins) GetAllPlugins() []plugins.Plugin {
	result := make([]plugins.Plugin, 0)
	for _, plugin := range h.plugins {
		result = append(result, plugin)
	}
	return result
}

// GetAllPluginsWithNames returns al of the known plugins with their names
func (h *eppHandlePlugins) GetAllPluginsWithNames() map[string]plugins.Plugin {
	return h.plugins
}

func newEppHandle(ctx context.Context) *eppHandle {
	return &eppHandle{
		ctx: ctx,
		HandlePlugins: &eppHandlePlugins{
			plugins: map[string]plugins.Plugin{},
		},
	}
}
