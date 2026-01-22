/*
Copyright 2024 The Kubernetes Authors.

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

package utils

import (
	"context"

	"k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugin"
)

// testHandle is an implmentation of plugin.Handle for test purposes
type testHandle struct {
	ctx context.Context
	plugin.HandlePlugins
}

// Context returns a context the plugins can use, if they need one
func (h *testHandle) Context() context.Context {
	return h.ctx
}

func (h *testHandle) PodList() []types.NamespacedName {
	return []types.NamespacedName{}
}

type testHandlePlugins struct {
	plugins map[string]plugin.Plugin
}

func (h *testHandlePlugins) Plugin(name string) plugin.Plugin {
	return h.plugins[name]
}

func (h *testHandlePlugins) AddPlugin(name string, plugin plugin.Plugin) {
	h.plugins[name] = plugin
}

func (h *testHandlePlugins) GetAllPlugins() []plugin.Plugin {
	result := make([]plugin.Plugin, 0)
	for _, plugin := range h.plugins {
		result = append(result, plugin)
	}
	return result
}

func (h *testHandlePlugins) GetAllPluginsWithNames() map[string]plugin.Plugin {
	return h.plugins
}

func NewTestHandle(ctx context.Context) plugin.Handle {
	return &testHandle{
		ctx: ctx,
		HandlePlugins: &testHandlePlugins{
			plugins: map[string]plugin.Plugin{},
		},
	}
}
