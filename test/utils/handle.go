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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
)

// testHandle is an implmentation of plugins.Handle for test purposes
type testHandle struct {
	ctx context.Context
	plugins.HandlePlugins
}

// Context returns a context the plugins can use, if they need one
func (h *testHandle) Context() context.Context {
	return h.ctx
}

func (h *testHandle) PodList() []types.NamespacedName {
	return []types.NamespacedName{}
}

type testHandlePlugins struct {
	plugins map[string]plugins.Plugin
}

func (h *testHandlePlugins) Plugin(name string) plugins.Plugin {
	return h.plugins[name]
}

func (h *testHandlePlugins) AddPlugin(name string, plugin plugins.Plugin) {
	h.plugins[name] = plugin
}

func (h *testHandlePlugins) GetAllPlugins() []plugins.Plugin {
	result := make([]plugins.Plugin, 0)
	for _, plugin := range h.plugins {
		result = append(result, plugin)
	}
	return result
}

func (h *testHandlePlugins) GetAllPluginsWithNames() map[string]plugins.Plugin {
	return h.plugins
}

func NewTestHandle(ctx context.Context) plugins.Handle {
	return &testHandle{
		ctx: ctx,
		HandlePlugins: &testHandlePlugins{
			plugins: map[string]plugins.Plugin{},
		},
	}
}
