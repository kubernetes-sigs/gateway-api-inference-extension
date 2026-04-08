/*
Copyright 2026 The Kubernetes Authors.

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

// Package filtering provides built-in EvictionFilterPolicy implementations.
package filtering

import (
	"encoding/json"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// SheddableFilterType only admits sheddable requests (priority < 0) into the eviction queue.
// This uses the project-wide definition of sheddable from requtil.IsSheddable.
const SheddableFilterType = "eviction-sheddable-filter"

func init() {
	plugin.Register(SheddableFilterType, SheddableFilterFactory)
}

// SheddableFilterFactory creates a SheddableFilter plugin.
func SheddableFilterFactory(name string, _ json.RawMessage, _ plugin.Handle) (plugin.Plugin, error) {
	f := &SheddableFilter{name: SheddableFilterType}
	if name != "" {
		f.name = name
	}
	return f, nil
}

// SheddableFilter only admits sheddable requests (priority < 0) into the eviction queue,
// consistent with the project-wide convention in pkg/epp/util/request.IsSheddable.
type SheddableFilter struct {
	name string
}

var _ flowcontrol.EvictionFilterPolicy = &SheddableFilter{}

func (f *SheddableFilter) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: SheddableFilterType, Name: f.name}
}

func (f *SheddableFilter) Accept(item *flowcontrol.EvictionItem) bool {
	return item.Priority < 0
}
