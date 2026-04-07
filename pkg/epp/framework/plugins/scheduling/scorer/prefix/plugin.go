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

package prefix

import (
	"context"
	"encoding/json"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrprefix "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

// Plugin implements the prefix cache aware scoring logic.
type Plugin struct {
	typedName plugin.TypedName
}

// compile-time type assertions
var (
	_ framework.Scorer = &Plugin{}
)

const (
	// Type is the unique identifier for the prefix cache scorer plugin.
	PrefixCacheScorerPluginType = "prefix-cache-scorer"
)

// PrefixCachePluginFactory defines the factory function for the Prefix plugin.
func PrefixCachePluginFactory(name string, _ json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
	p, err := New(handle.Context())
	if err != nil {
		return nil, err
	}
	if name != "" {
		p = p.WithName(name)
	}
	return p, nil
}

// New initializes a new prefix Plugin.
func New(_ context.Context) (*Plugin, error) {
	return &Plugin{
		typedName: plugin.TypedName{
			Type: PrefixCacheScorerPluginType,
			Name: PrefixCacheScorerPluginType,
		},
	}, nil
}

// TypedName returns the type and name of this plugin instance.
func (p *Plugin) TypedName() plugin.TypedName {
	return p.typedName
}

// Category returns the preference the scorer applies (Affinity).
func (p *Plugin) Category() framework.ScorerCategory {
	return framework.Affinity
}

// WithName sets the name of the plugin instance.
func (p *Plugin) WithName(name string) *Plugin {
	p.typedName.Name = name
	return p
}

// Produces returns the data produced by the plugin.
func (p *Plugin) Produces() map[string]any {
	return map[string]any{}
}

// Consumes returns the data consumed by the plugin.
func (p *Plugin) Consumes() map[string]any {
	return map[string]any{attrprefix.PrefixCacheMatchInfoKey: attrprefix.PrefixCacheMatchInfo{}}
}

// Score returns the scoring result for the given list of pods based on prefix cache match info.
func (p *Plugin) Score(ctx context.Context, _ *framework.CycleState, _ *framework.InferenceRequest, endpoints []framework.Endpoint) map[framework.Endpoint]float64 {
	scores := make(map[framework.Endpoint]float64, len(endpoints))
	logger := log.FromContext(ctx)

	for _, endpoint := range endpoints {
		// Default to score 0 if PrefixCacheMatchInfo is missing or invalid.
		scores[endpoint] = 0.0
		info, ok := endpoint.Get(attrprefix.PrefixCacheMatchInfoKey)
		if !ok {
			logger.V(logutil.DEFAULT).Error(nil, "PrefixCacheMatchInfo not found for endpoint, assigning score 0", "endpoint", endpoint)
			continue
		}

		if prefixMatchInfo, ok := info.(*attrprefix.PrefixCacheMatchInfo); ok {
			if prefixMatchInfo.TotalBlocks() != 0 {
				scores[endpoint] = float64(prefixMatchInfo.MatchBlocks()) / float64(prefixMatchInfo.TotalBlocks())
			}
		} else {
			logger.V(logutil.DEFAULT).Error(nil, "PrefixCacheMatchInfo has unexpected type, assigning score 0", "endpoint", endpoint)
		}
	}
	return scores
}
