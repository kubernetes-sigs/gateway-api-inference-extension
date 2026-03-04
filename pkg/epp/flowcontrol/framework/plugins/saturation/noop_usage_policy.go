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

package saturation

import (
	"context"
	"encoding/json"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

const (
	// NoOpUsagePolicyType is the type of the no-op usage limit policy plugin.
	NoOpUsagePolicyType = "noop-usage-policy"
)

// NoOpUsagePolicyFactory creates a new no-op usage limit policy.
func NoOpUsagePolicyFactory(name string, _ json.RawMessage, _ plugin.Handle) (plugin.Plugin, error) {
	return NewNoOpUsagePolicy(), nil
}

// NoOpUsagePolicy is a stub implementation that always returns 1.0 (no gating).
// This is useful as a default/fallback or when usage limiting is disabled.
type NoOpUsagePolicy struct {
	name string
}

var _ flowcontrol.UsageLimitPolicy = &NoOpUsagePolicy{}

// NewNoOpUsagePolicy creates a new no-op usage limit policy.
func NewNoOpUsagePolicy() *NoOpUsagePolicy {
	return &NoOpUsagePolicy{name: NoOpUsagePolicyType}
}

// TypedName returns the type and name tuple of this plugin instance.
func (p *NoOpUsagePolicy) TypedName() plugin.TypedName {
	return plugin.TypedName{
		Type: NoOpUsagePolicyType,
		Name: p.name,
	}
}

// ComputeLimit always returns 1.0 (no gating - allow all traffic).
func (p *NoOpUsagePolicy) ComputeLimit(ctx context.Context, priority int, saturation float64) float64 {
	return 1.0
}
