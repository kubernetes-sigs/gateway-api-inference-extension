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

package usagelimits

import (
	"context"
	"encoding/json"
	"fmt"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

const StaticUsageLimitPolicyType = "static-usage-limit-policy"

// staticPolicyConfig is the JSON configuration for the static usage limit policy.
type staticPolicyConfig struct {
	// Threshold is the fixed ceiling returned for all priorities.
	// Defaults to 1.0 (no gating).
	Threshold *float64 `json:"threshold,omitempty"`
}

// StaticPolicyFactory creates a StaticUsageLimitPolicy from JSON config.
// If no threshold is specified, it defaults to 1.0 (no gating).
func StaticPolicyFactory(name string, rawConfig json.RawMessage, _ plugin.Handle) (plugin.Plugin, error) {
	threshold := 1.0
	if len(rawConfig) > 0 {
		var cfg staticPolicyConfig
		if err := json.Unmarshal(rawConfig, &cfg); err != nil {
			return nil, fmt.Errorf("failed to parse static usage limit policy config: %w", err)
		}
		if cfg.Threshold != nil {
			threshold = *cfg.Threshold
		}
	}
	return NewConstPolicy(name, threshold), nil
}

// DefaultPolicy returns the default UsageLimitPolicy with threshold 1.0 (no gating).
// Useful for test cases.
func DefaultPolicy() flowcontrol.UsageLimitPolicy {
	return NewConstPolicy(StaticUsageLimitPolicyType, 1.0)
}

// NewConstPolicy implements a UsageLimitPolicy that returns a fixed threshold for all priorities.
func NewConstPolicy(usageLimitName string, threshold float64) flowcontrol.UsageLimitPolicy {
	return NewPolicyFunc(usageLimitName, func(_ context.Context, _ float64, priorities []int) []float64 {
		result := make([]float64, len(priorities))
		for i := range result {
			result[i] = threshold
		}
		return result
	})
}

// NewPolicyFunc implements a UsageLimitPolicy with a single func.
func NewPolicyFunc(usageLimitName string, f func(ctx context.Context, saturation float64, priorities []int) (ceilings []float64)) flowcontrol.UsageLimitPolicy {
	return &usageLimitPolicyFunc{
		tpe:  fmt.Sprint(usageLimitName, "-type"),
		name: usageLimitName,
		f:    f,
	}
}

type usageLimitPolicyFunc struct {
	tpe  string
	name string
	f    func(ctx context.Context, saturation float64, priorities []int) (ceilings []float64)
}

func (u usageLimitPolicyFunc) TypedName() plugin.TypedName {
	return plugin.TypedName{
		Type: u.tpe,
		Name: u.name,
	}
}

func (u usageLimitPolicyFunc) ComputeLimit(ctx context.Context, saturation float64, priorities []int) (ceilings []float64) {
	return u.f(ctx, saturation, priorities)
}
