package usagelimits

import (
	"context"
	"fmt"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// DefaultPolicy returns the default UsageLimitPolicy, which always returns 1.0 (no gating).
func DefaultPolicy() flowcontrol.UsageLimitPolicy {
	return NewConstPolicy("default-usage-limit-policy", 1.0)
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
