package ordering

import (
	"encoding/json"
	"fmt"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"
)

const (
	CustomDataOrderingPolicyType = "custom-data-ordering-policy"
)

type customOrderingParameters struct {
	Keys []customOrderingParameter `json:"keys"`
}

type customOrderingParameter struct {
	Key        string  `json:"key"`
	Direction  string  `json:"direction"`
	DefaultVal float64 `json:"default_value"`
}

func CustomDataOrderingPolicyFactory(name string, parameters json.RawMessage, _ plugin.Handle) (plugin.Plugin, error) {
	params := &customOrderingParameters{}
	if err := json.Unmarshal(parameters, params); err != nil {
		return nil, fmt.Errorf("failed to parse the parameters of the '%s' ordering policy - %w", CustomDataOrderingPolicyType, err)
	}
	return newCustomDataOrderingPolicy(params).withName(name), nil
}

type orderingKey struct {
	name      string
	direction int // 1 for ascending, -1 for descending
	defaultv  float64
}

type CustomDataOrderingPolicy struct {
	name string
	keys []orderingKey
}

var _ flowcontrol.OrderingPolicy = &CustomDataOrderingPolicy{}

func newCustomDataOrderingPolicy(params *customOrderingParameters) *CustomDataOrderingPolicy {
	if params == nil {
		return &CustomDataOrderingPolicy{
			name: CustomDataOrderingPolicyType,
			keys: []orderingKey{},
		}
	}
	keys := make([]orderingKey, 0, len(params.Keys))
	for _, p := range params.Keys {
		dir := 1
		if p.Direction == "desc" {
			dir = -1
		}
		keys = append(keys, orderingKey{
			name:      p.Key,
			direction: dir,
			defaultv:  p.DefaultVal,
		})
	}
	return &CustomDataOrderingPolicy{
		name: CustomDataOrderingPolicyType,
		keys: keys,
	}
}

func (p *CustomDataOrderingPolicy) withName(name string) *CustomDataOrderingPolicy {
	if name != "" {
		p.name = name
	}
	return p
}

func (p *CustomDataOrderingPolicy) Name() string {
	return p.name
}

func (p *CustomDataOrderingPolicy) RequiredQueueCapabilities() []flowcontrol.QueueCapability {
	return []flowcontrol.QueueCapability{flowcontrol.CapabilityPriorityConfigurable}
}

func (p *CustomDataOrderingPolicy) TypedName() plugin.TypedName {
	return plugin.TypedName{
		Type: CustomDataOrderingPolicyType,
		Name: p.name,
	}
}

func (p *CustomDataOrderingPolicy) Less(a, b flowcontrol.QueueItemAccessor) bool {
	for _, k := range p.keys {
		aVal := a.OriginalRequest().GetMetadata()[metadata.CustomOrderingNamespace+"."+k.name]
		bVal := b.OriginalRequest().GetMetadata()[metadata.CustomOrderingNamespace+"."+k.name]

		aValF, ok := aVal.(float64)
		if !ok {
			aValF = k.defaultv
		}
		bValF, ok := bVal.(float64)
		if !ok {
			bValF = k.defaultv
		}
		if aValF == bValF {
			continue
		}
		return aValF*float64(k.direction) < bValF*float64(k.direction)

	}
	return false
}
