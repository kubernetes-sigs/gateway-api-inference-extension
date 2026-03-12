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

// getMetadataValue extracts a numeric value for key from the custom ordering metadata m, or returns defaultVal if missing or invalid.
func getMetadataValue(m any, key string, defaultVal float64) float64 {
	if m == nil {
		return defaultVal
	}
	switch typedMap := m.(type) {
	case map[string]float64:
		if v, ok := typedMap[key]; ok {
			return v
		}
	case map[string]float32:
		if v, ok := typedMap[key]; ok {
			return float64(v)
		}
	case map[string]int:
		if v, ok := typedMap[key]; ok {
			return float64(v)
		}
	case map[string]int8:
		if v, ok := typedMap[key]; ok {
			return float64(v)
		}
	case map[string]int16:
		if v, ok := typedMap[key]; ok {
			return float64(v)
		}
	case map[string]int32:
		if v, ok := typedMap[key]; ok {
			return float64(v)
		}
	case map[string]int64:
		if v, ok := typedMap[key]; ok {
			return float64(v)
		}
	case map[string]any:
		if v, ok := typedMap[key]; ok {
			switch num := v.(type) {
			case float64:
				return num
			case float32:
				return float64(num)
			case int:
				return float64(num)
			case int8:
				return float64(num)
			case int16:
				return float64(num)
			case int32:
				return float64(num)
			case int64:
				return float64(num)
			}
		}
	}
	return defaultVal
}

func (p *CustomDataOrderingPolicy) Less(a, b flowcontrol.QueueItemAccessor) bool {
	for _, k := range p.keys {
		aVal := getMetadataValue(a.OriginalRequest().GetMetadata()[metadata.CustomOrderingNamespace], k.name, k.defaultv)
		bVal := getMetadataValue(b.OriginalRequest().GetMetadata()[metadata.CustomOrderingNamespace], k.name, k.defaultv)

		if aVal == bVal {
			continue
		}
		return aVal*float64(k.direction) < bVal*float64(k.direction)
	}
	return false
}
