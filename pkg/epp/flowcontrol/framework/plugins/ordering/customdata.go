package ordering

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"

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
	Key        string `json:"key"`
	Direction  string `json:"direction"`
	Type       string `json:"type"` // data type of the key value: "float" or "int"
	DefaultVal any    `json:"default_value"`
}

func CustomDataOrderingPolicyFactory(name string, parameters json.RawMessage, _ plugin.Handle) (plugin.Plugin, error) {
	params := &customOrderingParameters{}
	if err := json.Unmarshal(parameters, params); err != nil {
		return nil, fmt.Errorf("failed to parse the parameters of the '%s' ordering policy - %w", CustomDataOrderingPolicyType, err)
	}
	p, err := newCustomDataOrderingPolicy(params)
	if err != nil {
		return nil, err
	}
	return p.withName(name), nil
}

type orderingKey struct {
	name      string
	direction int // 1 for ascending, -1 for descending
	defaultv  any
}

type CustomDataOrderingPolicy struct {
	name string
	keys []orderingKey
}

var _ flowcontrol.OrderingPolicy = &CustomDataOrderingPolicy{}

func newCustomDataOrderingPolicy(params *customOrderingParameters) (*CustomDataOrderingPolicy, error) {
	if params == nil {
		return &CustomDataOrderingPolicy{
			name: CustomDataOrderingPolicyType,
			keys: []orderingKey{},
		}, nil
	}
	keys := make([]orderingKey, 0, len(params.Keys))
	for _, p := range params.Keys {
		dir := 1
		switch strings.ToLower(p.Direction) {
		case "desc":
			dir = -1
		case "asc", "": // default to ascending
			dir = 1
		default:
			return nil, fmt.Errorf("invalid direction '%s' for key '%s', must be 'asc' or 'desc'", p.Direction, p.Key)
		}
		defaultv, err := parseDefaultVal(p.Type, p.DefaultVal, p.Key)
		if err != nil {
			return nil, err
		}
		keys = append(keys, orderingKey{
			name:      p.Key,
			direction: dir,
			defaultv:  defaultv,
		})
	}
	return &CustomDataOrderingPolicy{
		name: CustomDataOrderingPolicyType,
		keys: keys,
	}, nil
}

// parseDefaultVal converts DefaultVal to the concrete type implied by typeStr ("float" or "int").
// For "float": accepts JSON numbers, int-like values, or strings that parse as float ("2", "2.0", "2.3").
// For "int": accepts only whole numbers; returns error if value has a non-zero fractional part.
// Empty typeStr defaults to "float".
func parseDefaultVal(typeStr string, v any, keyName string) (any, error) {
	t := strings.ToLower(strings.TrimSpace(typeStr))
	if t == "" {
		t = "float"
	}
	switch t {
	case "float":
		return parseDefaultFloat(v, keyName)
	case "int":
		intv, err := parseDefaultFloat(v, keyName)
		if err != nil {

		}
		if intv != math.Trunc(intv) {
			return nil, fmt.Errorf("key %q: default_value %v is not a whole number (int type requires integer)", keyName, intv)
		}
		return int(intv), nil
	default:
		return nil, fmt.Errorf("invalid type %q for key %q, must be 'float' or 'int'", typeStr, keyName)
	}
}

func parseDefaultFloat(v any, keyName string) (float64, error) {
	if v == nil {
		return 0, nil
	}
	switch n := v.(type) {
	case float64:
		return n, nil
	case float32:
		return float64(n), nil
	case int:
		return float64(n), nil
	case string:
		f, err := strconv.ParseFloat(strings.TrimSpace(n), 64)
		if err != nil {
			return 0, fmt.Errorf("key %q: default_value %q is not a valid float", keyName, n)
		}
		return f, nil
	default:
		return 0, fmt.Errorf("key %q: default_value has unsupported type for float", keyName)
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
		switch k.defaultv.(type) {
		case float64:
			aVal := getMetaData(a.OriginalRequest().GetMetadata()[metadata.CustomOrderingNamespace], k.name, k.defaultv.(float64))
			bVal := getMetaData(b.OriginalRequest().GetMetadata()[metadata.CustomOrderingNamespace], k.name, k.defaultv.(float64))
			if aVal == bVal {
				continue
			}
			return isLess(aVal, bVal, k.direction)
		case int:
			aVal := getMetaData(a.OriginalRequest().GetMetadata()[metadata.CustomOrderingNamespace], k.name, k.defaultv.(int))
			bVal := getMetaData(b.OriginalRequest().GetMetadata()[metadata.CustomOrderingNamespace], k.name, k.defaultv.(int))
			if aVal == bVal {
				continue
			}
			return isLess(aVal, bVal, k.direction)
		}
	}
	return false
}

func getMetaData[T ~float64 | ~int](m any, key string, dv T) T {
	if m == nil {
		return dv
	}
	switch typedMap := m.(type) {
	case map[string]T:
		if v, ok := typedMap[key]; ok {
			return v
		}
	case map[string]any:
		if v, ok := typedMap[key]; ok {
			switch num := v.(type) {
			case float64:
				return T(num)
			case int:
				return T(num)
			}
		}
	}
	return dv
}

func isLess[T ~float64 | ~int](a, b T, dir int) bool {
	return a*T(dir) < b*T(dir)
}
