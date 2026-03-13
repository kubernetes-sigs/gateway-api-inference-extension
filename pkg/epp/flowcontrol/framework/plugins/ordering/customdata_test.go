package ordering

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol/mocks"
)

var pParamsJson = `{
	"keys": [
		{
			"key": "score1",
			"direction": "asc",
			"default_value": 2
		},
		{
			"key": "score2",
			"direction": "asc",
			"default_value": 5
		}
	]
}`

func TestCustomPolicy_Factory(t *testing.T) {
	t.Parallel()
	p, err := CustomDataOrderingPolicyFactory("my-test-policy", json.RawMessage(pParamsJson), nil)

	assert.NoError(t, err)
	assert.Equal(t, "my-test-policy", p.TypedName().Name)
	assert.Equal(t, CustomDataOrderingPolicyType, p.TypedName().Type)
}

func TestCustomPolicy_Name(t *testing.T) {
	t.Parallel()
	p, err := newCustomDataOrderingPolicy(nil)
	assert.NoError(t, err)
	assert.Equal(t, CustomDataOrderingPolicyType, p.Name())
}

func TestCustomPolicy_RequiredQueueCapabilities(t *testing.T) {
	t.Parallel()
	p, err := newCustomDataOrderingPolicy(nil)
	assert.NoError(t, err)
	c := p.RequiredQueueCapabilities()
	assert.Len(t, c, 1)
	assert.Equal(t, flowcontrol.CapabilityPriorityConfigurable, c[0])
}

func TestCustomPolicy_NewCustomDataOrderingPolicy_Error(t *testing.T) {
	t.Parallel()
	_, err := newCustomDataOrderingPolicy(&customOrderingParameters{
		Keys: []customOrderingParameter{
			{
				Key:        "score1",
				Direction:  "invalid",
				DefaultVal: 2,
			},
		},
	})
	assert.NotNil(t, err, "error should not be nil")
	assert.Equal(t, "invalid direction 'invalid' for key 'score1', must be 'asc' or 'desc'", err.Error())
}

func TestCustomPolicy_Less(t *testing.T) {
	t.Parallel()
	p, err := newCustomDataOrderingPolicy(&customOrderingParameters{
		Keys: []customOrderingParameter{
			{
				Key:        "score1",
				Direction:  "asc",
				DefaultVal: 2,
			},
		},
	})
	assert.NoError(t, err)

	a := mocks.NewMockQueueItemAccessor(1, "a", testFlowKey)
	b := mocks.NewMockQueueItemAccessor(2, "b", testFlowKey)
	c := mocks.NewMockQueueItemAccessor(3, "c", testFlowKey)

	ma := a.OriginalRequest().GetMetadata()
	ma["ordering.custom"] = map[string]float64{
		"score1": 3.2,
	}

	mb := b.OriginalRequest().GetMetadata()
	mb["ordering.custom"] = map[string]float64{
		"score1": 5.9,
	}

	mc := c.OriginalRequest().GetMetadata()
	mc["ordering.custom"] = map[string]float64{
		"score1": 3.2,
	}

	assert.True(t, p.Less(a, b))  // 3.2 < 5.9
	assert.False(t, p.Less(b, a)) // 5.9 not < 3.2
	assert.False(t, p.Less(a, c)) // 3.2 not < 3.2
}

func TestCustomPolicy_Less_NilData(t *testing.T) {
	t.Parallel()
	p, err := newCustomDataOrderingPolicy(&customOrderingParameters{
		Keys: []customOrderingParameter{
			{
				Key:        "score1",
				Direction:  "asc",
				DefaultVal: 2,
			},
		},
	})
	assert.NoError(t, err)

	a := mocks.NewMockQueueItemAccessor(1, "a", testFlowKey)
	ma := a.OriginalRequest().GetMetadata()
	ma["ordering.custom"] = map[string]float64{
		"score1": 3.2,
	}
	nilItem := mocks.NewMockQueueItemAccessor(1, "nil-item", testFlowKey)

	// nilItem will use the default value of 2 since no value was set in the metadata for ordering.custom.score1
	assert.True(t, p.Less(nilItem, a))        // 2 < 3.2
	assert.False(t, p.Less(a, nilItem))       // 3.2 not < 2
	assert.False(t, p.Less(nilItem, nilItem)) // 2 not < 2
}

func TestCustomPolicy_Less_Desc(t *testing.T) {
	t.Parallel()
	p, err := newCustomDataOrderingPolicy(&customOrderingParameters{
		Keys: []customOrderingParameter{
			{
				Key:        "score1",
				Direction:  "desc",
				DefaultVal: 2,
			},
		},
	})
	assert.NoError(t, err)

	a := mocks.NewMockQueueItemAccessor(1, "a", testFlowKey)
	b := mocks.NewMockQueueItemAccessor(2, "b", testFlowKey)
	c := mocks.NewMockQueueItemAccessor(3, "c", testFlowKey)

	ma := a.OriginalRequest().GetMetadata()
	ma["ordering.custom"] = map[string]float64{
		"score1": 3.2,
	}

	mb := b.OriginalRequest().GetMetadata()
	mb["ordering.custom"] = map[string]float64{
		"score1": 5.9,
	}

	mc := c.OriginalRequest().GetMetadata()
	mc["ordering.custom"] = map[string]float64{
		"score1": 3.2,
	}

	assert.False(t, p.Less(a, b)) // 3.2 not < 5.9 (in descending order)
	assert.True(t, p.Less(b, a))  // 5.9 < 3.2 (in descending order)
	assert.False(t, p.Less(a, c)) // 3.2 not < 3.2
}

func TestCustomPolicy_Less_MultipleKeys(t *testing.T) {
	t.Parallel()
	p, err := newCustomDataOrderingPolicy(&customOrderingParameters{
		Keys: []customOrderingParameter{
			{
				Key:        "score1",
				Direction:  "asc",
				DefaultVal: 2,
			},
			{
				Key:        "score2",
				Direction:  "asc",
				DefaultVal: 5,
			},
		},
	})
	assert.NoError(t, err)

	a := mocks.NewMockQueueItemAccessor(1, "a", testFlowKey)
	b := mocks.NewMockQueueItemAccessor(2, "b", testFlowKey)
	c := mocks.NewMockQueueItemAccessor(3, "c", testFlowKey)
	d := mocks.NewMockQueueItemAccessor(4, "d", testFlowKey)

	ma := a.OriginalRequest().GetMetadata()
	ma["ordering.custom"] = map[string]float64{
		"score1": 3.2,
		"score2": 4.9,
	}

	mb := b.OriginalRequest().GetMetadata()
	mb["ordering.custom"] = map[string]float64{
		"score1": 5.9,
		"score2": 2.9,
	}

	mc := c.OriginalRequest().GetMetadata()
	mc["ordering.custom"] = map[string]float64{
		"score1": 3.2,
		"score2": 6.3,
	}

	md := d.OriginalRequest().GetMetadata()
	md["ordering.custom"] = map[string]float64{
		"score1": 3.2,
		"score2": 4.9,
	}

	assert.True(t, p.Less(a, b))  // 3.2 < 5.9
	assert.False(t, p.Less(b, a)) // 5.9 not < 3.2
	assert.True(t, p.Less(a, c))  // 4.9 < 6.3 (checking score2 since score1 is the same for a and c)
	assert.False(t, p.Less(a, d)) // 3.2 not < 3.2 and 4.9 not < 4.9
}

func TestCustomPolicy_Less_MultipleKeys_Desc(t *testing.T) {
	t.Parallel()
	p, err := newCustomDataOrderingPolicy(&customOrderingParameters{
		Keys: []customOrderingParameter{
			{
				Key:        "score1",
				Direction:  "asc",
				DefaultVal: 2,
			},
			{
				Key:        "score2",
				Direction:  "desc",
				DefaultVal: 5,
			},
		},
	})
	assert.NoError(t, err)

	a := mocks.NewMockQueueItemAccessor(1, "a", testFlowKey)
	b := mocks.NewMockQueueItemAccessor(2, "b", testFlowKey)
	c := mocks.NewMockQueueItemAccessor(3, "c", testFlowKey)
	d := mocks.NewMockQueueItemAccessor(4, "d", testFlowKey)

	ma := a.OriginalRequest().GetMetadata()
	ma["ordering.custom"] = map[string]float64{
		"score1": 3.2,
		"score2": 4.9,
	}

	mb := b.OriginalRequest().GetMetadata()
	mb["ordering.custom"] = map[string]float64{
		"score1": 5.9,
		"score2": 2.9,
	}

	mc := c.OriginalRequest().GetMetadata()
	mc["ordering.custom"] = map[string]float64{
		"score1": 3.2,
		"score2": 6.3,
	}

	md := d.OriginalRequest().GetMetadata()
	md["ordering.custom"] = map[string]float64{
		"score1": 3.2,
		"score2": 4.9,
	}

	assert.True(t, p.Less(a, b))  // 3.2 < 5.9
	assert.False(t, p.Less(b, a)) // 5.9 not < 3.2
	assert.True(t, p.Less(c, a))  // 6.3 > 4.9 (in descending order, checking score2 since score1 is the same for c and a)
	assert.False(t, p.Less(a, d)) // 3.2 not < 3.2 and 4.9 not < 4.9
}

// --- Type field: float ---

func TestCustomPolicy_TypeFloat_EmptyDefaultsToFloat(t *testing.T) {
	t.Parallel()
	// Empty Type defaults to "float"; DefaultVal 2 is accepted as float64
	p, err := newCustomDataOrderingPolicy(&customOrderingParameters{
		Keys: []customOrderingParameter{
			{Key: "x", Direction: "asc", Type: "", DefaultVal: 2},
		},
	})
	assert.NoError(t, err)
	assert.NotNil(t, p)
}

func TestCustomPolicy_TypeFloat_AcceptsStringWholeAndDecimal(t *testing.T) {
	t.Parallel()
	// Type "float" with string "2", "2.0", "2.3" all accepted
	for _, defaultVal := range []any{"2", "2.0", "2.3"} {
		p, err := newCustomDataOrderingPolicy(&customOrderingParameters{
			Keys: []customOrderingParameter{
				{Key: "x", Direction: "asc", Type: "float", DefaultVal: defaultVal},
			},
		})
		assert.NoError(t, err, "default_value %v should be accepted for type float", defaultVal)
		assert.NotNil(t, p)
	}
}

func TestCustomPolicy_TypeFloat_JSONStringDefault(t *testing.T) {
	t.Parallel()
	// Factory with JSON: type "float", default_value as string "2.0"
	jsonParams := `{"keys": [{"key": "score1", "direction": "asc", "type": "float", "default_value": "2.0"}]}`
	p, err := CustomDataOrderingPolicyFactory("p", json.RawMessage(jsonParams), nil)
	assert.NoError(t, err)
	assert.NotNil(t, p)

	a := mocks.NewMockQueueItemAccessor(1, "a", testFlowKey)
	b := mocks.NewMockQueueItemAccessor(2, "b", testFlowKey)
	ma := a.OriginalRequest().GetMetadata()
	ma["ordering.custom"] = map[string]float64{"score1": 1.5}
	// b has no score1 -> uses default 2.0
	assert.True(t, p.(*CustomDataOrderingPolicy).Less(a, b))  // 1.5 < 2.0
	assert.False(t, p.(*CustomDataOrderingPolicy).Less(b, a)) // 2.0 not < 1.5
}

// --- Type field: int ---

func TestCustomPolicy_TypeInt_AcceptsWholeNumbers(t *testing.T) {
	t.Parallel()
	// Type "int" with 2, "2", "2.0", "5.0" accepted
	for _, defaultVal := range []any{2, "2", "2.0", "5.0"} {
		p, err := newCustomDataOrderingPolicy(&customOrderingParameters{
			Keys: []customOrderingParameter{
				{Key: "x", Direction: "asc", Type: "int", DefaultVal: defaultVal},
			},
		})
		assert.NoError(t, err, "default_value %v should be accepted for type int", defaultVal)
		assert.NotNil(t, p)
	}
}

func TestCustomPolicy_TypeInt_RejectsNonWholeString(t *testing.T) {
	t.Parallel()
	_, err := newCustomDataOrderingPolicy(&customOrderingParameters{
		Keys: []customOrderingParameter{
			{Key: "x", Direction: "asc", Type: "int", DefaultVal: "4.6"},
		},
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not a whole number")
}

func TestCustomPolicy_TypeInt_RejectsNonWholeFloat64(t *testing.T) {
	t.Parallel()
	_, err := newCustomDataOrderingPolicy(&customOrderingParameters{
		Keys: []customOrderingParameter{
			{Key: "x", Direction: "asc", Type: "int", DefaultVal: 2.3},
		},
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not a whole number")
}

func TestCustomPolicy_TypeInt_LessWithIntKey(t *testing.T) {
	t.Parallel()
	p, err := newCustomDataOrderingPolicy(&customOrderingParameters{
		Keys: []customOrderingParameter{
			{Key: "priority", Direction: "asc", Type: "int", DefaultVal: 5},
		},
	})
	assert.NoError(t, err)

	a := mocks.NewMockQueueItemAccessor(1, "a", testFlowKey)
	b := mocks.NewMockQueueItemAccessor(2, "b", testFlowKey)
	ma := a.OriginalRequest().GetMetadata()
	ma["ordering.custom"] = map[string]int{"priority": 1}
	mb := b.OriginalRequest().GetMetadata()
	mb["ordering.custom"] = map[string]int{"priority": 10}
	assert.True(t, p.Less(a, b))  // 1 < 10
	assert.False(t, p.Less(b, a)) // 10 not < 1
}

func TestCustomPolicy_TypeInt_DefaultUsedWhenKeyMissing(t *testing.T) {
	t.Parallel()
	p, err := newCustomDataOrderingPolicy(&customOrderingParameters{
		Keys: []customOrderingParameter{
			{Key: "priority", Direction: "asc", Type: "int", DefaultVal: "3"},
		},
	})
	assert.NoError(t, err)

	a := mocks.NewMockQueueItemAccessor(1, "a", testFlowKey)
	ma := a.OriginalRequest().GetMetadata()
	ma["ordering.custom"] = map[string]int{"priority": 1}
	noMeta := mocks.NewMockQueueItemAccessor(2, "b", testFlowKey)
	// noMeta has no ordering.custom -> default 3 used
	assert.True(t, p.Less(a, noMeta))  // 1 < 3
	assert.False(t, p.Less(noMeta, a)) // 3 not < 1
}

// --- Type field: invalid ---

func TestCustomPolicy_InvalidTypeReturnsError(t *testing.T) {
	t.Parallel()
	_, err := newCustomDataOrderingPolicy(&customOrderingParameters{
		Keys: []customOrderingParameter{
			{Key: "x", Direction: "asc", Type: "foo", DefaultVal: 2},
		},
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "must be 'float' or 'int'")
}

func TestCustomPolicy_TypeFloat_InvalidStringReturnsError(t *testing.T) {
	t.Parallel()
	_, err := newCustomDataOrderingPolicy(&customOrderingParameters{
		Keys: []customOrderingParameter{
			{Key: "x", Direction: "asc", Type: "float", DefaultVal: "not-a-number"},
		},
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not a valid float")
}
