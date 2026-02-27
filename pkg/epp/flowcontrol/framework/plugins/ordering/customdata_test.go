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

	assert.Nil(t, err, "error should be nil")
	assert.Equal(t, "my-test-policy", p.TypedName().Name)
	assert.Equal(t, CustomDataOrderingPolicyType, p.TypedName().Type)
}

func TestCustomPolicy_Name(t *testing.T) {
	t.Parallel()
	p := newCustomDataOrderingPolicy(nil)
	assert.Equal(t, CustomDataOrderingPolicyType, p.Name())
}

func TestCustomPolicy_RequiredQueueCapabilities(t *testing.T) {
	t.Parallel()
	p := newCustomDataOrderingPolicy(nil)
	c := p.RequiredQueueCapabilities()
	assert.Len(t, c, 1)
	assert.Equal(t, flowcontrol.CapabilityPriorityConfigurable, c[0])
}

func TestCustomPolicy_Less(t *testing.T) {
	t.Parallel()
	p := newCustomDataOrderingPolicy(&customOrderingParameters{
		Keys: []customOrderingParameter{
			{
				Key:        "score1",
				Direction:  "asc",
				DefaultVal: 2,
			},
		},
	})

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
	p := newCustomDataOrderingPolicy(&customOrderingParameters{
		Keys: []customOrderingParameter{
			{
				Key:        "score1",
				Direction:  "asc",
				DefaultVal: 2,
			},
		},
	})

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
	p := newCustomDataOrderingPolicy(&customOrderingParameters{
		Keys: []customOrderingParameter{
			{
				Key:        "score1",
				Direction:  "desc",
				DefaultVal: 2,
			},
		},
	})

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
	p := newCustomDataOrderingPolicy(&customOrderingParameters{
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
	p := newCustomDataOrderingPolicy(&customOrderingParameters{
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
