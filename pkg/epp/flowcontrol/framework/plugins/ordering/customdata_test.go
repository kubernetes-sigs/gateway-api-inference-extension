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

	a := mocks.NewMockQueueItemAccessor(1, "a", testFlowKey)
	b := mocks.NewMockQueueItemAccessor(2, "b", testFlowKey)
	c := mocks.NewMockQueueItemAccessor(3, "c", testFlowKey)

	ma := a.OriginalRequest().GetMetadata()
	ma["ordering.custom.score1"] = 3.2

	mb := b.OriginalRequest().GetMetadata()
	mb["ordering.custom.score1"] = 5.9

	mc := c.OriginalRequest().GetMetadata()
	mc["ordering.custom.score1"] = 3.2

	p := newCustomDataOrderingPolicy(&customOrderingParameters{
		Keys: []customOrderingParameter{
			{
				Key:        "score1",
				Direction:  "asc",
				DefaultVal: 2,
			},
		},
	})

	assert.True(t, p.Less(a, b))
	assert.False(t, p.Less(b, a))
	assert.False(t, p.Less(a, c))

	nilA := mocks.NewMockQueueItemAccessor(1, "nil-a", testFlowKey)

	mnilA := nilA.OriginalRequest().GetMetadata()
	mnilA["ordering.custom.score1"] = nil

	assert.True(t, p.Less(nilA, a))
	assert.False(t, p.Less(a, nilA))
	assert.False(t, p.Less(nilA, nilA))

	p = newCustomDataOrderingPolicy(&customOrderingParameters{
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

	ma["ordering.custom.score2"] = 4.9
	mb["ordering.custom.score2"] = 2.9
	mc["ordering.custom.score2"] = 6.3

	assert.True(t, p.Less(a, b))
	assert.False(t, p.Less(b, a))
	assert.True(t, p.Less(a, c))

	d := mocks.NewMockQueueItemAccessor(4, "d", testFlowKey)
	md := d.OriginalRequest().GetMetadata()
	md["ordering.custom.score1"] = 3.2
	md["ordering.custom.score2"] = 4.9

	assert.False(t, p.Less(a, d))

	p = newCustomDataOrderingPolicy(&customOrderingParameters{
		Keys: []customOrderingParameter{
			{
				Key:        "score1",
				Direction:  "desc",
				DefaultVal: 2,
			},
		},
	})

	assert.False(t, p.Less(a, b))
	assert.True(t, p.Less(b, a))
	assert.False(t, p.Less(a, c))
}

func TestCustomPolicy_Less_Desc(t *testing.T) {
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
	ma["ordering.custom.score1"] = 3.2
	ma["ordering.custom.score2"] = 4.9

	mb := b.OriginalRequest().GetMetadata()
	mb["ordering.custom.score1"] = 5.9
	mb["ordering.custom.score2"] = 2.9

	mc := c.OriginalRequest().GetMetadata()
	mc["ordering.custom.score1"] = 3.2
	mc["ordering.custom.score2"] = 6.3

	md := d.OriginalRequest().GetMetadata()
	md["ordering.custom.score1"] = 3.2
	md["ordering.custom.score2"] = 4.9

	assert.True(t, p.Less(a, b))
	assert.False(t, p.Less(b, a))
	assert.True(t, p.Less(c, a))
	assert.False(t, p.Less(a, d))
}
