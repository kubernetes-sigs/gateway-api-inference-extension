/*
Copyright 2026 The Kubernetes Authors.

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

package ordering

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
)

func TestPriorityThenTimeOrdering_Less(t *testing.T) {
	t.Parallel()
	policy := &PriorityThenTimeOrdering{name: "test"}

	now := time.Now()

	itemLowPriOld := &flowcontrol.EvictionItem{Priority: -2, DispatchTime: now.Add(-time.Second)}
	itemLowPriNew := &flowcontrol.EvictionItem{Priority: -2, DispatchTime: now}
	itemMidPri := &flowcontrol.EvictionItem{Priority: 0, DispatchTime: now}
	itemHighPri := &flowcontrol.EvictionItem{Priority: 5, DispatchTime: now}

	testCases := []struct {
		name     string
		a, b     *flowcontrol.EvictionItem
		expected bool
	}{
		{"lower priority evicted first", itemLowPriOld, itemMidPri, true},
		{"higher priority not evicted first", itemHighPri, itemMidPri, false},
		{"same priority older evicted first", itemLowPriOld, itemLowPriNew, true},
		{"same priority newer not evicted first", itemLowPriNew, itemLowPriOld, false},
		{"same item no preference", itemMidPri, itemMidPri, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tc.expected, policy.Less(tc.a, tc.b))
		})
	}
}

func TestPriorityThenTimeOrderingFactory(t *testing.T) {
	t.Parallel()

	p, err := PriorityThenTimeOrderingFactory("my-ordering", nil, nil)
	assert.NoError(t, err)

	o, ok := p.(*PriorityThenTimeOrdering)
	assert.True(t, ok)
	assert.Equal(t, "my-ordering", o.TypedName().Name)
	assert.Equal(t, PriorityThenTimeOrderingType, o.TypedName().Type)
}
