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

package filtering

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
)

func TestSheddableFilter_Accept(t *testing.T) {
	t.Parallel()
	f := &SheddableFilter{name: "test"}

	tests := []struct {
		name     string
		priority int
		want     bool
	}{
		{name: "negative priority is sheddable", priority: -1, want: true},
		{name: "very negative priority is sheddable", priority: -100, want: true},
		{name: "zero priority is not sheddable", priority: 0, want: false},
		{name: "positive priority is not sheddable", priority: 1, want: false},
		{name: "high priority is not sheddable", priority: 100, want: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			item := &flowcontrol.EvictionItem{Priority: tt.priority}
			assert.Equal(t, tt.want, f.Accept(item))
		})
	}
}

func TestSheddableFilterFactory(t *testing.T) {
	t.Parallel()

	p, err := SheddableFilterFactory("my-filter", nil, nil)
	assert.NoError(t, err)

	f, ok := p.(*SheddableFilter)
	assert.True(t, ok)
	assert.Equal(t, "my-filter", f.TypedName().Name)
	assert.Equal(t, SheddableFilterType, f.TypedName().Type)
}
