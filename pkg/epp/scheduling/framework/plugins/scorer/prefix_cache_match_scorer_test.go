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

package scorer

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	dplugins "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

// mockPod is a mock implementation of types.Pod for testing purposes.
type mockPod struct {
	data map[string]datalayer.Cloneable
}

func newMockPod() *mockPod {
	return &mockPod{
		data: make(map[string]datalayer.Cloneable),
	}
}

func (p *mockPod) Get(key string) (datalayer.Cloneable, bool) {
	val, ok := p.data[key]
	return val, ok
}

func (p *mockPod) Put(key string, value datalayer.Cloneable) {
	p.data[key] = value
}

func (p *mockPod) GetPod() *backend.Pod {
	return nil
}

func (p *mockPod) GetMetrics() *backendmetrics.MetricsState {
	return nil
}

func (p *mockPod) String() string {
	return ""
}

func (p *mockPod) Keys() []string {
	keys := make([]string, 0, len(p.data))
	for k := range p.data {
		keys = append(keys, k)
	}
	return keys
}

func TestPrefixCacheScorer_Score(t *testing.T) {
	pod1 := newMockPod()
	pod1.Put(dplugins.PrefixCacheMatchInfoKey, dplugins.NewPrefixCacheMatchInfo(50.0))

	pod2 := newMockPod()
	pod2.Put(dplugins.PrefixCacheMatchInfoKey, dplugins.NewPrefixCacheMatchInfo(100.0))

	pod3 := newMockPod()

	testCases := []struct {
		name     string
		pods     []types.Pod
		expected map[types.Pod]float64
	}{
		{
			name: "pods with prefix cache match percent",
			pods: []types.Pod{pod1, pod2},
			expected: map[types.Pod]float64{
				pod1: 50.0,
				pod2: 100.0,
			},
		},
		{
			name: "pod without prefix cache match percent",
			pods: []types.Pod{pod3},
			expected: map[types.Pod]float64{
				pod3: 0.0,
			},
		},
		{
			name: "mixed pods",
			pods: []types.Pod{pod1, pod3},
			expected: map[types.Pod]float64{
				pod1: 50.0,
				pod3: 0.0,
			},
		},
		{
			name:     "empty pods list",
			pods:     []types.Pod{},
			expected: map[types.Pod]float64{},
		},
	}

	scorer := NewPrefixCacheScorer()

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			scores := scorer.Score(context.Background(), nil, nil, tc.pods)
			assert.Equal(t, tc.expected, scores)
		})
	}
}

func TestNewPrefixCacheScorer(t *testing.T) {
	scorer := NewPrefixCacheScorer()
	assert.NotNil(t, scorer)
	assert.Equal(t, PrefixCacheMatchScorerType, scorer.tn.Type)
	assert.Equal(t, PrefixCacheMatchScorerType, scorer.tn.Name)
}

func TestPrefixCacheScorer_WithName(t *testing.T) {
	scorer := NewPrefixCacheScorer()
	customName := "custom-scorer"
	scorer.WithName(customName)
	assert.Equal(t, customName, scorer.TypedName().Name)
}

func TestPrefixCacheScorer_TypedName(t *testing.T) {
	scorer := NewPrefixCacheScorer()
	tn := scorer.TypedName()
	assert.Equal(t, PrefixCacheMatchScorerType, tn.Type)
	assert.Equal(t, PrefixCacheMatchScorerType, tn.Name)
}

func TestPrefixCacheScorer_Consumes(t *testing.T) {
	scorer := NewPrefixCacheScorer()
	consumes := scorer.Consumes()
	assert.Empty(t, consumes)
}
