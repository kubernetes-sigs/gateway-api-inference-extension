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

package preparedata

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrprefix "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	prefix "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/scheduling/scorer/prefix"
)

// mockIndexer is a mock implementation of the prefix.Indexer for testing.
type mockIndexer struct {
	cache map[prefix.BlockHash]prefix.PodSet
}

// Get returns the servers that have the given block hash.
func (m *mockIndexer) Get(hash prefix.BlockHash) prefix.PodSet {
	if servers, ok := m.cache[hash]; ok {
		return servers
	}
	return nil
}

// Put adds a block hash to the cache for a given server.
func (m *mockIndexer) Put(server prefix.ServerID, hash prefix.BlockHash) {
	if m.cache == nil {
		m.cache = make(map[prefix.BlockHash]prefix.PodSet)
	}
	if _, ok := m.cache[hash]; !ok {
		m.cache[hash] = make(prefix.PodSet)
	}
	m.cache[hash][server] = struct{}{}
}

func (m *mockIndexer) Add(hashes []prefix.BlockHash, server prefix.Server) {
	for _, hash := range hashes {
		m.Put(server.ServerID, hash)
	}
}

func (m *mockIndexer) RemovePod(server prefix.ServerID) {}
func (m *mockIndexer) Pods() []prefix.ServerID {
	return nil
}

// mockEndpoint is a mock implementation of the scheduling.Endpoint for testing.
type mockEndpoint struct {
	data    map[string]datalayer.Cloneable
	meta    *datalayer.EndpointMetadata
	metrics *datalayer.Metrics
}

func newMockEndpoint(name string) *mockEndpoint {
	return &mockEndpoint{
		data: make(map[string]datalayer.Cloneable),
		meta: &datalayer.EndpointMetadata{
			NamespacedName: types.NamespacedName{Name: name, Namespace: "default"},
		},
		metrics: &datalayer.Metrics{},
	}
}

func (m *mockEndpoint) Get(key string) (datalayer.Cloneable, bool) {
	val, ok := m.data[key]
	return val, ok
}

func (m *mockEndpoint) Put(key string, val datalayer.Cloneable) {
	m.data[key] = val
}

func (m *mockEndpoint) GetMetadata() *datalayer.EndpointMetadata {
	return m.meta
}

func (m *mockEndpoint) GetMetrics() *datalayer.Metrics {
	return m.metrics
}

func (m *mockEndpoint) String() string {
	return m.meta.NamespacedName.String()
}

func (m *mockEndpoint) Keys() []string {
	keys := make([]string, 0, len(m.data))
	for k := range m.data {
		keys = append(keys, k)
	}
	return keys
}

func TestNew(t *testing.T) {
	p := New()
	assert.NotNil(t, p)
	assert.IsType(t, &ApproxPrefixCache{}, p)
}

func TestTypedName(t *testing.T) {
	p := New()
	expected := plugin.TypedName{
		Type: PrepareDataPluginType,
		Name: ApproxPrefixCachePlugin,
	}
	assert.Equal(t, expected, p.TypedName())
}

func TestMatchLongestPrefix(t *testing.T) {
	server1 := prefix.ServerID(types.NamespacedName{Name: "server1", Namespace: "default"})
	server2 := prefix.ServerID(types.NamespacedName{Name: "server2", Namespace: "default"})

	hash1 := prefix.BlockHash(1)
	hash2 := prefix.BlockHash(2)
	hash3 := prefix.BlockHash(3)

	testCases := []struct {
		name         string
		hashes       []prefix.BlockHash
		indexerCache map[prefix.BlockHash]prefix.PodSet
		expected     map[prefix.ServerID]int
	}{
		{
			name:         "no match",
			hashes:       []prefix.BlockHash{hash1, hash2, hash3},
			indexerCache: map[prefix.BlockHash]prefix.PodSet{},
			expected:     map[prefix.ServerID]int{},
		},
		{
			name:   "partial match",
			hashes: []prefix.BlockHash{hash1, hash2, hash3},
			indexerCache: map[prefix.BlockHash]prefix.PodSet{
				hash1: {server1: {}},
			},
			expected: map[prefix.ServerID]int{server1: 1},
		},
		{
			name:   "full match",
			hashes: []prefix.BlockHash{hash1, hash2, hash3},
			indexerCache: map[prefix.BlockHash]prefix.PodSet{
				hash1: {server1: {}},
				hash2: {server1: {}},
				hash3: {server1: {}},
			},
			expected: map[prefix.ServerID]int{server1: 3},
		},
		{
			name:   "multiple servers",
			hashes: []prefix.BlockHash{hash1, hash2, hash3},
			indexerCache: map[prefix.BlockHash]prefix.PodSet{
				hash1: {server1: {}, server2: {}},
				hash2: {server1: {}},
			},
			expected: map[prefix.ServerID]int{server1: 2, server2: 1},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			p := &ApproxPrefixCache{
				indexer: &mockIndexer{cache: tc.indexerCache},
			}
			result := p.matchLongestPrefix(context.Background(), tc.hashes)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestPrepareRequestData(t *testing.T) {
	server1 := prefix.ServerID(types.NamespacedName{Name: "server1", Namespace: "default"})

	request := &scheduling.LLMRequest{
		Body: &scheduling.LLMRequestBody{
			Completions: &scheduling.CompletionsRequest{
				Prompt: "This is a prompt",
			},
		},
	}

	hashes := prefix.HashPrompt(context.Background(), request, 4, 10)
	hash1 := hashes[0]

	p := &ApproxPrefixCache{
		config: prefix.Config{
			BlockSizeTokens:        4,
			MaxPrefixBlocksToMatch: 10,
		},
		indexer: &mockIndexer{
			cache: map[prefix.BlockHash]prefix.PodSet{
				hash1: {server1: {}},
			},
		},
	}

	pods := []scheduling.Endpoint{
		newMockEndpoint("server1"),
		newMockEndpoint("server2"),
	}

	err := p.PrepareRequestData(context.Background(), request, pods)
	assert.NoError(t, err)

	for _, pod := range pods {
		info, ok := pod.Get(attrprefix.PrefixCacheMatchInfoKey)
		assert.True(t, ok)
		prefixInfo, ok := info.(*attrprefix.PrefixCacheMatchInfo)
		assert.True(t, ok)

		podName := pod.GetMetadata().NamespacedName.Name
		if podName == "server1" {
			assert.Equal(t, 1, prefixInfo.MatchBlocks())
			assert.Equal(t, len(hashes), prefixInfo.TotalBlocks())
			assert.Equal(t, 4, prefixInfo.BlockSizeTokens())
		} else if podName == "server2" {
			assert.Equal(t, 0, prefixInfo.MatchBlocks())
			assert.Equal(t, len(hashes), prefixInfo.TotalBlocks())
			assert.Equal(t, 4, prefixInfo.BlockSizeTokens())
		} else {
			t.Errorf("unexpected pod name: %s", podName)
		}
		fmt.Printf("pod: %s, info: %+v\n", pod.GetMetadata().NamespacedName.Name, prefixInfo)
	}
}
