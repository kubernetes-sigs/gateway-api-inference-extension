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
package prefix

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIndexer_AddAndGet(t *testing.T) {
	server := Server{
		ServerID:       ServerID{Namespace: "default", Name: "server1"},
		numOfGPUBlocks: 2,
	}
	i := newIndexer(context.Background(), 3) // Initialize with an lruSize greater than server.numOfGPUBlocks to verify server-defined limits take precedence.

	hash1 := BlockHash(1)
	// Add an entry to the cache
	i.Add([]BlockHash{hash1}, server)

	// Retrieve the entry
	assert.Equal(t, 1, i.podToLRU[server.ServerID].Len(), "Cache size should be 1 after adding an entry")
	servers := i.Get(hash1)
	assert.Contains(t, servers, server.ServerID, "Cache should contain the added server")

	// Add another entry to the cache, the cache size should be incremented to 2.
	i.Add([]BlockHash{BlockHash(2)}, server)
	assert.Equal(t, 2, i.podToLRU[server.ServerID].Len(), "Cache size should  be 2 after adding an entry")

	// Add another entry to the cache, which should evict the first one due to max size.
	i.Add([]BlockHash{BlockHash(3)}, server)
	assert.Equal(t, 2, i.podToLRU[server.ServerID].Len(), "Cache size should still be 2 after adding an entry")

	servers = i.Get(BlockHash(4))
	assert.Empty(t, servers, "Cache should not contain non-existent hash")
}

func TestIndexer_RemovePodAndEviction(t *testing.T) {
	const indexerSize = 10

	i := newIndexer(context.Background(), indexerSize)

	server1 := Server{ServerID: ServerID{Namespace: "default", Name: "server1"}}
	server2 := Server{ServerID: ServerID{Namespace: "default", Name: "server2"}}

	// Add indexerSize hashes to both servers
	var hashes []BlockHash
	for j := 0; j < indexerSize; j++ {
		h := BlockHash(j)
		hashes = append(hashes, h)
		i.Add([]BlockHash{h}, server1)
		i.Add([]BlockHash{h}, server2)
	}

	// Ensure all entries are added
	assert.Equal(t, indexerSize, i.podToLRU[server1.ServerID].Len(), "server1 should have 10 entries")
	assert.Equal(t, indexerSize, i.podToLRU[server2.ServerID].Len(), "server2 should have 10 entries")

	// Ensure each hash in hashToPods maps to both server1 and server2
	for _, h := range hashes {
		pods := i.hashToPods[h]
		assert.Len(t, pods, 2, "Each hash should be associated with exactly 2 pods")
		assert.Contains(t, pods, server1.ServerID, "hash should be associated with server1")
		assert.Contains(t, pods, server2.ServerID, "hash should be associated with server2")
	}

	// Add indexerSize hash to server1 → should evict BlockHash(0)
	evictedHash := BlockHash(0)
	newHash := BlockHash(indexerSize)
	i.Add([]BlockHash{newHash}, server1)

	// server1 LRU should still be at max capacity
	assert.Equal(t, indexerSize, i.podToLRU[server1.ServerID].Len(), "server1 LRU should maintain max size")

	// BlockHash(0) should no longer have server1 in hashToPods
	pods := i.Get(evictedHash)
	assert.NotContains(t, pods, server1.ServerID, "server1 should be evicted from hashToPods for hash 0")
	assert.Contains(t, pods, server2.ServerID, "server2 should still have hash 0")

	// Remove server2
	i.RemovePod(server2.ServerID)

	// hashToPods for hash 0 should now be empty
	pods = i.Get(evictedHash)
	assert.NotContains(t, pods, server2.ServerID, "server2 should be removed from hash 0")
	assert.Empty(t, pods, "hash 0 should have no pods after both eviction and removal")

	// All remaining hashes should map only to server1
	for hash, pods := range i.hashToPods {
		assert.Len(t, pods, 1, "hash %v should have only 1 pod after server2 removal", hash)
		assert.Contains(t, pods, server1.ServerID, "hash %v should only contain server1", hash)
	}

	// Ensure hashToPods contains exactly indexerSize hashes (post-eviction and server2 removal)
	assert.Len(t, i.hashToPods, indexerSize, "hashToPods should contain %d hashes after cleanup", indexerSize)
}
