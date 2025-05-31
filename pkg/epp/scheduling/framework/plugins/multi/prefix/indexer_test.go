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
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIndexer_AddAndGet(t *testing.T) {
	cache := newIndexer(2)

	hash1 := BlockHash(1)
	server := ServerID{Namespace: "default", Name: "server1"}

	// Add an entry to the cache
	cache.Add([]BlockHash{hash1}, server)

	// Retrieve the entry
	assert.Equal(t, 1, cache.ll.Len(), "Cache size should be 1 after adding an entry")
	servers := cache.Get(hash1)
	assert.Contains(t, servers, server, "Cache should contain the added server")

	// Add another entry to the cache, the cache size should be incremented to 2.
	cache.Add([]BlockHash{BlockHash(2)}, server)
	assert.Equal(t, 2, cache.ll.Len(), "Cache size should  be 2 after adding an entry")

	// Add another entry to the cache, which should evict the first one due to max size.
	cache.Add([]BlockHash{BlockHash(3)}, server)
	assert.Equal(t, 2, cache.ll.Len(), "Cache size should still be 2 after adding an entry")
}
