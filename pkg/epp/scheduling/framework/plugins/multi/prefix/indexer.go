/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package prefix

import (
	"context"
	"sync"
	"time"
	"unsafe"

	"container/list"

	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

func newIndexer(maxCacheSize int) *indexer {
	t := &indexer{
		maxCacheSize: maxCacheSize,
		table:        make(map[BlockHash]map[ServerID]*list.Element),
		ll:           list.New(),
	}
	go t.ReportCacheSize(time.Second)
	return t
}

// An indexer maintains an LRU cache of prompt prefix hashes and the server(s) that might have that
// prefix cached .
type indexer struct {
	mu           sync.RWMutex
	maxCacheSize int
	table        map[BlockHash]map[ServerID]*list.Element // from any prefix cache to the cache entry to find the server
	ll           *list.List                               // LinkedList to keep track of the order of entries
}

// value is the value stored in the linked list.
type value struct {
	server ServerID
	hash   BlockHash
}

// Get returns the set of servers that have the given prefix hash cached.
func (i *indexer) Get(hash BlockHash) map[ServerID]bool {
	i.mu.RLock()
	defer i.mu.RUnlock()
	res := map[ServerID]bool{}
	for server := range i.table[hash] {
		res[server] = true
	}
	return res
}

// Add adds a list of prefix hashes of a single request to the server the request was sent to.
// The intuition is that this server is likely to have the prefix cached, so next time a request
// sharing the longest prefix should be sent to the same server to take advantage of the cache hit.
func (i *indexer) Add(hashes []BlockHash, server ServerID) {
	i.mu.Lock()
	defer i.mu.Unlock()
	for _, hash := range hashes {
		i.add(hash, server)
	}
}

func (i *indexer) check(hash BlockHash, server ServerID) (*list.Element, bool) {
	servers, ok := i.table[hash]
	if !ok {
		return nil, false
	}
	e, ok := servers[server]
	return e, ok
}

func (i *indexer) add(hash BlockHash, server ServerID) {
	e, exists := i.check(hash, server)
	if exists {
		i.ll.MoveToBack(e)
	} else {
		i.create(hash, server)
	}
}

func (i *indexer) create(hash BlockHash, server ServerID) {
	for i.ll.Len() >= i.maxCacheSize {
		// Evict the least recently used entry if we've exceeded the max cache size
		i.evict()
	}

	if _, ok := i.table[hash]; !ok {
		i.table[hash] = make(map[ServerID]*list.Element)
	}
	v := &value{
		server: server,
		hash:   hash,
	}
	e := i.ll.PushBack(v)
	i.table[hash][server] = e
}

// evict removes the least recently used entry from the cache
func (i *indexer) evict() {
	oldestNode := i.ll.Front()
	if oldestNode == nil {
		return
	}
	i.ll.Remove(oldestNode)

	v := oldestNode.Value.(*value)
	hash := v.hash
	server := v.server
	// Remove from the hash map
	serverMap := i.table[hash]
	delete(serverMap, server)

	// If this was the last server for this hash, remove the hash entry entirely
	if len(serverMap) == 0 {
		delete(i.table, hash)
	}

	log.FromContext(context.TODO()).V(logutil.TRACE).Info("Evicted LRU entry", "hash", hash, "server", server)
}

// ReportCacheSize starts a goroutine that periodically reports the cache size metric
func (i *indexer) ReportCacheSize(interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for range ticker.C {
		i.mu.RLock()
		metrics.RecordPrefixCacheSize(int64(i.ll.Len()))
		log.FromContext(context.TODO()).V(logutil.TRACE).Info("LRU", "# entries", i.ll.Len(), "estimated size MB", i.ll.Len()*i.estimateEntrySize()/1000000)
		i.mu.RUnlock()
	}
}

// estimateEntrySize estimates the memory size of a cache entry in bytes.
func (i *indexer) estimateEntrySize() int {
	size := 0

	// Estimate the size of a node in the linked list.
	// First get the size of the node struct via unsafe.Sizeof.
	// The prev and next pointers are 8 bytes each on a 64-bit system.
	// The BlockHash is a uint64, which is 8 bytes.
	// The ServerID is a NamespacedName, which contains two strings (Name and Namespace).
	// The headers for the strings are 16 bytes each (8 bytes for the pointer and 8 bytes for the length).
	// So unsafe.Sizeof(node{}) should return 2*8 + 8 + 2*16 = 48 bytes.
	size += int(unsafe.Sizeof(value{}))
	// Size of the Name and Namespace strings in ServerID, assuming 63 bytes each (max length for Kubernetes NamespacedName).
	size += 2 * 63

	// Estimate the size of an entry in the hash map. Note the overhead of the map headers and buckets are ignored.
	size += 8      // Size of the BlockHash (uint64).
	size += 2 * 16 // Size of the ServerID string headers (NamespacedName).
	size += 2 * 63 // Size of the Name and Namespace strings in ServerID.
	size += 8      // Size of the pointer to the node in the hash map.

	// Based on the above estimates, the estimated size of an entry is:
	// (48 + 2*63) + (8 + 2*16 + 2*63 + 8) = 348 bytes.
	return size
}
