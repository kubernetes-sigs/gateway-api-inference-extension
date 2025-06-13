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
	"fmt"
	"sync"
	"time"

	lru "github.com/hashicorp/golang-lru/v2"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metrics"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// An indexer maintains an LRU cache of prompt prefix hashes and the server(s) that might have that
// prefix cached.
type indexer struct {
	mu         sync.RWMutex
	hashToPods map[BlockHash]podSet                         // the lookup data structure to find pods that have the BlockHash cached
	podToLRU   map[ServerID]*lru.Cache[BlockHash, struct{}] // key is pod namespacedName, value is an LRU cache
	maxLRUSize int
}

// newIndexer initializes an indexer with size limits and starts cache size reporting.
func newIndexer(maxLRUSize int) *indexer {
	ix := &indexer{
		hashToPods: make(map[BlockHash]podSet),
		podToLRU:   make(map[ServerID]*lru.Cache[BlockHash, struct{}]),
		maxLRUSize: maxLRUSize,
	}
	go ix.ReportLRUSize(time.Second)
	return ix
}

// Add adds a list of prefix hashes to the cache, tied to the server.
func (i *indexer) Add(hashes []BlockHash, pod ServerID) error {
	i.mu.Lock()
	// Check if the LRU pod exist
	lruForPod, exists := i.podToLRU[pod]
	if !exists {
		newLRU, err := lru.NewWithEvict[BlockHash, struct{}](i.maxLRUSize, i.makeEvictionFn(pod))
		if err != nil {
			i.mu.Unlock()
			return fmt.Errorf("failed to create LRU for pod %s: %w", pod, err)
		}
		i.podToLRU[pod] = newLRU
		lruForPod = newLRU
	}
	i.mu.Unlock()

	// Add to LRU (may evict)
	for _, hash := range hashes {
		lruForPod.Add(hash, struct{}{})
	}

	// Update hashToPods once under lock
	i.mu.Lock()
	for _, hash := range hashes {
		pods := i.hashToPods[hash]
		if pods == nil {
			pods = make(podSet)
		}
		pods[pod] = struct{}{}
		i.hashToPods[hash] = pods
	}
	i.mu.Unlock()
	return nil
}

// Get returns a set of servers that have the given prefix hash cached.
func (i *indexer) Get(hash BlockHash) podSet {
	i.mu.RLock()
	defer i.mu.RUnlock()

	res := podSet{}
	pods, ok := i.hashToPods[hash]
	if !ok {
		return res
	}

	return pods
}

// makeEvictionFn returns a per-pod LRU eviction callback that removes the pod from hashToPods on eviction.
func (i *indexer) makeEvictionFn(pod ServerID) func(BlockHash, struct{}) {
	return func(hash BlockHash, _ struct{}) {
		i.mu.Lock()
		defer i.mu.Unlock()
		// Remove the pod from the hash→pods map
		if podSet, ok := i.hashToPods[hash]; ok {
			delete(podSet, pod)
			if len(podSet) == 0 {
				delete(i.hashToPods, hash)
			}
		}
	}
}

// ReportLRUSize starts a goroutine that periodically reports the LRU cache size metric.
func (i *indexer) ReportLRUSize(interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for range ticker.C {
		i.mu.RLock()
		totalEntries := 0
		maxPodEntries := 0
		maxPodName := ServerID{}

		for pod, lruCache := range i.podToLRU {
			size := lruCache.Len()
			totalEntries += size
			if size > maxPodEntries {
				maxPodEntries = size
				maxPodName = pod
			}
		}

		numPods := len(i.podToLRU)
		avg := 0.0
		if numPods > 0 {
			avg = float64(totalEntries) / float64(numPods)
		}

		metrics.RecordPrefixCacheSize(int64(totalEntries))
		log.FromContext(context.TODO()).V(logutil.TRACE).Info("Prefix cache state",
			"total entries", totalEntries,
			"# pods", numPods,
			"avg entries per pod", avg,
			"pod with max cache", maxPodName,
			"max pod size", maxPodEntries,
			"global max LRU cache capacity per pod", i.maxLRUSize,
		)

		i.mu.RUnlock()
	}
}
