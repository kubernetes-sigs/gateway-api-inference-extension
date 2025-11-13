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
	mu             sync.RWMutex
	hashToPods     map[BlockHash]podSet                         // the lookup data structure to find pods that have the BlockHash cached
	podToLRU       map[ServerID]*lru.Cache[BlockHash, struct{}] // key is pod namespacedName, value is an LRU cache
	defaultLRUSize int
}

// newIndexer initializes an indexer with size limits and starts cache size reporting.
func newIndexer(ctx context.Context, defaultLRUSize int) *indexer {
	indexer := &indexer{
		hashToPods:     make(map[BlockHash]podSet),
		podToLRU:       make(map[ServerID]*lru.Cache[BlockHash, struct{}]),
		defaultLRUSize: defaultLRUSize,
	}

	go indexer.reportLRUSize(ctx, time.Second)
	return indexer
}

// Add adds a list of prefix hashes to the cache, tied to the server.
func (i *indexer) Add(hashes []BlockHash, pod Server) {
	i.mu.Lock()
	// Check if the LRU pod exist
	lruForPod, exists := i.podToLRU[pod.ServerID]
	if !exists {
		lruSize := pod.numOfGPUBlocks
		if lruSize <= 0 {
			lruSize = i.defaultLRUSize
		}
		newLRU, _ := lru.NewWithEvict(lruSize, i.makeEvictionFn(pod.ServerID))
		i.podToLRU[pod.ServerID] = newLRU
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
		podIDs := i.hashToPods[hash]
		if podIDs == nil {
			podIDs = make(podSet)
		}
		podIDs[pod.ServerID] = struct{}{}
		i.hashToPods[hash] = podIDs
	}

	i.mu.Unlock()
}

// Get returns a set of servers that have the given prefix hash cached.
func (i *indexer) Get(hash BlockHash) podSet {
	i.mu.RLock()
	defer i.mu.RUnlock()

	pods := i.hashToPods[hash]
	res := make(podSet, len(pods))
	for pod := range pods {
		// Deep copy to avoid race condition.
		res[pod] = struct{}{}
	}

	return res
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

// reportLRUSize starts a goroutine that periodically reports the LRU cache size metric.
func (i *indexer) reportLRUSize(ctx context.Context, interval time.Duration) {
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
		log.FromContext(ctx).V(logutil.TRACE).Info("Prefix cache state",
			"total entries", totalEntries,
			"# pods", numPods,
			"avg entries per pod", avg,
			"pod with max cache", maxPodName,
			"max pod size", maxPodEntries,
			"global max LRU cache capacity per pod", i.defaultLRUSize,
		)

		i.mu.RUnlock()
	}
}

// RemovePod removes a pod and its associated entries from the indexer.
func (i *indexer) RemovePod(pod ServerID) {
	i.mu.RLock()
	lruCache, exists := i.podToLRU[pod]
	i.mu.RUnlock()

	if !exists {
		return
	}

	// Remove all hashes associated with the pod from hashToPods (triggers eviction callbacks).
	for _, hash := range lruCache.Keys() {
		lruCache.Remove(hash)
	}

	i.mu.Lock()
	delete(i.podToLRU, pod)
	i.mu.Unlock()
}

// Pods returns the list of all pods currently tracked in the indexer.
func (i *indexer) Pods() []ServerID {
	i.mu.RLock()
	defer i.mu.RUnlock()

	pods := make([]ServerID, 0, len(i.podToLRU))
	for pod := range i.podToLRU {
		pods = append(pods, pod)
	}
	return pods
}
