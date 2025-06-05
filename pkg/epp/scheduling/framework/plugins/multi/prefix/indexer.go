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

// block holds an LRU cache of servers that may have a specific prefix hash.
type block struct {
	Pods *lru.Cache[ServerID, struct{}] // Can be extended with metadata (e.g., timestamp).
}

// An indexer maintains an LRU cache of prompt prefix hashes and the server(s) that might have that
// prefix cached .
type indexer struct {
	mu                sync.RWMutex
	cache             *lru.Cache[BlockHash, *block]
	maxCacheSize      int
	maxServersToMatch int
}

// newIndexer initializes an indexer with size limits and starts cache size reporting.
func newIndexer(maxCacheSize, maxServersToMatch int) *indexer {
	c, err := lru.New[BlockHash, *block](maxCacheSize)
	if err != nil {
		panic(err)
	}
	ix := &indexer{
		cache:             c,
		maxCacheSize:      maxCacheSize,
		maxServersToMatch: maxServersToMatch,
	}
	go ix.ReportCacheSize(time.Second)
	return ix
}

// Add adds a list of prefix hashes to the cache, tied to the server.
func (i *indexer) Add(hashes []BlockHash, pod ServerID) {
	if len(hashes) == 0 || pod.Name == "" {
		return
	}

	i.mu.Lock()
	defer i.mu.Unlock()

	for _, hash := range hashes {
		b, ok := i.cache.Get(hash)
		if !ok {
			// Create block with new LRU
			podLRU, _ := lru.New[ServerID, struct{}](i.maxServersToMatch)
			b = &block{Pods: podLRU}
			i.cache.Add(hash, b)
		}

		b.Pods.Add(pod, struct{}{})
	}
}

// Get returns a set of servers that have the given prefix hash cached.
func (i *indexer) Get(hash BlockHash) map[ServerID]bool {
	i.mu.RLock()
	defer i.mu.RUnlock()

	res := map[ServerID]bool{}
	block, ok := i.cache.Get(hash)
	if !ok {
		return res
	}
	for _, pod := range block.Pods.Keys() {
		res[pod] = true
	}
	return res
}

// ReportCacheSize starts a goroutine that periodically reports the cache size metric.
func (i *indexer) ReportCacheSize(interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for range ticker.C {
		i.mu.RLock()
		size := i.cache.Len()
		metrics.RecordPrefixCacheSize(int64(size))
		log.FromContext(context.TODO()).V(logutil.TRACE).Info("LRU",
			"# entries", size,
			"prefix cache utilization [%]", float64(size)*100/float64(i.maxCacheSize),
		)
		i.mu.RUnlock()
	}
}
