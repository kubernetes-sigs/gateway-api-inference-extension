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

package requestcontrol

import (
	"context"
	"sort"
	"strings"
	"sync"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

const (
	// defaultCacheTTL is the duration for which a pod lookup result is considered valid.
	// This trades off "Scale-from-Zero" responsiveness (latency to see new pods) against Datastore lock contention.
	// 50ms aligns roughly with standard Prometheus scrape intervals or high-frequency control loops.
	defaultCacheTTL = 50 * time.Millisecond

	// cleanupInterval dictates how often we sweep the map for expired entries.
	cleanupInterval = 1 * time.Minute

	// defaultCacheKey is used when no subset filter is present (Return All Pods).
	defaultCacheKey = "__default__"

	// emptySubsetCacheKey is used when a subset filter is present but empty (Return No Pods).
	emptySubsetCacheKey = "__explicit_empty__"
)

// --- DatastorePodLocator (The Delegate) ---

// DatastorePodLocator implements contracts.PodLocator by querying the EPP Datastore.
// It centralizes the logic for resolving candidate pods based on request metadata (specifically Envoy subset filters).
type DatastorePodLocator struct {
	datastore Datastore
}

var _ contracts.PodLocator = &DatastorePodLocator{}

// NewDatastorePodLocator creates a new DatastorePodLocator.
func NewDatastorePodLocator(ds Datastore) *DatastorePodLocator {
	return &DatastorePodLocator{
		datastore: ds,
	}
}

// Locate retrieves the list of candidate pods from the datastore that match the criteria defined in the request
// metadata.
//
// It supports:
// 1. Returning all pods if no specific subset filter is present.
// 2. Returning a filtered list of pods if "x-gateway-destination-endpoint-subset" is present.
func (d *DatastorePodLocator) Locate(ctx context.Context, requestMetadata map[string]any) []backendmetrics.PodMetrics {
	loggerTrace := log.FromContext(ctx).V(logutil.TRACE)

	// Check if the subset filter namespace exists in metadata.
	// If not, we assume the request targets the default pool (all pods).
	if requestMetadata == nil {
		return d.datastore.PodList(datastore.AllPodsPredicate)
	}

	subsetMap, found := requestMetadata[metadata.SubsetFilterNamespace].(map[string]any)
	if !found {
		return d.datastore.PodList(datastore.AllPodsPredicate)
	}

	// Check if the specific endpoint key exists within the subset map.
	endpointSubsetList, found := subsetMap[metadata.SubsetFilterKey].([]any)
	if !found {
		return d.datastore.PodList(datastore.AllPodsPredicate)
	}

	// If the filter key exists but the list is empty, it implies a filter that matched nothing upstream (or malformed
	// data), so we return nothing.
	if len(endpointSubsetList) == 0 {
		loggerTrace.Info("found empty subset filter in request metadata, returning empty pod list")
		return []backendmetrics.PodMetrics{}
	}

	// Build a lookup map for efficient filtering.
	// The subset list contains strings in the format "<address>:<port>" (e.g., "10.0.1.0:8080").
	// We only care about the IP address for matching against PodMetrics.
	endpoints := make(map[string]bool, len(endpointSubsetList))
	for _, endpoint := range endpointSubsetList {
		epStr, ok := endpoint.(string)
		if !ok {
			loggerTrace.Info("ignoring non-string endpoint in subset list", "value", endpoint)
			continue
		}
		// Extract address from endpoint string.
		if idx := strings.LastIndexByte(epStr, ':'); idx >= 0 {
			endpoints[epStr[:idx]] = true
		} else {
			endpoints[epStr] = true
		}
	}

	// Query the Datastore with a predicate.
	podTotalCount := 0
	podFilteredList := d.datastore.PodList(func(pm backendmetrics.PodMetrics) bool {
		podTotalCount++
		// If the pod's IP is in our allowed map, include it.
		// Note: We use GetIPAddress() which should align with the subset address.
		if pod := pm.GetMetadata(); pod != nil {
			if _, found := endpoints[pod.GetIPAddress()]; found {
				return true
			}
		}
		return false
	})

	loggerTrace.Info("filtered candidate pods by subset filtering",
		"podTotalCount", podTotalCount,
		"filteredCount", len(podFilteredList))

	return podFilteredList
}

// --- CachedPodLocator (The Decorator) ---

// cacheEntry represents a snapshot of pod metrics at a specific point in time.
type cacheEntry struct {
	pods   []backendmetrics.PodMetrics
	expiry time.Time
}

// CachedPodLocator is a decorator for contracts.PodLocator that caches resultscto reduce lock contention on the
// underlying Datastore.
//
// It is designed for high-throughput paths (like the Flow Control dispatch loop)cwhere fetching fresh data every
// millisecond is unnecessary and expensive.
type CachedPodLocator struct {
	// delegate is the underlying source of truth (usually the DatastorePodLocator).
	delegate contracts.PodLocator

	// ttl defines how long a cache entry remains valid.
	ttl time.Duration

	// mu protects the cache map.
	mu    sync.RWMutex
	cache map[string]cacheEntry
}

var _ contracts.PodLocator = &CachedPodLocator{}

// NewCachedPodLocator creates a new CachedPodLocator and starts a background cleanup routine.
// The provided context is used to control the lifecycle of the cleanup goroutine.
func NewCachedPodLocator(ctx context.Context, delegate contracts.PodLocator, ttl time.Duration) *CachedPodLocator {
	if ttl <= 0 {
		ttl = defaultCacheTTL
	}

	c := &CachedPodLocator{
		delegate: delegate,
		ttl:      ttl,
		cache:    make(map[string]cacheEntry),
	}

	// Start background cleanup to prevent memory leaks from unused keys.
	go c.runCleanup(ctx)

	return c
}

// Locate returns the list of candidate pods for the given request metadata, using a cached result if available and
// fresh.
func (c *CachedPodLocator) Locate(ctx context.Context, requestMetadata map[string]any) []backendmetrics.PodMetrics {
	key := c.generateCacheKey(requestMetadata)

	// Fast Path: Read Lock
	c.mu.RLock()
	entry, found := c.cache[key]
	c.mu.RUnlock()

	if found && time.Now().Before(entry.expiry) {
		return entry.pods
	}

	// Slow Path: Write Lock with Double-Check
	// We missed the cache. Acquire write lock to update it.
	c.mu.Lock()
	defer c.mu.Unlock()

	// Double-check: Someone else might have updated the cache while we were waiting for the lock.
	entry, found = c.cache[key]
	if found && time.Now().Before(entry.expiry) {
		return entry.pods
	}

	// Fetch from Delegate.
	// Note: We hold the lock during the fetch. This serializes requests for the same key, preventing a "thundering herd"
	// on the underlying Datastore.
	// Since Datastore lookups are fast in-memory scans, this lock duration is acceptable.
	freshPods := c.delegate.Locate(ctx, requestMetadata)

	// Update cache.
	c.cache[key] = cacheEntry{
		pods:   freshPods,
		expiry: time.Now().Add(c.ttl),
	}

	return freshPods
}

// generateCacheKey creates a deterministic string key representing the pod selection criteria.
// It handles the "x-gateway-destination-endpoint-subset" structure specifically.
func (c *CachedPodLocator) generateCacheKey(reqMetadata map[string]any) string {
	// No Metadata -> All Pods
	if reqMetadata == nil {
		return defaultCacheKey
	}

	subsetMap, found := reqMetadata[metadata.SubsetFilterNamespace].(map[string]any)
	if !found {
		return defaultCacheKey
	}

	// The subset filter key contains a list of endpoint strings (e.g., "10.0.0.1:8080").
	// We must treat this list as a set (order independent).
	endpointSubsetList, found := subsetMap[metadata.SubsetFilterKey].([]any)

	// Namespace exists, but "subset" key is missing -> All Pods
	if !found {
		return defaultCacheKey
	}

	// "subset" key exists, but is empty list -> No Pods
	if len(endpointSubsetList) == 0 {
		return emptySubsetCacheKey
	}

	// Optimization: If there's only one endpoint, return it directly to avoid allocation.
	if len(endpointSubsetList) == 1 {
		if s, ok := endpointSubsetList[0].(string); ok {
			return s
		}
		return defaultCacheKey // Fallback for malformed data.
	}

	// Copy and sort to ensure determinism ( [A, B] must equal [B, A] ).
	endpoints := make([]string, 0, len(endpointSubsetList))
	for _, ep := range endpointSubsetList {
		if s, ok := ep.(string); ok {
			endpoints = append(endpoints, s)
		}
	}

	sort.Strings(endpoints)
	return strings.Join(endpoints, "|")
}

// runCleanup periodically removes expired entries from the cache to prevent unbounded growth.
func (c *CachedPodLocator) runCleanup(ctx context.Context) {
	logger := log.FromContext(ctx).WithName("CachedPodLocatorCleanup")
	ticker := time.NewTicker(cleanupInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			logger.V(logutil.DEBUG).Info("Stopping cleanup routine")
			return
		case <-ticker.C:
			c.cleanup()
		}
	}
}

// cleanup iterates over the map and removes expired entries.
func (c *CachedPodLocator) cleanup() {
	c.mu.Lock()
	defer c.mu.Unlock()

	now := time.Now()
	for key, entry := range c.cache {
		if now.After(entry.expiry) {
			delete(c.cache, key)
		}
	}
}
