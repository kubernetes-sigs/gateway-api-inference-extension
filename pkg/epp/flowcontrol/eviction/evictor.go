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

package eviction

import (
	"context"
	"fmt"
	"sync"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
)

// Evictor handles evicting an in-flight request on a model server.
type Evictor interface {
	Evict(ctx context.Context, item *flowcontrol.EvictionItem) error
}

// NoOpEvictor logs the eviction but does not evict the request on the model server.
type NoOpEvictor struct{}

func (e *NoOpEvictor) Evict(ctx context.Context, item *flowcontrol.EvictionItem) error {
	log.FromContext(ctx).V(logutil.DEBUG).Info("Eviction selected request (no-op: eviction mechanism not available)",
		"requestID", item.RequestID,
		"priority", item.Priority,
		"targetURL", item.TargetURL)
	return nil
}

// ImmediateResponseEvictor evicts requests by closing the EvictionItem's EvictCh.
// The ext_proc Process() goroutine selects on this channel and sends an ImmediateResponse
// to Envoy when it is closed, causing Envoy to reset the upstream connection to the model server.
type ImmediateResponseEvictor struct {
	// closeOnce tracks which channels have been closed to prevent double-close panics.
	closeOnce sync.Map // requestID → *sync.Once
}

// NewImmediateResponseEvictor creates an ImmediateResponseEvictor.
func NewImmediateResponseEvictor() *ImmediateResponseEvictor {
	return &ImmediateResponseEvictor{}
}

func (e *ImmediateResponseEvictor) Evict(ctx context.Context, item *flowcontrol.EvictionItem) error {
	if item.EvictCh == nil {
		return fmt.Errorf("eviction item %s has no eviction channel", item.RequestID)
	}

	once, _ := e.closeOnce.LoadOrStore(item.RequestID, &sync.Once{})
	once.(*sync.Once).Do(func() {
		close(item.EvictCh)
	})

	log.FromContext(ctx).Info("Eviction signal sent",
		"requestID", item.RequestID,
		"priority", item.Priority,
		"targetURL", item.TargetURL)
	return nil
}

// Cleanup removes the sync.Once entry for a request ID to prevent unbounded map growth.
// Called when a request completes or is untracked.
func (e *ImmediateResponseEvictor) Cleanup(requestID string) {
	e.closeOnce.Delete(requestID)
}
