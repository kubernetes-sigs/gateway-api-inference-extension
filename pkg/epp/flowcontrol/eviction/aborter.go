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

// Aborter handles aborting an in-flight request on a model server.
type Aborter interface {
	Abort(ctx context.Context, item *flowcontrol.EvictionItem) error
}

// NoOpAborter logs the eviction but does not abort the request on the model server.
type NoOpAborter struct{}

func (a *NoOpAborter) Abort(ctx context.Context, item *flowcontrol.EvictionItem) error {
	log.FromContext(ctx).V(logutil.DEBUG).Info("Eviction selected request for abort (no-op: abort mechanism not available)",
		"requestID", item.RequestID,
		"priority", item.Priority,
		"targetURL", item.TargetURL)
	return nil
}

// ImmediateResponseAborter aborts requests by closing the EvictionItem's AbortCh.
// The ext_proc Process() goroutine selects on this channel and sends an ImmediateResponse
// to Envoy when it is closed, causing Envoy to reset the upstream connection to the model server.
type ImmediateResponseAborter struct {
	// closeOnce tracks which channels have been closed to prevent double-close panics.
	closeOnce sync.Map // requestID → *sync.Once
}

// NewImmediateResponseAborter creates an ImmediateResponseAborter.
func NewImmediateResponseAborter() *ImmediateResponseAborter {
	return &ImmediateResponseAborter{}
}

func (a *ImmediateResponseAborter) Abort(ctx context.Context, item *flowcontrol.EvictionItem) error {
	if item.AbortCh == nil {
		return fmt.Errorf("eviction item %s has no abort channel", item.RequestID)
	}

	// Use sync.Once to safely close the channel exactly once.
	once, _ := a.closeOnce.LoadOrStore(item.RequestID, &sync.Once{})
	once.(*sync.Once).Do(func() {
		close(item.AbortCh)
	})

	log.FromContext(ctx).Info("Abort signal sent",
		"requestID", item.RequestID,
		"priority", item.Priority,
		"targetURL", item.TargetURL)
	return nil
}
