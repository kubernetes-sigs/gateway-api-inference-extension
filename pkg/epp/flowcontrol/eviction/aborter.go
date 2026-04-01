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

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
)

// Aborter handles aborting an in-flight request on a model server.
type Aborter interface {
	Abort(ctx context.Context, item *flowcontrol.EvictionItem) error
}

// NoOpAborter logs the eviction but does not abort the request on the model server.
// This is the default aborter until a general-purpose abort mechanism is available
// (e.g., vLLM exposes /v1/abort_requests for all request types, or Envoy ext_proc
// supports downstream connection termination).
type NoOpAborter struct{}

func (a *NoOpAborter) Abort(ctx context.Context, item *flowcontrol.EvictionItem) error {
	log.FromContext(ctx).V(logutil.DEBUG).Info("Eviction selected request for abort (no-op: abort mechanism not available)",
		"requestID", item.RequestID,
		"priority", item.Priority,
		"targetURL", item.TargetURL)
	return nil
}
