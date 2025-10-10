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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	backendmetrics "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/handlers"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

const (
	PreScheduleExtensionPoint  = "PreSchedule"
	PreRequestExtensionPoint   = "PreRequest"
	PostResponseExtensionPoint = "PostResponse"
)

// PreSchedule is called by the director before sending the request to the scheduler.
// It gets the set of candidate pods to be filtered and scored.
type PreSchedule interface {
	plugins.Plugin
	GetCandidatePods(ctx context.Context, request *handlers.Request) []backendmetrics.PodMetrics
}

// PreRequest is called by the director after a getting result from scheduling layer and
// before a request is sent to the selected model server.
type PreRequest interface {
	plugins.Plugin
	PreRequest(ctx context.Context, request *types.LLMRequest, schedulingResult *types.SchedulingResult, targetPort int)
}

// PostResponse is called by the director after a successful response was sent.
// The given pod argument is the pod that served the request.
type PostResponse interface {
	plugins.Plugin
	PostResponse(ctx context.Context, request *types.LLMRequest, response *Response, targetPod *backend.Pod)
}
