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

package types

import (
	"context"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

func NewSchedulingContext(ctx context.Context, req *LLMRequest, resp *LLMResponse, pods []Pod) *SchedulingContext {

	logger := log.FromContext(ctx).WithValues("request", req)
	return &SchedulingContext{
		Context:      ctx,
		Logger:       logger,
		Req:          req,
		Resp:         resp,
		PodsSnapshot: pods,
		CycleState:   NewCycleState(),
	}
}

// SchedulingContext holds contextual information during a scheduling operation.
type SchedulingContext struct {
	context.Context
	Logger       logr.Logger
	Req          *LLMRequest
	Resp         *LLMResponse
	PodsSnapshot []Pod
	// CycleState can be used by plugins to store state during a scheduling cycle, to communicate
	// between different extension points.
	CycleState *CycleState
}
