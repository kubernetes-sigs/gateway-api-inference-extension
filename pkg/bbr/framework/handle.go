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

package framework

import (
	"context"

	ctrl "sigs.k8s.io/controller-runtime"
	ctrlbuilder "sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// Handle provides plugins a set of standard data and tools to work with
type Handle interface {
	// Context returns a context the plugins can use, if they need one
	Context() context.Context
	ClientReader() client.Reader
	ReconcilerBuilder() *ctrlbuilder.Builder
}

// bbrHandle is an implementation of the Handle interface.
type bbrHandle struct {
	ctx context.Context
	mgr ctrl.Manager
}

// Context returns a context the plugins can use, if they need one
func (h *bbrHandle) Context() context.Context {
	return h.ctx
}

func (h *bbrHandle) ClientReader() client.Reader {
	return h.mgr.GetAPIReader()
}

func (h *bbrHandle) ReconcilerBuilder() *ctrlbuilder.Builder {
	return ctrl.NewControllerManagedBy(h.mgr)
}

func NewBbrHandle(ctx context.Context, mgr ctrl.Manager) Handle {
	return &bbrHandle{
		ctx: ctx,
		mgr: mgr,
	}
}
