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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
)

func TestImmediateResponseEvictor_ClosesChannel(t *testing.T) {
	t.Parallel()
	evictor := NewImmediateResponseEvictor()

	abortCh := make(chan struct{})
	item := &flowcontrol.EvictionItem{
		RequestID: "req-1",
		AbortCh:   abortCh,
	}

	err := evictor.Evict(context.Background(), item)
	require.NoError(t, err)

	select {
	case <-abortCh:
	default:
		t.Fatal("abort channel should be closed after Evict()")
	}
}

func TestImmediateResponseEvictor_DoubleEvictSafe(t *testing.T) {
	t.Parallel()
	evictor := NewImmediateResponseEvictor()

	abortCh := make(chan struct{})
	item := &flowcontrol.EvictionItem{
		RequestID: "req-1",
		AbortCh:   abortCh,
	}

	err := evictor.Evict(context.Background(), item)
	require.NoError(t, err)

	// Second evict on same request should not panic.
	err = evictor.Evict(context.Background(), item)
	require.NoError(t, err)
}

func TestImmediateResponseEvictor_NilChannel(t *testing.T) {
	t.Parallel()
	evictor := NewImmediateResponseEvictor()

	item := &flowcontrol.EvictionItem{
		RequestID: "req-1",
		AbortCh:   nil,
	}

	err := evictor.Evict(context.Background(), item)
	assert.Error(t, err, "Evict with nil channel should return error")
}

func TestNoOpEvictor(t *testing.T) {
	t.Parallel()
	evictor := &NoOpEvictor{}

	item := &flowcontrol.EvictionItem{
		RequestID: "req-1",
	}

	err := evictor.Evict(context.Background(), item)
	assert.NoError(t, err, "NoOpEvictor should always succeed")
}
