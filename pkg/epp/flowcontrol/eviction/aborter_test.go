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

func TestImmediateResponseAborter_ClosesChannel(t *testing.T) {
	t.Parallel()
	aborter := NewImmediateResponseAborter()

	abortCh := make(chan struct{})
	item := &flowcontrol.EvictionItem{
		RequestID: "req-1",
		AbortCh:   abortCh,
	}

	err := aborter.Abort(context.Background(), item)
	require.NoError(t, err)

	// Channel should be closed.
	select {
	case <-abortCh:
		// success — channel is closed
	default:
		t.Fatal("abort channel should be closed after Abort()")
	}
}

func TestImmediateResponseAborter_DoubleAbortSafe(t *testing.T) {
	t.Parallel()
	aborter := NewImmediateResponseAborter()

	abortCh := make(chan struct{})
	item := &flowcontrol.EvictionItem{
		RequestID: "req-1",
		AbortCh:   abortCh,
	}

	// First abort should succeed.
	err := aborter.Abort(context.Background(), item)
	require.NoError(t, err)

	// Second abort on same request should not panic.
	err = aborter.Abort(context.Background(), item)
	require.NoError(t, err)
}

func TestImmediateResponseAborter_NilChannel(t *testing.T) {
	t.Parallel()
	aborter := NewImmediateResponseAborter()

	item := &flowcontrol.EvictionItem{
		RequestID: "req-1",
		AbortCh:   nil,
	}

	err := aborter.Abort(context.Background(), item)
	assert.Error(t, err, "Abort with nil channel should return error")
}

func TestNoOpAborter(t *testing.T) {
	t.Parallel()
	aborter := &NoOpAborter{}

	item := &flowcontrol.EvictionItem{
		RequestID: "req-1",
	}

	err := aborter.Abort(context.Background(), item)
	assert.NoError(t, err, "NoOpAborter should always succeed")
}
