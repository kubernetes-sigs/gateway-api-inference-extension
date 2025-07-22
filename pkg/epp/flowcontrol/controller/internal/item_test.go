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

package internal

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	typesmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types/mocks"
)

func TestItem(t *testing.T) {
	t.Parallel()

	t.Run("should correctly set and get handle", func(t *testing.T) {
		t.Parallel()
		item := &flowItem{}
		handle := &typesmocks.MockQueueItemHandle{}
		item.SetHandle(handle)
		assert.Same(t, handle, item.Handle(), "Handle() should retrieve the same handle instance set by SetHandle()")
	})

	t.Run("should have a non-finalized state upon creation", func(t *testing.T) {
		t.Parallel()
		req := typesmocks.NewMockFlowControlRequest(100, "req-1", "flow-a", context.Background())
		item := NewItem(req, time.Minute, time.Now())
		require.NotNil(t, item, "NewItem should not return nil")
		outcome, err := item.FinalState()
		assert.Equal(t, types.QueueOutcomeNotYetFinalized, outcome, "A new item's outcome should be NotYetFinalized")
		assert.NoError(t, err, "A new item should have a nil error")
	})
}
