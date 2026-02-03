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

package concurrencydetector

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewTokenLedger(t *testing.T) {
	tl := newTokenLedger()
	require.NotNil(t, tl)
	require.NotNil(t, tl.tokenCountByReq)

	t.Run("remove non-existent request", func(t *testing.T) {
		tokenCount, ok := tl.remove("dummy")
		assert.False(t, ok)
		assert.Equal(t, int64(0), tokenCount)
	})

	t.Run("add single entry and verify remove", func(t *testing.T) {
		tl.add("req1", "ep1", 100)
		tokenCount, ok := tl.remove("req1")
		require.True(t, ok)
		assert.Equal(t, int64(100), tokenCount)
	})

	t.Run("add overwrites same requestID", func(t *testing.T) {
		tl.add("req2", "ep1", 50)
		tl.add("req2", "ep2", 75) // overwrite with different endpoint and count
		tokenCount, ok := tl.remove("req2")
		require.True(t, ok)
		assert.Equal(t, int64(75), tokenCount)
	})

	t.Run("add multiple different requests", func(t *testing.T) {
		tl.add("req1", "ep1", 10)
		tl.add("req2", "ep1", 20)
		tl.add("req3", "ep2", 30)

		tc1, ok1 := tl.remove("req1")
		tc2, ok2 := tl.remove("req2")
		tc3, ok3 := tl.remove("req3")

		require.True(t, ok1)
		require.True(t, ok2)
		require.True(t, ok3)
		assert.Equal(t, int64(10), tc1)
		assert.Equal(t, int64(20), tc2)
		assert.Equal(t, int64(30), tc3)
	})

	t.Run("remove deletes from ledger", func(t *testing.T) {
		tl.add("req2", "ep1", 100)
		_, ok := tl.remove("req2")
		require.True(t, ok)

		// Second remove should fail
		_, ok = tl.remove("req2")
		assert.False(t, ok)
	})
}

func TestTokenLedger_RemoveForEndpoint(t *testing.T) {
	t.Run("empty ledger returns empty slice", func(t *testing.T) {
		tl := newTokenLedger()
		out := tl.removeForEndpoint("ep1")
		assert.Empty(t, out)
	})

	t.Run("no matching endpoint returns empty slice", func(t *testing.T) {
		tl := newTokenLedger()
		tl.add("req1", "ep1", 10)
		out := tl.removeForEndpoint("ep2")
		assert.Empty(t, out)

		// Original entry should still exist
		tokenCount, ok := tl.remove("req1")
		require.True(t, ok)
		assert.Equal(t, int64(10), tokenCount)
	})

	t.Run("removes single matching entry", func(t *testing.T) {
		tl := newTokenLedger()
		tl.add("req1", "ep1", 50)
		out := tl.removeForEndpoint("ep1")
		require.Len(t, out, 1)
		assert.Equal(t, "ep1", out[0].endpointID)
		assert.Equal(t, int64(50), out[0].tokenCount)

		// Entry should be removed from ledger
		_, ok := tl.remove("req1")
		assert.False(t, ok)
	})

	t.Run("removes multiple matching entries for endpoint", func(t *testing.T) {
		tl := newTokenLedger()
		tl.add("req1", "ep1", 10)
		tl.add("req2", "ep1", 20)
		tl.add("req3", "ep1", 30)

		out := tl.removeForEndpoint("ep1")
		require.Len(t, out, 3)

		// Collect token counts and verify
		tokenCounts := make(map[int64]bool)
		for _, e := range out {
			assert.Equal(t, "ep1", e.endpointID)
			tokenCounts[e.tokenCount] = true
		}
		assert.True(t, tokenCounts[10])
		assert.True(t, tokenCounts[20])
		assert.True(t, tokenCounts[30])

		// All entries should be removed
		_, ok1 := tl.remove("req1")
		_, ok2 := tl.remove("req2")
		_, ok3 := tl.remove("req3")
		assert.False(t, ok1)
		assert.False(t, ok2)
		assert.False(t, ok3)
	})

	t.Run("does not remove entries for other endpoints", func(t *testing.T) {
		tl := newTokenLedger()
		tl.add("req1", "ep1", 10)
		tl.add("req2", "ep2", 20)
		tl.add("req3", "ep1", 30)

		out := tl.removeForEndpoint("ep1")
		require.Len(t, out, 2)

		// ep2 entry should still exist
		tokenCount, ok := tl.remove("req2")
		require.True(t, ok)
		assert.Equal(t, int64(20), tokenCount)
	})
}
