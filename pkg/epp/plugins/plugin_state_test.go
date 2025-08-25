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

package plugins

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// pluginTestData implements the StateData interface for testing purposes.
// It provides a simple string value that can be stored and retrieved.
type pluginTestData struct {
	value string
}

// Clone implements the StateData interface, creating a deep copy of the data.
func (d *pluginTestData) Clone() StateData {
	if d == nil {
		return nil
	}
	return &pluginTestData{value: d.value}
}

// TestPluginState_ReadWrite verifies the basic operations of PluginState:
// - Writing data for a request
// - Reading the data back
// - Deleting the data and confirming it's removed
func TestPluginState_ReadWrite(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	state := NewPluginState(ctx)

	req1 := "req1"
	key := StateKey("key")
	data1 := "bar1"
	req2 := "req2"
	data2 := "bar2"

	// Write data to the state storage
	state.Write(req1, key, &pluginTestData{value: data1})
	state.Write(req2, key, &pluginTestData{value: data2})

	// Read back the req1 data and verify its content
	readData, err := state.Read(req1, key)
	assert.NoError(t, err)
	td, ok := readData.(*pluginTestData)
	assert.True(t, ok, "should be able to cast to pluginTestData")
	assert.Equal(t, data1, td.value)

	// Delete the req2 data and verify content that was read before is still valid
	readData, err = state.Read(req2, key)
	assert.NoError(t, err)
	state.Delete(req2)
	td, ok = readData.(*pluginTestData)
	assert.True(t, ok, "should be able to cast to pluginTestData")
	assert.Equal(t, data2, td.value)
	// try to read again aftet deletion, verify error
	readData, err = state.Read(req2, key)
	assert.Equal(t, ErrNotFound, err)
	assert.Nil(t, readData, "expected no data after delete")

	// Read back the req1 data and verify its content after the req2 deleted
	readData, err = state.Read(req1, key)
	assert.NoError(t, err)
	td, ok = readData.(*pluginTestData)
	assert.True(t, ok, "should be able to cast to pluginTestData")
	assert.Equal(t, data1, td.value)
}

// TestReadPluginStateKey tests the generic helper function ReadPluginStateKey which provides
// type-safe access to stored data. It verifies:
// - Successful type assertion and data retrieval
// - Error handling for non-existent keys
func TestReadPluginStateKey(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	state := NewPluginState(ctx)

	requestID := "req-1"
	key := StateKey("foo")
	data := &pluginTestData{value: "bar"}

	state.Write(requestID, key, data)

	// Read
	val, err := ReadPluginStateKey[*pluginTestData](state, requestID, key)
	assert.NoError(t, err)
	assert.Equal(t, "bar", val.value)

	// Not Found
	_, err = ReadPluginStateKey[*pluginTestData](state, "not-exist", key)
	assert.Equal(t, ErrNotFound, err)
}

// TestPluginState_Cleanup verifies the automatic cleanup of stale data.
// It tests that data which hasn't been accessed for longer than stalenessThreshold
// is properly removed from the storage.
func TestPluginState_Cleanup(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	state := NewPluginState(ctx)

	requestID := "req-stale"
	key := StateKey("foo")
	data := &pluginTestData{value: "bar"}

	state.Write(requestID, key, data)

	// Manually set last access time to far in the past
	state.requestToLastAccessTime.Store(requestID, time.Now().Add(-2*stalenessThreshold))
	// Manually CleanUp
	state.cleanStaleRequests()

	_, err := state.Read(requestID, key)
	assert.Equal(t, ErrNotFound, err)
}
