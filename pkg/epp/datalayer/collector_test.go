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

package datalayer

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer/mocks"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
)

// --- Test Stubs ---

func defaultEndpoint() fwkdl.Endpoint {
	meta := &fwkdl.EndpointMetadata{
		NamespacedName: types.NamespacedName{
			Name:      "pod-name",
			Namespace: "default",
		},
		Address: "1.2.3.4:5678",
	}
	ms := fwkdl.NewEndpoint(meta, nil)
	return ms
}

// --- Tests ---

var (
	endpoint = defaultEndpoint()
	sources  = []fwkdl.DataSource{&FakeDataSource{}}
)

func TestCollectorCanStartOnlyOnce(t *testing.T) {
	c := NewCollector(nil)
	ctx := context.Background()
	ticker := mocks.NewTicker()

	err := c.Start(ctx, ticker, endpoint, sources)
	require.NoError(t, err, "first Start call should succeed")

	err = c.Start(ctx, ticker, endpoint, sources)
	assert.Error(t, err, "multiple collector start should error")
}

func TestCollectorStopBeforeStartIsAnError(t *testing.T) {
	c := NewCollector(nil)
	err := c.Stop()
	assert.Error(t, err, "collector stop called before start should error")
}

func TestCollectorCanStopOnlyOnce(t *testing.T) {
	c := NewCollector(nil)
	ctx := context.Background()
	ticker := mocks.NewTicker()

	require.NoError(t, c.Start(ctx, ticker, endpoint, sources))
	require.NoError(t, c.Stop(), "first Stop should succeed")
	assert.Error(t, c.Stop(), "second Stop should fail")
}

func TestCollectorCollectsOnTicks(t *testing.T) {
	source := &FakeDataSource{}
	c := NewCollector(nil)
	ticker := mocks.NewTicker()
	ctx := context.Background()

	require.NoError(t, c.Start(ctx, ticker, endpoint, []fwkdl.DataSource{source}))
	ticker.Tick()
	ticker.Tick()

	// use Eventually for async processing
	require.Eventually(t, func() bool {
		return atomic.LoadInt64(&source.callCount) == 2
	}, 1*time.Second, 2*time.Millisecond, "expected 2 collections")

	require.NoError(t, c.Stop())
}

func TestCollectorStopCancelsContext(t *testing.T) {
	source := &FakeDataSource{}
	c := NewCollector(nil)
	ticker := mocks.NewTicker()
	ctx := context.Background()

	require.NoError(t, c.Start(ctx, ticker, endpoint, []fwkdl.DataSource{source}))
	ticker.Tick() // should be processed
	time.Sleep(20 * time.Millisecond)

	require.NoError(t, c.Stop())
	before := atomic.LoadInt64(&source.callCount)

	ticker.Tick()
	time.Sleep(20 * time.Millisecond) // let collector run again
	after := atomic.LoadInt64(&source.callCount)
	assert.Equal(t, before, after, "call count changed after stop")
}
