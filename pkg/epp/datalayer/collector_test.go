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
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer/mocks"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	datasourcemocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/source/mocks"
)

// errSource is a test stub that returns a configurable error from Poll.
// The error is guarded by a mutex so it can safely be changed between ticks.
type errSource struct {
	datasourcemocks.MetricsDataSource
	mu  sync.Mutex
	err error
}

func (e *errSource) setErr(err error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.err = err
}

func (e *errSource) Poll(_ context.Context, _ fwkdl.Endpoint) (any, error) {
	atomic.AddInt64(&e.CallCount, 1)
	e.mu.Lock()
	defer e.mu.Unlock()
	return nil, e.err
}

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
	sources  = []fwkdl.PollingDataSource{&datasourcemocks.MetricsDataSource{}}
)

func TestCollectorCanStartOnlyOnce(t *testing.T) {
	c := NewCollector()
	ctx := context.Background()
	ticker := mocks.NewTicker()

	err := c.Start(ctx, ticker, endpoint, sources, nil)
	require.NoError(t, err, "first Start call should succeed")

	err = c.Start(ctx, ticker, endpoint, sources, nil)
	assert.Error(t, err, "multiple collector start should error")
}

func TestCollectorStopBeforeStartIsAnError(t *testing.T) {
	c := NewCollector()
	err := c.Stop()
	assert.Error(t, err, "collector stop called before start should error")
}

func TestCollectorCanStopOnlyOnce(t *testing.T) {
	c := NewCollector()
	ctx := context.Background()
	ticker := mocks.NewTicker()

	require.NoError(t, c.Start(ctx, ticker, endpoint, sources, nil))
	require.NoError(t, c.Stop(), "first Stop should succeed")
	assert.Error(t, c.Stop(), "second Stop should fail")
}

func TestCollectorCollectsOnTicks(t *testing.T) {
	source := &datasourcemocks.MetricsDataSource{}
	c := NewCollector()
	ticker := mocks.NewTicker()
	ctx := context.Background()

	require.NoError(t, c.Start(ctx, ticker, endpoint, []fwkdl.PollingDataSource{source}, nil))
	ticker.Tick()
	ticker.Tick()

	// use Eventually for async processing
	require.Eventually(t, func() bool {
		return atomic.LoadInt64(&source.CallCount) == 2
	}, 1*time.Second, 2*time.Millisecond, "expected 2 collections")

	require.NoError(t, c.Stop())
}

func TestCollectorStopCancelsContext(t *testing.T) {
	source := &datasourcemocks.MetricsDataSource{}
	c := NewCollector()
	ticker := mocks.NewTicker()
	ctx := context.Background()

	require.NoError(t, c.Start(ctx, ticker, endpoint, []fwkdl.PollingDataSource{source}, nil))
	ticker.Tick() // should be processed
	time.Sleep(20 * time.Millisecond)

	require.NoError(t, c.Stop())
	before := atomic.LoadInt64(&source.CallCount)

	ticker.Tick()
	time.Sleep(20 * time.Millisecond) // let collector run again
	after := atomic.LoadInt64(&source.CallCount)
	assert.Equal(t, before, after, "call count changed after stop")
}

func TestCollectorStartSourceValidation(t *testing.T) {
	tests := []struct {
		name    string
		sources []fwkdl.PollingDataSource
		wantErr bool
	}{
		{
			name:    "empty sources returns error",
			sources: []fwkdl.PollingDataSource{},
			wantErr: true,
		},
		{
			name:    "nil source returns error",
			sources: []fwkdl.PollingDataSource{nil},
			wantErr: true,
		},
		{
			name:    "valid polling source succeeds",
			sources: []fwkdl.PollingDataSource{&datasourcemocks.MetricsDataSource{}},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := NewCollector()
			ticker := mocks.NewTicker()
			ctx := context.Background()

			err := c.Start(ctx, ticker, endpoint, tt.sources, nil)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				require.NoError(t, c.Stop())
			}
		})
	}
}

func TestCollectorLogsFirstPollError(t *testing.T) {
	pollErr := errors.New("metric family not found")
	src := &errSource{err: pollErr}
	c := NewCollector()
	ticker := mocks.NewTicker()
	ctx := context.Background()

	require.NoError(t, c.Start(ctx, ticker, endpoint, []fwkdl.PollingDataSource{src}, nil))

	ticker.Tick()
	ticker.Tick()
	ticker.Tick()

	require.Eventually(t, func() bool {
		return atomic.LoadInt64(&src.CallCount) == 3
	}, 1*time.Second, 2*time.Millisecond, "expected 3 poll calls")

	require.NoError(t, c.Stop()) // Stop waits for the goroutine to exit

	// Error is recorded exactly once regardless of how many ticks delivered it.
	key := src.TypedName().String()
	require.Len(t, c.lastPollErrors, 1, "expected exactly one error state entry")
	assert.Equal(t, pollErr, c.lastPollErrors[key])
}

func TestCollectorLogsRecoveryAfterError(t *testing.T) {
	pollErr := errors.New("transient error")
	src := &errSource{err: pollErr}
	c := NewCollector()
	ticker := mocks.NewTicker()
	ctx := context.Background()

	require.NoError(t, c.Start(ctx, ticker, endpoint, []fwkdl.PollingDataSource{src}, nil))

	// Two ticks with an error.
	ticker.Tick()
	ticker.Tick()
	require.Eventually(t, func() bool {
		return atomic.LoadInt64(&src.CallCount) == 2
	}, 1*time.Second, 2*time.Millisecond, "expected 2 poll calls with error")

	// Clear the error, then send more ticks.
	src.setErr(nil)
	ticker.Tick()
	ticker.Tick()
	require.Eventually(t, func() bool {
		return atomic.LoadInt64(&src.CallCount) == 4
	}, 1*time.Second, 2*time.Millisecond, "expected 4 total poll calls after recovery")

	require.NoError(t, c.Stop()) // Stop waits for the goroutine to exit

	// After recovery the entry exists but holds nil.
	key := src.TypedName().String()
	entry, seen := c.lastPollErrors[key]
	require.True(t, seen, "recovery should leave a nil entry in lastPollErrors")
	assert.Nil(t, entry)
}

func TestCollectorContinuesAfterPollError(t *testing.T) {
	src := &errSource{err: errors.New("persistent error")}
	c := NewCollector()
	ticker := mocks.NewTicker()
	ctx := context.Background()

	require.NoError(t, c.Start(ctx, ticker, endpoint, []fwkdl.PollingDataSource{src}, nil))

	ticker.Tick()
	ticker.Tick()
	ticker.Tick()

	require.Eventually(t, func() bool {
		return atomic.LoadInt64(&src.CallCount) == 3
	}, 1*time.Second, 2*time.Millisecond, "poll should be called on every tick despite errors")

	require.NoError(t, c.Stop())
}
