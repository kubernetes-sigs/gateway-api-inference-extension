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
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer/mocks"
)

// --- Test Stubs ---

type DummySource struct {
	callCount int64
}

func (d *DummySource) Name() string                   { return "test-dummy-data-source" }
func (d *DummySource) AddExtractor(_ Extractor) error { return nil }
func (d *DummySource) Collect(ctx context.Context, ep Endpoint) error {
	atomic.AddInt64(&d.callCount, 1)
	return nil
}

func defaultEndpoint() Endpoint {
	ms := NewEndpoint()
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "pod-name",
			Namespace: "default",
		},
		Status: corev1.PodStatus{
			PodIP: "1.2.3.4",
		},
	}
	ms.UpdatePod(pod)
	return ms
}

// --- Tests ---

var (
	endpoint = defaultEndpoint()
	sources  = []DataSource{&DummySource{}}
)

func TestCollectorCanStartOnlyOnce(t *testing.T) {
	c := NewCollector()
	ctx := context.Background()
	ticker := mocks.NewTicker()

	err := c.Start(ctx, ticker, endpoint, sources)
	require.NoError(t, err, "first Start call should succeed")

	err = c.Start(ctx, ticker, endpoint, sources)
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

	require.NoError(t, c.Start(ctx, ticker, endpoint, sources))
	require.NoError(t, c.Stop(), "first Stop should succeed")
	assert.Error(t, c.Stop(), "second Stop should fail")
}

func TestCollectorCollectsOnTicks(t *testing.T) {
	source := &DummySource{}
	c := NewCollector()
	ticker := mocks.NewTicker()
	ctx := context.Background()
	require.NoError(t, c.Start(ctx, ticker, endpoint, []DataSource{source}))

	ticker.Tick()
	ticker.Tick()
	time.Sleep(20 * time.Millisecond) // let collector process the ticks

	got := atomic.LoadInt64(&source.callCount)
	want := int64(2)
	assert.Equal(t, want, got, "call count mismatch")
	require.NoError(t, c.Stop())
}

func TestCollectorStopCancelsContext(t *testing.T) {
	source := &DummySource{}
	c := NewCollector()
	ticker := mocks.NewTicker()
	ctx := context.Background()

	require.NoError(t, c.Start(ctx, ticker, endpoint, []DataSource{source}))
	ticker.Tick() // should be processed
	time.Sleep(20 * time.Millisecond)

	require.NoError(t, c.Stop())
	before := atomic.LoadInt64(&source.callCount)

	ticker.Tick()
	time.Sleep(20 * time.Millisecond) // let collector run again
	after := atomic.LoadInt64(&source.callCount)
	assert.Equal(t, before, after, "call count changed after stop")
}
