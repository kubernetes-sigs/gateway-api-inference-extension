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
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	reqcommon "sigs.k8s.io/gateway-api-inference-extension/pkg/common/request"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

// --- Test helpers ---

func makeSchedulingResult() *scheduling.SchedulingResult {
	endpoint := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{Address: "10.0.0.1", Port: "8000"},
		nil, nil,
	)
	return &scheduling.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"default": {
				TargetEndpoints: []scheduling.Endpoint{endpoint},
			},
		},
	}
}

func makeLLMRequest(requestID string, priority int) *scheduling.LLMRequest { //nolint:unparam
	return &scheduling.LLMRequest{
		RequestId: requestID,
		Headers: map[string]string{
			reqcommon.RequestIdHeaderKey: requestID,
		},
		Objectives: scheduling.RequestObjectives{Priority: priority},
	}
}

// --- Tests ---

func TestPlugin_PreRequest_CreatesAbortChannel(t *testing.T) {
	t.Parallel()
	p := NewPlugin(&testOrdering{}, &acceptAllFilter{}, &NoOpAborter{})

	ctx := context.Background()
	p.PreRequest(ctx, makeLLMRequest("req-1", -1), makeSchedulingResult())

	// Verify the abort channel is registered.
	abortCh := p.AbortRegistry().Get("req-1")
	require.NotNil(t, abortCh, "AbortCh should be registered after PreRequest")

	// Verify tracked in queue.
	assert.Equal(t, 1, p.queue.InFlightLen())
	assert.Equal(t, 1, p.queue.EvictableLen())
}

func TestPlugin_ResponseBody_DeregistersAbortChannel(t *testing.T) {
	t.Parallel()
	p := NewPlugin(&testOrdering{}, &acceptAllFilter{}, &NoOpAborter{})

	ctx := context.Background()
	request := makeLLMRequest("req-1", -1)
	p.PreRequest(ctx, request, makeSchedulingResult())
	require.NotNil(t, p.AbortRegistry().Get("req-1"))

	// Complete the request.
	p.ResponseBody(ctx, request, &requestcontrol.Response{EndOfStream: true}, nil)

	assert.Nil(t, p.AbortRegistry().Get("req-1"), "AbortCh should be deregistered after completion")
	assert.Equal(t, 0, p.queue.InFlightLen())
}

func TestPlugin_EvictN_ClosesAbortChannel(t *testing.T) {
	t.Parallel()
	aborter := NewImmediateResponseAborter()
	p := NewPlugin(&testOrdering{}, &acceptAllFilter{}, aborter)

	ctx := context.Background()
	p.PreRequest(ctx, makeLLMRequest("req-1", -1), makeSchedulingResult())

	// Grab the channel before eviction.
	abortCh := p.AbortRegistry().Get("req-1")
	require.NotNil(t, abortCh)

	// Evict.
	aborted, err := p.EvictN(ctx, 1)
	require.NoError(t, err)
	require.Equal(t, []string{"req-1"}, aborted)

	// Channel should be closed.
	select {
	case <-abortCh:
		// success
	default:
		t.Fatal("abort channel should be closed after EvictN")
	}
}

func TestPlugin_EvictN_ReTracksOnAbortFailure(t *testing.T) {
	t.Parallel()
	p := NewPlugin(&testOrdering{}, &acceptAllFilter{}, &failingAborter{})

	ctx := context.Background()
	p.PreRequest(ctx, makeLLMRequest("req-1", -1), makeSchedulingResult())

	aborted, err := p.EvictN(ctx, 1)
	require.NoError(t, err)
	assert.Empty(t, aborted)

	// Item should be re-tracked.
	assert.Equal(t, 1, p.queue.EvictableLen())
}

func TestPlugin_RaceBetweenEvictAndCompletion(t *testing.T) {
	t.Parallel()
	aborter := NewImmediateResponseAborter()
	p := NewPlugin(&testOrdering{}, &acceptAllFilter{}, aborter)

	ctx := context.Background()

	// Track multiple requests.
	requests := make([]*scheduling.LLMRequest, 10)
	for i := range requests {
		requests[i] = makeLLMRequest(
			"req-"+string(rune('a'+i)),
			-1,
		)
		p.PreRequest(ctx, requests[i], makeSchedulingResult())
	}

	// Concurrently evict and complete.
	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		for range 5 {
			_, _ = p.EvictN(ctx, 1)
			time.Sleep(time.Millisecond)
		}
	}()

	go func() {
		defer wg.Done()
		for _, req := range requests {
			p.ResponseBody(ctx, req, &requestcontrol.Response{EndOfStream: true}, nil)
			time.Sleep(time.Millisecond)
		}
	}()

	wg.Wait()

	// No panics, no deadlocks. State should be consistent.
	inFlight, evictable := p.Stats()
	assert.GreaterOrEqual(t, inFlight, 0)
	assert.GreaterOrEqual(t, evictable, 0)
	assert.GreaterOrEqual(t, inFlight, evictable)
}

// failingAborter always returns an error.
type failingAborter struct{}

func (a *failingAborter) Abort(_ context.Context, _ *flowcontrol.EvictionItem) error {
	return assert.AnError
}
