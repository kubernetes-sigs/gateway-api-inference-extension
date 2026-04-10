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

func TestRequestEvictor_PreRequest_CreatesAbortChannel(t *testing.T) {
	t.Parallel()
	re := NewRequestEvictor(&testOrdering{}, &acceptAllFilter{}, &NoOpEvictor{})

	ctx := context.Background()
	re.PreRequest(ctx, makeLLMRequest("req-1", -1), makeSchedulingResult())

	abortCh := re.AbortRegistry().Get("req-1")
	require.NotNil(t, abortCh, "AbortCh should be registered after PreRequest")

	assert.Equal(t, 1, re.queue.InFlightLen())
	assert.Equal(t, 1, re.queue.EvictableLen())
}

func TestRequestEvictor_ResponseBody_DeregistersAbortChannel(t *testing.T) {
	t.Parallel()
	re := NewRequestEvictor(&testOrdering{}, &acceptAllFilter{}, &NoOpEvictor{})

	ctx := context.Background()
	request := makeLLMRequest("req-1", -1)
	re.PreRequest(ctx, request, makeSchedulingResult())
	require.NotNil(t, re.AbortRegistry().Get("req-1"))

	re.ResponseBody(ctx, request, &requestcontrol.Response{EndOfStream: true}, nil)

	assert.Nil(t, re.AbortRegistry().Get("req-1"), "AbortCh should be deregistered after completion")
	assert.Equal(t, 0, re.queue.InFlightLen())
}

func TestRequestEvictor_EvictN_ClosesAbortChannel(t *testing.T) {
	t.Parallel()
	evictor := NewImmediateResponseEvictor()
	re := NewRequestEvictor(&testOrdering{}, &acceptAllFilter{}, evictor)

	ctx := context.Background()
	re.PreRequest(ctx, makeLLMRequest("req-1", -1), makeSchedulingResult())

	abortCh := re.AbortRegistry().Get("req-1")
	require.NotNil(t, abortCh)

	evicted, err := re.EvictN(ctx, 1)
	require.NoError(t, err)
	require.Equal(t, []string{"req-1"}, evicted)

	select {
	case <-abortCh:
	default:
		t.Fatal("abort channel should be closed after EvictN")
	}
}

func TestRequestEvictor_EvictN_ReTracksOnFailure(t *testing.T) {
	t.Parallel()
	re := NewRequestEvictor(&testOrdering{}, &acceptAllFilter{}, &failingEvictor{})

	ctx := context.Background()
	re.PreRequest(ctx, makeLLMRequest("req-1", -1), makeSchedulingResult())

	evicted, err := re.EvictN(ctx, 1)
	require.NoError(t, err)
	assert.Empty(t, evicted)

	assert.Equal(t, 1, re.queue.EvictableLen())
}

func TestRequestEvictor_RaceBetweenEvictAndCompletion(t *testing.T) {
	t.Parallel()
	evictor := NewImmediateResponseEvictor()
	re := NewRequestEvictor(&testOrdering{}, &acceptAllFilter{}, evictor)

	ctx := context.Background()

	requests := make([]*scheduling.LLMRequest, 10)
	for i := range requests {
		requests[i] = makeLLMRequest("req-"+string(rune('a'+i)), -1)
		re.PreRequest(ctx, requests[i], makeSchedulingResult())
	}

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		for range 5 {
			_, _ = re.EvictN(ctx, 1)
			time.Sleep(time.Millisecond)
		}
	}()

	go func() {
		defer wg.Done()
		for _, req := range requests {
			re.ResponseBody(ctx, req, &requestcontrol.Response{EndOfStream: true}, nil)
			time.Sleep(time.Millisecond)
		}
	}()

	wg.Wait()

	inFlight, evictable := re.Stats()
	assert.GreaterOrEqual(t, inFlight, 0)
	assert.GreaterOrEqual(t, evictable, 0)
	assert.GreaterOrEqual(t, inFlight, evictable)
}

// failingEvictor always returns an error.
type failingEvictor struct{}

func (e *failingEvictor) Evict(_ context.Context, _ *flowcontrol.EvictionItem) error {
	return assert.AnError
}
