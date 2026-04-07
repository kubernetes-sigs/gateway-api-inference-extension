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

package inflightload

import (
	"context"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	fwkrh "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrconcurrency "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
)

func TestInFlightLoadProducer_PrepareRequestData(t *testing.T) {
	t.Parallel()

	producer := &InFlightLoadProducer{
		requestTracker: newConcurrencyTracker(),
		tokenTracker:   newConcurrencyTracker(),
	}

	endpointName := "test-endpoint"
	endpointID := fullEndpointName(endpointName)

	// Mock some initial load
	producer.requestTracker.add(endpointID, 5)
	producer.tokenTracker.add(endpointID, 500)

	ctx := context.Background()
	endpoints := []schedulingtypes.Endpoint{newStubSchedulingEndpoint(endpointName)}

	err := producer.PrepareRequestData(ctx, nil, endpoints)
	require.NoError(t, err)

	// Verify AttributeMap population
	val, ok := endpoints[0].Get(attrconcurrency.InFlightLoadKey)
	require.True(t, ok)
	load := val.(*attrconcurrency.InFlightLoad)
	require.Equal(t, int64(5), load.Requests)
	require.Equal(t, int64(500), load.Tokens)
}

func TestInFlightLoadProducer_Lifecycle(t *testing.T) {
	t.Parallel()

	producer := &InFlightLoadProducer{
		requestTracker: newConcurrencyTracker(),
		tokenTracker:   newConcurrencyTracker(),
		tokenEstimator: NewSimpleTokenEstimator(),
	}
	ctx := context.Background()
	endpointName := "lifecycle-endpoint"
	endpointID := fullEndpointName(endpointName)

	// 1. PreRequest (Inc)
	req := makeTokenRequest("req1", "1234567890123456") // 16 chars / 4 = 4 input + 6 output = 10 tokens
	res := makeSchedulingResult(endpointName)
	producer.PreRequest(ctx, req, res)

	require.Equal(t, int64(1), producer.requestTracker.get(endpointID))
	require.Equal(t, int64(10), producer.tokenTracker.get(endpointID))

	// 2. ResponseBody EndOfStream (Dec)
	targetEndpoint := &datalayer.EndpointMetadata{NamespacedName: types.NamespacedName{Name: endpointName, Namespace: "default"}}
	producer.ResponseBody(ctx, req, &requestcontrol.Response{EndOfStream: true}, targetEndpoint)

	require.Equal(t, int64(0), producer.requestTracker.get(endpointID))
	require.Equal(t, int64(0), producer.tokenTracker.get(endpointID))
}

func TestInFlightLoadProducer_NotificationCleanup(t *testing.T) {
	t.Parallel()

	producer := &InFlightLoadProducer{
		requestTracker: newConcurrencyTracker(),
		tokenTracker:   newConcurrencyTracker(),
	}
	ctx := context.Background()
	endpointName := "deleted-endpoint"
	endpointID := fullEndpointName(endpointName)

	// Seed load
	producer.requestTracker.add(endpointID, 10)
	producer.tokenTracker.add(endpointID, 1000)

	// Simulate Delete Notification
	pod := &unstructured.Unstructured{}
	pod.SetNamespace("default")
	pod.SetName(endpointName)

	event := datalayer.NotificationEvent{
		Type:   datalayer.EventDelete,
		Object: pod,
	}

	err := producer.ExtractNotification(ctx, event)
	require.NoError(t, err)

	// Verify Cleanup
	require.Equal(t, int64(0), producer.requestTracker.get(endpointID))
	require.Equal(t, int64(0), producer.tokenTracker.get(endpointID))
}

func TestInFlightLoadProducer_ConcurrencyStress(t *testing.T) {
	t.Parallel()

	producer := &InFlightLoadProducer{
		requestTracker: newConcurrencyTracker(),
		tokenTracker:   newConcurrencyTracker(),
		tokenEstimator: NewSimpleTokenEstimator(),
	}
	ctx := context.Background()
	endpointName := "stress-endpoint"
	endpointID := fullEndpointName(endpointName)

	const (
		numGoroutines = 50
		opsPerRoutine = 1000
	)

	var wg sync.WaitGroup
	wg.Add(numGoroutines * 2)

	// Launch increments
	for range numGoroutines {
		go func() {
			defer wg.Done()
			res := makeSchedulingResult(endpointName)
			for range opsPerRoutine {
				producer.PreRequest(ctx, nil, res)
			}
		}()
	}

	// Launch decrements
	for range numGoroutines {
		go func() {
			defer wg.Done()
			targetEndpoint := &datalayer.EndpointMetadata{NamespacedName: types.NamespacedName{Name: endpointName, Namespace: "default"}}
			for range opsPerRoutine {
				producer.ResponseBody(ctx, nil, &requestcontrol.Response{EndOfStream: true}, targetEndpoint)
			}
		}()
	}

	wg.Wait()

	require.Equal(t, int64(0), producer.requestTracker.get(endpointID), "request count drift detected")
}

// --- Helpers ---

func fullEndpointName(name string) string {
	return types.NamespacedName{Name: name, Namespace: "default"}.String()
}

func makeSchedulingResult(endpointName string) *schedulingtypes.SchedulingResult {
	return &schedulingtypes.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*schedulingtypes.ProfileRunResult{
			"default": {
				TargetEndpoints: []schedulingtypes.Endpoint{newStubSchedulingEndpoint(endpointName)},
			},
		},
	}
}

type stubSchedulingEndpoint struct {
	schedulingtypes.Endpoint
	metadata *datalayer.EndpointMetadata
	attr     datalayer.AttributeMap
}

func newStubSchedulingEndpoint(name string) *stubSchedulingEndpoint {
	return &stubSchedulingEndpoint{
		metadata: &datalayer.EndpointMetadata{NamespacedName: types.NamespacedName{Name: name, Namespace: "default"}},
		attr:     datalayer.NewAttributes(),
	}
}

func (f *stubSchedulingEndpoint) GetMetadata() *datalayer.EndpointMetadata { return f.metadata }
func (f *stubSchedulingEndpoint) Put(key string, val datalayer.Cloneable)  { f.attr.Put(key, val) }
func (f *stubSchedulingEndpoint) Get(key string) (datalayer.Cloneable, bool) {
	return f.attr.Get(key)
}
func (f *stubSchedulingEndpoint) Keys() []string { return f.attr.Keys() }

func makeTokenRequest(requestID, prompt string) *schedulingtypes.InferenceRequest {
	return &schedulingtypes.InferenceRequest{
		RequestId: requestID,
		Body: &fwkrh.InferenceRequestBody{
			Completions: &fwkrh.CompletionsRequest{Prompt: fwkrh.Prompt{Raw: prompt}},
		},
	}
}
