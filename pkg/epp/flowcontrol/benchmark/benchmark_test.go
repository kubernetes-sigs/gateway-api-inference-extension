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

package benchmark

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/controller"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
)

// BenchmarkFlowController_PerformanceMatrix evaluates throughput across a matrix of variables.
// It systematically evaluates the impact of strict egress limits, data parallelism, priority
// levels, flow density, and concurrent connections.
func BenchmarkFlowController_PerformanceMatrix(b *testing.B) {
	if testing.Short() {
		b.Skip("skipping PerformanceMatrix in short mode")
	}
	limits := []egressConcurrencyLimit{0, 1, 100}
	shards := []shardCount{1, 8}
	priorities := []priorityLevels{1, 8}
	flows := []flowCount{10, 50000}
	concurrencies := []ingressConcurrency{10, 5000, 50000}

	for _, L := range limits {
		for _, S := range shards {
			for _, P := range priorities {
				for _, F := range flows {
					for _, W := range concurrencies {
						// Skip illogical boundaries.
						if L == 0 && W > 100 {
							continue // High concurrency is redundant for free-flow.
						}
						if L > 0 && int64(W) <= int64(L) {
							continue // Requires W > L to generate backpressure.
						}

						matrix := benchMatrix{limit: L, shards: S, priorities: P, flows: F, concurrency: W}
						b.Run(matrix.name(), func(b *testing.B) {
							runMatrixCoordinate(b, matrix)
						})
					}
				}
			}
		}
	}
}

// BenchmarkFlowController_HighShardSort isolates the O(S log S) sorting overhead of shortest-queue
// load balancing by evaluating highly sharded configurations.
func BenchmarkFlowController_HighShardSort(b *testing.B) {
	shards := []shardCount{16, 64, 128, 256}
	for _, S := range shards {
		matrix := benchMatrix{limit: 50, shards: S, priorities: 1, flows: 100, concurrency: 100}
		b.Run(matrix.name(), func(b *testing.B) {
			runMatrixCoordinate(b, matrix)
		})
	}
}

// runMatrixCoordinate executes a single coordinate of the performance hypercube.
func runMatrixCoordinate(b *testing.B, m benchMatrix) {
	ctx, cancel := context.WithCancel(context.Background())

	fc, detector := setupBenchmarkHarness(b, ctx, m.shards, m.priorities, m.limit, nil, nil)

	// Yield briefly to allow the background supervisor to bootstrap the data plane.
	time.Sleep(10 * time.Millisecond)

	reqs := make([]*benchRequest, m.flows)
	for i := 0; i < int(m.flows); i++ {
		// Use the Knuth 32-bit multiplier to deterministically scatter payload sizes (100B - 9KB).
		hash := uint32(i) * 2654435769
		reqs[i] = &benchRequest{
			key: flowcontrol.FlowKey{
				ID:       fmt.Sprintf("flow-%d", i),
				Priority: i % int(m.priorities),
			},
			byteSize: 100 + uint64(hash%9000), // Payload entropy between 100B and 9KB.
		}
	}

	telemetry := newBenchmarkTelemetry()

	// Scale execution threads to match simulated concurrency.
	procs := runtime.GOMAXPROCS(0)
	parallelism := max(int(m.concurrency)/procs, 1)
	b.SetParallelism(parallelism)

	numFlows := int(m.flows)

	// Round up to the next power of two for fast modulo via bitmasking.
	zipfSize := 1
	for zipfSize < numFlows*4 {
		zipfSize <<= 1
	}
	zipfMask := zipfSize - 1
	zipfIndices := make([]int, zipfSize)

	if numFlows > 1 {
		// Pre-compute with a deterministic seed to ensure benchmark consistency.
		// The (1.1, 1.0) parameters bias selections toward lower indices, simulating a "hot tenant".
		rng := rand.New(rand.NewSource(1))
		zipf := rand.NewZipf(rng, 1.1, 1.0, uint64(numFlows-1))
		for i := 0; i < zipfSize; i++ {
			zipfIndices[i] = int(zipf.Uint64())
		}
	}

	b.ResetTimer()
	b.ReportAllocs()

	var globalThreadID atomic.Uint64

	b.RunParallel(func(pb *testing.PB) {
		var localTelemetry threadTelemetry

		// Offset the starting index per thread to prevent identical striding over the array.
		// Multiply by a prime to guarantee threads start at different offsets.
		localIdx := int(globalThreadID.Add(1)) * 9973

		for pb.Next() {
			localIdx++

			flowIdx := zipfIndices[localIdx&zipfMask]
			sourceReq := reqs[flowIdx]
			outcome, err := fc.EnqueueAndWait(ctx, sourceReq)

			if err != nil && !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
				telemetry.recordError(&localTelemetry)
				if outcome != types.QueueOutcomeDispatched {
					telemetry.recordReject(&localTelemetry)
				}
				continue
			}

			if outcome == types.QueueOutcomeDispatched {
				telemetry.recordDispatch(&localTelemetry)
				if m.limit > 0 {
					// Free capacity instantly to maintain active queue depth (W - L).
					detector.Release()
				}
			}
		}

		telemetry.commit(&localTelemetry)
	})

	b.StopTimer()
	elapsed := b.Elapsed().Seconds()
	telemetry.report(b, elapsed)

	cancel()                          // Graceful teardown to prevent async skewing of subsequent coordinates.
	time.Sleep(50 * time.Millisecond) // Wait for SUT background goroutines to terminate.
}

// BenchmarkFlowController_TopologyChurn evaluates dynamic provisioning overhead by continuously
// registering new flows, forcing the Registry to acquire write locks.
func BenchmarkFlowController_TopologyChurn(b *testing.B) {
	ctx := b.Context()

	cfg := &controller.Config{
		DefaultRequestTTL:               5 * time.Minute,
		ProcessorReconciliationInterval: 1 * time.Hour, // Effectively disabled
		ExpiryCleanupInterval:           1 * time.Hour, // Effectively disabled
		EnqueueChannelBufferSize:        100,
	}

	fc, detector := setupBenchmarkHarness(b, ctx, 4, 1, 100, nil, cfg)

	const numKeys = 50000
	preAllocatedReqs := make([]*benchRequest, numKeys)
	for i := range numKeys {
		preAllocatedReqs[i] = &benchRequest{
			key:      flowcontrol.FlowKey{ID: fmt.Sprintf("novel-flow-%d", i), Priority: 0},
			byteSize: 1024,
		}
	}

	var dispatchCount atomic.Int64
	var globalThreadID atomic.Uint64

	b.ResetTimer()
	b.ReportAllocs()
	b.SetParallelism(100)

	b.RunParallel(func(pb *testing.PB) {
		var localDisp int64

		// Multiply by a prime to guarantee threads start at different modulo offsets, avoiding lockstep
		// contention on the exact same Registry keys.
		localID := int(globalThreadID.Add(1)) * 9973

		for pb.Next() {
			localID++
			req := preAllocatedReqs[localID%numKeys]

			outcome, _ := fc.EnqueueAndWait(ctx, req)
			if outcome == types.QueueOutcomeDispatched {
				localDisp++
				detector.Release()
			}
		}
		dispatchCount.Add(localDisp)
	})

	b.StopTimer()
	elapsed := b.Elapsed().Seconds()
	if elapsed > 0 {
		b.ReportMetric(math.Round(float64(dispatchCount.Load())/elapsed), "d/s")
	}
}

// BenchmarkFlowController_MassCancellation evaluates the cleanup overhead of client abandonment by
// aggressively timing out requests under high load.
func BenchmarkFlowController_MassCancellation(b *testing.B) {
	ctx := b.Context()

	cfg := &controller.Config{
		DefaultRequestTTL:               5 * time.Minute,
		ProcessorReconciliationInterval: 1 * time.Hour,         // Effectively disabled
		ExpiryCleanupInterval:           10 * time.Millisecond, // Hyper-aggressive sweep for benchmark
		EnqueueChannelBufferSize:        100,
	}

	// Use the permanently saturated detector to guarantee all requests queue and definitively rot.
	fc, _ := setupBenchmarkHarness(b, ctx, 4, 1, 100, &alwaysSaturatedDetector{}, cfg)

	var timeoutCount atomic.Int64

	b.ResetTimer()
	b.ReportAllocs()
	b.SetParallelism(100)

	b.RunParallel(func(pb *testing.PB) {
		var localTimeout int64
		req := &benchRequest{
			key:      flowcontrol.FlowKey{ID: "zombie-flow", Priority: 0},
			byteSize: 1024,
		}

		for pb.Next() {
			reqCtx, reqCancel := context.WithTimeout(ctx, 50*time.Millisecond)
			_, err := fc.EnqueueAndWait(reqCtx, req)
			reqCancel()

			if errors.Is(err, context.DeadlineExceeded) ||
				errors.Is(err, types.ErrTTLExpired) ||
				errors.Is(err, context.Canceled) {
				localTimeout++
			}
		}
		timeoutCount.Add(localTimeout)
	})

	b.StopTimer()
	elapsed := b.Elapsed().Seconds()
	if elapsed > 0 {
		b.ReportMetric(math.Round(float64(timeoutCount.Load())/elapsed), "zombies/s")
	}
}
