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

// Package benchmark implements a synchronous steady-state pipeline for load-testing the Flow
// Control layer.
//
// Benchmarking Flow Control is challenging. Data plane routing is fast, while downstream inference
// is slow. Simulating downstream latency with thread sleeps skews CPU profiles by parking
// goroutines, which measures Go scheduler overhead rather than computational throughput.
//
// This harness avoids sleeps by using a synchronous pipeline:
//
//  1. Intentional Backpressure (W > L): Ingress Concurrency (W) is driven higher than the Capacity
//     Limit (L), forcing requests to queue.
//  2. Strict Capacity Checking: A mock SaturationDetector atomically grants capacity only when
//     evaluated, preventing race conditions where dispatch outruns clients.
//  3. Immediate Draining: When a client unblocks, it immediately frees its capacity slot,
//     triggering the next dispatch and keeping the system continuously active.
//  4. Consistent Queue Depth: Maintaining a fixed, deep queue forces continuous evaluation of
//     fairness and ordering policies at limits, isolating CPU performance.
//
// # Interpreting Metrics in b.RunParallel
//
// In a highly concurrent queuing system, standard Go benchmark metrics require careful
// interpretation:
//
//  1. ns/op (system-wide amortized time): Because this uses b.RunParallel, ns/op represents inverse
//     throughput, not latency. If the system processes 1,000,000 requests in 1 second, it reports
//     1,000,000 ns/op. (The d/s custom metric converts this to RPS automatically).
//  2. ops (the definition of an operation): One "op" is the complete lifecycle of a single
//     simulated request: Ingress, queuing, policy evaluation, and egress.
//  3. allocs/op and B/op (GC Pressure): High allocations per request mean the Go Garbage Collector
//     will thrash under load, causing latency jitter.
//  4. Saturated Coordinates (W > L): When Concurrency (W) exceeds Capacity (L), EnqueueAndWait
//     blocks. Because we immediately release capacity upon dispatch, the duration of an "op" is
//     strictly governed by the Flow Control layer's CPU overhead.
//
// # Custom Metrics Reported
//
//   - d/s:                  (Dispatches/sec) The primary throughput metric.
//   - r/s:                  (Rejects/sec) Rate of requests rejected due to capacity or timeouts.
//   - errors:               Total unexpected runtime errors encountered.
//   - zombies/s:            Rate of requests hitting context deadlines/TTLs.
//   - burst_dispatches/sec: Drain rate when capacity is instantaneously freed.
package benchmark

import (
	"context"
	"fmt"
	"math"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts/mocks"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/controller"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/fairness"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/plugins/ordering"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/registry"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	testutils "sigs.k8s.io/gateway-api-inference-extension/test/utils"
)

func init() {
	// Silence verbose logging during aggressive scaling benchmarks to prevent I/O blocking.
	log.SetLogger(logr.Discard())
}

// egressConcurrencyLimit (L) defines the maximum capacity of the simulated pool.
type egressConcurrencyLimit int64

// shardCount (S) dictates the data parallelism of the Flow Control layer.
type shardCount int

// priorityLevels (P) dictates the number of priority bands.
type priorityLevels int

// flowCount (F) dictates the number of active flows.
// Relative to queue depth (W - L), sweeping F shifts load between a few deep queues (stressing
// Ordering policies) and many shallow queues (stressing Fairness policies).
type flowCount int

// ingressConcurrency (W) dictates the volume of simultaneous incoming requests.
type ingressConcurrency int

// benchMatrix defines a single coordinate in the performance hypercube.
type benchMatrix struct {
	limit       egressConcurrencyLimit
	shards      shardCount
	priorities  priorityLevels
	flows       flowCount
	concurrency ingressConcurrency
}

// name returns a human-readable string representation of the matrix coordinate.
func (m benchMatrix) name() string {
	return fmt.Sprintf("L=%03d/S=%03d/P=%03d/F=%06d/W=%05d",
		m.limit, m.shards, m.priorities, m.flows, m.concurrency)
}

// testDetector exposes an API to manually release downstream capacity during a test run.
type testDetector interface {
	flowcontrol.SaturationDetector
	Release()
}

// benchDetector models target saturation based strictly on active request counts.
type benchDetector struct {
	plugin.Plugin
	concurrencyLimit atomic.Int64
	// _ prevents false sharing between atomic counters on multicore CPU cache lines.
	_        [56]byte
	inFlight atomic.Int64
}

// Release frees one unit of capacity.
func (d *benchDetector) Release() {
	if d.concurrencyLimit.Load() > 0 {
		d.inFlight.Add(-1)
	}
}

// Saturation reserves a concurrency slot atomically to guarantee the queues remain strictly at the
// target depth.
func (d *benchDetector) Saturation(ctx context.Context, candidates []metrics.PodMetrics) float64 {
	limit := d.concurrencyLimit.Load()
	if limit <= 0 {
		return 0.0 // Free-flow
	}

	// Optimistic increment to reserve a slot
	if d.inFlight.Add(1) <= limit {
		return 0.99 // Return < 1.0 so the dispatcher proceeds.
	}

	// Capacity exceeded; rollback the optimistic increment.
	d.inFlight.Add(-1)
	return 1.0 // Saturated - forces the Flow Control layer to hold the item.
}

// alwaysSaturatedDetector simulates a permanently saturated downstream pool.
// It is strictly used to evaluate garbage collection pathways and client abandonment scenarios.
type alwaysSaturatedDetector struct {
	plugin.Plugin
}

// Release is a no-op for the permanently saturated mock.
func (d *alwaysSaturatedDetector) Release() {}

// Saturation permanently returns 1.0 (100% saturated) to ensure requests queue.
func (d *alwaysSaturatedDetector) Saturation(ctx context.Context, candidates []metrics.PodMetrics) float64 {
	return 1.0
}

// benchRequest models an inbound inference request with realistic payload entropy.
type benchRequest struct {
	key      flowcontrol.FlowKey
	byteSize uint64
}

// --- stubs required by FlowControlRequest interface ---
func (r *benchRequest) FlowKey() flowcontrol.FlowKey             { return r.key }
func (r *benchRequest) ByteSize() uint64                         { return r.byteSize }
func (r *benchRequest) InitialEffectiveTTL() time.Duration       { return 5 * time.Minute }
func (r *benchRequest) ID() string                               { return "bench-req" }
func (r *benchRequest) GetMetadata() map[string]any              { return nil }
func (r *benchRequest) InferencePoolName() string                { return "bench-pool" }
func (r *benchRequest) ModelName() string                        { return "bench-model" }
func (r *benchRequest) TargetModelName() string                  { return "bench-target" }
func (r *benchRequest) InferenceRequest() *scheduling.LLMRequest { return nil }
func (r *benchRequest) ReceivedTimestamp() time.Time             { return time.Now() }

// setupRegistry provisions the concrete FlowRegistry.
func setupRegistry(
	b *testing.B,
	ctx context.Context,
	handle plugin.Handle,
	s shardCount,
	p priorityLevels,
) contracts.FlowRegistry {
	b.Helper()

	cfgOpts := []registry.ConfigOption{
		registry.WithInitialShardCount(int(s)),
		registry.WithMaxBytes(0), // Capacity restricted strictly via concurrency (L).
	}

	for i := 0; i < int(p); i++ {
		band, err := registry.NewPriorityBandConfig(
			handle, i, fmt.Sprintf("band-%d", i),
			registry.WithBandMaxBytes(10_000_000_000), // Prevent capacity-based rejections.
		)
		if err != nil {
			b.Fatalf("Failed to init priority band %d: %v", i, err)
		}
		cfgOpts = append(cfgOpts, registry.WithPriorityBand(band))
	}

	regCfg, err := registry.NewConfig(handle, cfgOpts...)
	if err != nil {
		b.Fatalf("Failed to create registry config: %v", err)
	}

	reg, err := registry.NewFlowRegistry(regCfg, logr.Discard())
	if err != nil {
		b.Fatalf("Failed to initialize concrete registry: %v", err)
	}

	go reg.Run(ctx)
	return reg
}

// setupBenchmarkHarness creates the standard SUT environment.
func setupBenchmarkHarness(
	b *testing.B,
	ctx context.Context,
	s shardCount,
	p priorityLevels,
	limit egressConcurrencyLimit,
	customDetector testDetector,
	customCfg *controller.Config,
) (*controller.FlowController, testDetector) {
	b.Helper()
	handle := testutils.NewTestHandle(ctx)

	fPolicy, err := fairness.GlobalStrictFairnessPolicyFactory(registry.DefaultFairnessPolicyRef, nil, handle)
	if err != nil {
		b.Fatalf("Failed to create fairness policy: %v", err)
	}
	handle.AddPlugin(registry.DefaultFairnessPolicyRef, fPolicy)

	oPolicy, err := ordering.FCFSOrderingPolicyFactory(registry.DefaultOrderingPolicyRef, nil, handle)
	if err != nil {
		b.Fatalf("Failed to create ordering policy: %v", err)
	}
	handle.AddPlugin(registry.DefaultOrderingPolicyRef, oPolicy)

	reg := setupRegistry(b, ctx, handle, s, p)

	detector := customDetector
	if detector == nil {
		defaultDetector := &benchDetector{}
		defaultDetector.concurrencyLimit.Store(int64(limit))
		detector = defaultDetector
	}

	cfg := customCfg
	if cfg == nil {
		// Buffer size dynamically scales down with parallel shards to isolate queue sorting overhead.
		bufferSize := max(2000/int(s), 10)
		cfg = &controller.Config{
			DefaultRequestTTL:               5 * time.Minute,
			ProcessorReconciliationInterval: 1 * time.Hour, // Effectively disabled
			ExpiryCleanupInterval:           1 * time.Hour, // Effectively disabled
			EnqueueChannelBufferSize:        bufferSize,
		}
	}

	fc, err := controller.NewFlowController(ctx, "benchmark", cfg, reg, detector, &mocks.MockPodLocator{})
	if err != nil {
		b.Fatalf("Failed to init FlowController: %v", err)
	}

	return fc, detector
}

// benchmarkTelemetry aggregates benchmark statistics lock-free.
// Threads mutate local structs and commit totals once at the end.
type benchmarkTelemetry struct {
	errorCount    atomic.Int64
	dispatchCount atomic.Int64
	rejectCount   atomic.Int64
}

// newBenchmarkTelemetry provisions the global telemetry aggregator.
func newBenchmarkTelemetry() *benchmarkTelemetry {
	return &benchmarkTelemetry{}
}

// threadTelemetry is a thread-local accumulator for benchmark statistics.
type threadTelemetry struct {
	errs int64
	disp int64
	rej  int64
}

// recordDispatch logs a successful dispatch for the thread.
func (t *benchmarkTelemetry) recordDispatch(local *threadTelemetry) {
	local.disp++
}

// recordError tracks system evaluation errors.
func (t *benchmarkTelemetry) recordError(local *threadTelemetry) {
	local.errs++
}

// recordReject logs explicit QueueOutcomeRejected events.
func (t *benchmarkTelemetry) recordReject(local *threadTelemetry) {
	local.rej++
}

// commit adds thread-local statistics to the global atomic counts.
func (t *benchmarkTelemetry) commit(local *threadTelemetry) {
	if local.errs > 0 {
		t.errorCount.Add(local.errs)
	}
	if local.disp > 0 {
		t.dispatchCount.Add(local.disp)
	}
	if local.rej > 0 {
		t.rejectCount.Add(local.rej)
	}
}

// report aggregates the committed globals and issues standard b.ReportMetric calls into the Go
// benchmarking framework.
func (t *benchmarkTelemetry) report(b *testing.B, elapsed float64) {
	if elapsed <= 0 {
		return
	}

	b.ReportMetric(math.Round(float64(t.dispatchCount.Load())/elapsed), "d/s")
	if rejects := t.rejectCount.Load(); rejects > 0 {
		b.ReportMetric(math.Round(float64(rejects)/elapsed), "r/s")
	}
	if errs := t.errorCount.Load(); errs > 0 {
		b.ReportMetric(float64(errs), "errors")
	}
}
