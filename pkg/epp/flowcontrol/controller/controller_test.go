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

// Note on Time-Based Lifecycle Tests:
// Tests validating the controller's handling of request TTLs (e.g., OnReqCtxTimeout*) rely on real-time timers
// (context.WithDeadline). The injected testclock.FakeClock is used to control the timing of internal loops (like
// reconciliation), but it cannot manipulate the timers used by the standard context package. Therefore, these specific
// tests use time.Sleep or assertions on real-time durations.

package controller

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/utils/clock"
	testclock "k8s.io/utils/clock/testing"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/contracts/mocks"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/controller/internal"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework"
	frameworkmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/framework/mocks"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types"
	typesmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/flowcontrol/types/mocks"
)

// --- Test Harness & Fixtures ---

// withClock returns a test-only option to inject a clock.
// test-only
func withClock(c clock.WithTicker) flowControllerOption {
	return func(fc *FlowController) {
		fc.clock = c
	}
}

// withRegistryClient returns a test-only option to inject a mock or fake registry client.
// test-only
func withRegistryClient(client registryClient) flowControllerOption {
	return func(fc *FlowController) {
		fc.registry = client
	}
}

// withShardProcessorFactory returns a test-only option to inject a processor factory.
// test-only
func withShardProcessorFactory(factory shardProcessorFactory) flowControllerOption {
	return func(fc *FlowController) {
		fc.shardProcessorFactory = factory
	}
}

// testHarness holds the `FlowController` and its dependencies under test.
type testHarness struct {
	fc  *FlowController
	cfg Config
	// clock is the clock interface used by the controller.
	clock        clock.WithTicker
	mockRegistry *mockRegistryClient
	// mockClock provides access to FakeClock methods (Step, HasWaiters) if and only if the underlying clock is a
	// FakeClock.
	mockClock            *testclock.FakeClock
	mockProcessorFactory *mockShardProcessorFactory
}

// newUnitHarness creates a test environment with a mock processor factory, suitable for focused unit tests of the
// controller's logic. It starts the controller's run loop using the provided context for lifecycle management.
func newUnitHarness(t *testing.T, ctx context.Context, cfg Config, registry *mockRegistryClient) *testHarness {
	t.Helper()
	mockDetector := &mocks.MockSaturationDetector{}
	mockPodLocator := &mocks.MockPodLocator{}

	// Initialize the FakeClock with the current system time.
	// The controller implementation uses the injected clock to calculate the deadline timestamp,vbut uses the standard
	// context.WithDeadline (which relies on the system clock) to enforce it.
	// If the FakeClock's time is far from the system time, deadlines calculated based on the FakeClockvmight already be
	// expired according to the system clock, causing immediate TTL failures.
	mockClock := testclock.NewFakeClock(time.Now())

	mockProcessorFactory := &mockShardProcessorFactory{
		processors: make(map[string]*mockShardProcessor),
	}

	// Default the registry if nil, simplifying tests that don't focus on registry interaction.
	if registry == nil {
		registry = &mockRegistryClient{}
	}

	opts := []flowControllerOption{
		withRegistryClient(registry),
		withClock(mockClock),
		withShardProcessorFactory(mockProcessorFactory.new),
	}
	fc, err := NewFlowController(ctx, cfg, registry, mockDetector, mockPodLocator, logr.Discard(), opts...)
	require.NoError(t, err, "failed to create FlowController for unit test harness")

	h := &testHarness{
		fc:                   fc,
		cfg:                  cfg,
		clock:                mockClock,
		mockRegistry:         registry,
		mockClock:            mockClock,
		mockProcessorFactory: mockProcessorFactory,
	}
	return h
}

// newIntegrationHarness creates a test environment that uses real `ShardProcessor`s, suitable for integration tests
// validating the controller-processor interaction.
func newIntegrationHarness(t *testing.T, ctx context.Context, cfg Config, registry *mockRegistryClient) *testHarness {
	t.Helper()
	mockDetector := &mocks.MockSaturationDetector{}
	mockPodLocator := &mocks.MockPodLocator{}

	// Align FakeClock with system time. See explanation in newUnitHarness.
	mockClock := testclock.NewFakeClock(time.Now())
	if registry == nil {
		registry = &mockRegistryClient{}
	}

	opts := []flowControllerOption{
		withRegistryClient(registry),
		withClock(mockClock),
	}
	fc, err := NewFlowController(ctx, cfg, registry, mockDetector, mockPodLocator, logr.Discard(), opts...)
	require.NoError(t, err, "failed to create FlowController for integration test harness")

	h := &testHarness{
		fc:           fc,
		cfg:          cfg,
		clock:        mockClock,
		mockRegistry: registry,
		mockClock:    mockClock,
	}
	return h
}

// mockActiveFlowConnection is a local mock for the `contracts.ActiveFlowConnection` interface.
type mockActiveFlowConnection struct {
	contracts.ActiveFlowConnection
	ActiveShardsV []contracts.RegistryShard
}

func (m *mockActiveFlowConnection) ActiveShards() []contracts.RegistryShard {
	return m.ActiveShardsV
}

// mockRegistryClient is a mock for the private `registryClient` interface.
type mockRegistryClient struct {
	contracts.FlowRegistryObserver
	contracts.FlowRegistryDataPlane
	WithConnectionFunc func(key types.FlowKey, fn func(conn contracts.ActiveFlowConnection) error) error
	ShardStatsFunc     func() []contracts.ShardStats
}

func (m *mockRegistryClient) WithConnection(
	key types.FlowKey,
	fn func(conn contracts.ActiveFlowConnection) error,
) error {
	if m.WithConnectionFunc != nil {
		return m.WithConnectionFunc(key, fn)
	}
	return fn(&mockActiveFlowConnection{})
}

func (m *mockRegistryClient) ShardStats() []contracts.ShardStats {
	if m.ShardStatsFunc != nil {
		return m.ShardStatsFunc()
	}
	return nil
}

// mockShardProcessor is a mock for the internal `shardProcessor` interface.
type mockShardProcessor struct {
	SubmitFunc        func(item *internal.FlowItem) error
	SubmitOrBlockFunc func(ctx context.Context, item *internal.FlowItem) error
	// runCtx captures the context provided to the Run method for lifecycle assertions.
	runCtx   context.Context
	runCtxMu sync.RWMutex
	// runStarted is closed when the Run method is called, allowing tests to synchronize with worker startup.
	runStarted chan struct{}
}

func (m *mockShardProcessor) Submit(item *internal.FlowItem) error {
	if m.SubmitFunc != nil {
		return m.SubmitFunc(item)
	}
	return nil
}

func (m *mockShardProcessor) SubmitOrBlock(ctx context.Context, item *internal.FlowItem) error {
	if m.SubmitOrBlockFunc != nil {
		return m.SubmitOrBlockFunc(ctx, item)
	}
	return nil
}

func (m *mockShardProcessor) Run(ctx context.Context) {
	m.runCtxMu.Lock()
	m.runCtx = ctx
	m.runCtxMu.Unlock()
	if m.runStarted != nil {
		close(m.runStarted)
	}
	// Block until the context is cancelled, simulating a running worker.
	<-ctx.Done()
}

// Context returns the context captured during the Run method call.
func (m *mockShardProcessor) Context() context.Context {
	m.runCtxMu.RLock()
	defer m.runCtxMu.RUnlock()
	return m.runCtx
}

// mockShardProcessorFactory allows tests to inject specific `mockShardProcessor` instances.
type mockShardProcessorFactory struct {
	mu         sync.Mutex
	processors map[string]*mockShardProcessor
}

// new is the factory function conforming to the `shardProcessorFactory` signature.
func (f *mockShardProcessorFactory) new(
	_ context.Context, // The factory does not use the lifecycle context; it's passed to the processor's Run method later.
	shard contracts.RegistryShard,
	_ contracts.SaturationDetector,
	_ contracts.PodLocator,
	_ clock.WithTicker,
	_ time.Duration,
	_ int,
	_ logr.Logger,
) shardProcessor {
	f.mu.Lock()
	defer f.mu.Unlock()
	if proc, ok := f.processors[shard.ID()]; ok {
		return proc
	}
	// Return a default mock processor if one is not explicitly registered by the test.
	return &mockShardProcessor{}
}

// stubManagedQueue is a simple stub for the `contracts.ManagedQueue`  interface.
type stubManagedQueue struct {
	contracts.ManagedQueue
	byteSizeV uint64
}

func (s *stubManagedQueue) ByteSize() uint64 { return s.byteSizeV }

func (s *stubManagedQueue) FlowQueueAccessor() framework.FlowQueueAccessor {
	return &frameworkmocks.MockFlowQueueAccessor{ByteSizeV: s.byteSizeV}
}

// mockShardBuilder is a fixture to declaratively build mock `contracts.RegistryShard` for tests.
type mockShardBuilder struct {
	id       string
	byteSize uint64
}

func newMockShard(id string) *mockShardBuilder {
	return &mockShardBuilder{id: id}
}

func (b *mockShardBuilder) withByteSize(size uint64) *mockShardBuilder {
	b.byteSize = size
	return b
}

func (b *mockShardBuilder) build() contracts.RegistryShard {
	return &mocks.MockRegistryShard{
		IDFunc: func() string { return b.id },
		ManagedQueueFunc: func(_ types.FlowKey) (contracts.ManagedQueue, error) {
			return &stubManagedQueue{byteSizeV: b.byteSize}, nil
		},
	}
}

var defaultFlowKey = types.FlowKey{ID: "test-flow", Priority: 100}

func newTestRequest(key types.FlowKey) *typesmocks.MockFlowControlRequest {
	return &typesmocks.MockFlowControlRequest{
		FlowKeyV:  key,
		ByteSizeV: 100,
		IDV:       "req-" + key.ID,
	}
}

// --- Test Cases ---

// TestFlowController_EnqueueAndWait covers the primary API entry point, focusing on validation, distribution logic,
// retries, and the request lifecycle (including post-distribution cancellation/timeout).
func TestFlowController_EnqueueAndWait(t *testing.T) {
	t.Parallel()

	t.Run("Rejections", func(t *testing.T) {
		t.Parallel()

		t.Run("OnReqCtxExpiredBeforeDistribution", func(t *testing.T) {
			t.Parallel()
			// Test that if the request context provided to EnqueueAndWait is already expired, it returns immediately.
			h := newUnitHarness(t, t.Context(), Config{DefaultRequestTTL: 1 * time.Minute}, nil)

			// Configure registry to return a shard.
			shardA := newMockShard("shard-A").build()
			h.mockRegistry.WithConnectionFunc = func(_ types.FlowKey, fn func(_ contracts.ActiveFlowConnection) error) error {
				return fn(&mockActiveFlowConnection{ActiveShardsV: []contracts.RegistryShard{shardA}})
			}
			// Configure processor to block until context expiry.
			h.mockProcessorFactory.processors["shard-A"] = &mockShardProcessor{
				SubmitFunc: func(_ *internal.FlowItem) error { return internal.ErrProcessorBusy },
				SubmitOrBlockFunc: func(ctx context.Context, _ *internal.FlowItem) error {
					<-ctx.Done()              // Wait for the context to be done.
					return context.Cause(ctx) // Return the cause.
				},
			}

			req := newTestRequest(defaultFlowKey)
			// Use a context with a deadline in the past.
			reqCtx, cancel := context.WithDeadlineCause(
				context.Background(),
				h.clock.Now().Add(-1*time.Second),
				types.ErrTTLExpired)
			defer cancel()

			outcome, err := h.fc.EnqueueAndWait(reqCtx, req)
			require.Error(t, err, "EnqueueAndWait must fail if request context deadline is exceeded")
			assert.ErrorIs(t, err, types.ErrRejected, "error should wrap ErrRejected")
			assert.ErrorIs(t, err, types.ErrTTLExpired, "error should wrap types.ErrTTLExpired from the context cause")
			assert.Equal(t, types.QueueOutcomeRejectedOther, outcome, "outcome should be QueueOutcomeRejectedOther")
		})
		t.Run("OnControllerShutdown", func(t *testing.T) {
			t.Parallel()
			// Create a context specifically for the controller's lifecycle.
			ctx, cancel := context.WithCancel(t.Context())
			h := newUnitHarness(t, ctx, Config{}, nil)
			cancel() // Immediately stop the controller.

			// Wait for the controller's run loop and all workers (none in this case) to exit.
			// We need to wait because the shutdown process is asynchronous.
			h.fc.wg.Wait()

			req := newTestRequest(defaultFlowKey)
			// The request context is valid, but the controller itself is stopped.
			outcome, err := h.fc.EnqueueAndWait(context.Background(), req)
			require.Error(t, err, "EnqueueAndWait must reject requests if controller is not running")
			assert.ErrorIs(t, err, types.ErrRejected, "error should wrap ErrRejected")
			assert.ErrorIs(t, err, types.ErrFlowControllerNotRunning, "error should wrap ErrFlowControllerNotRunning")
			assert.Equal(t, types.QueueOutcomeRejectedOther, outcome,
				"outcome should be QueueOutcomeRejectedOther on shutdown")
		})

		t.Run("OnNoShardsAvailable", func(t *testing.T) {
			t.Parallel()
			// The default mockRegistryClient returns an empty list of ActiveShards.
			h := newUnitHarness(t, t.Context(), Config{}, nil)

			req := newTestRequest(defaultFlowKey)
			outcome, err := h.fc.EnqueueAndWait(context.Background(), req)
			require.Error(t, err, "EnqueueAndWait must reject requests if no shards are available")
			assert.ErrorIs(t, err, types.ErrRejected, "error should wrap ErrRejected")
			assert.Equal(t, types.QueueOutcomeRejectedCapacity, outcome,
				"outcome should be QueueOutcomeRejectedCapacity when no shards exist for the flow")
		})

		t.Run("OnRegistryConnectionError", func(t *testing.T) {
			t.Parallel()
			mockRegistry := &mockRegistryClient{}
			h := newUnitHarness(t, t.Context(), Config{}, mockRegistry)

			expectedErr := errors.New("simulated connection failure")
			// Configure the registry to fail when attempting to retrieve ActiveFlowConnection.
			mockRegistry.WithConnectionFunc = func(
				_ types.FlowKey,
				_ func(conn contracts.ActiveFlowConnection) error,
			) error {
				return expectedErr
			}

			req := newTestRequest(defaultFlowKey)
			outcome, err := h.fc.EnqueueAndWait(context.Background(), req)
			require.Error(t, err, "EnqueueAndWait must reject requests if registry connection fails")
			assert.ErrorIs(t, err, types.ErrRejected, "error should wrap ErrRejected")
			assert.ErrorIs(t, err, expectedErr, "error should wrap the underlying connection error")
			assert.Equal(t, types.QueueOutcomeRejectedOther, outcome,
				"outcome should be QueueOutcomeRejectedOther for transient registry errors")
		})

		t.Run("OnManagedQueueError", func(t *testing.T) {
			t.Parallel()
			mockRegistry := &mockRegistryClient{}
			h := newUnitHarness(t, t.Context(), Config{}, mockRegistry)

			// Create a faulty shard that successfully leases the flow but fails to return the
			// ManagedQueue. This shard should be considered as unavailable.
			faultyShard := &mocks.MockRegistryShard{
				IDFunc: func() string { return "faulty-shard" },
				ManagedQueueFunc: func(_ types.FlowKey) (contracts.ManagedQueue, error) {
					return nil, errors.New("invariant violation: queue retrieval failed")
				},
			}
			mockRegistry.WithConnectionFunc = func(
				_ types.FlowKey,
				fn func(conn contracts.ActiveFlowConnection) error,
			) error {
				return fn(&mockActiveFlowConnection{ActiveShardsV: []contracts.RegistryShard{faultyShard}})
			}

			req := newTestRequest(defaultFlowKey)
			outcome, err := h.fc.EnqueueAndWait(context.Background(), req)
			require.Error(t, err, "EnqueueAndWait must reject requests if no shards are available")
			assert.ErrorIs(t, err, types.ErrRejected, "error should wrap ErrRejected")
			assert.Equal(t, types.QueueOutcomeRejectedCapacity, outcome,
				"outcome should be QueueOutcomeRejectedCapacity when no shards exist for the flow")
		})
	})

	// Distribution tests validate the JSQ-Bytes algorithm, the two-phase submission strategy, and error handling during
	// the handoff, including time-based failures during blocking fallback.
	t.Run("Distribution", func(t *testing.T) {
		t.Parallel()

		// Define a long default TTL to prevent unexpected timeouts unless a test case explicitly sets a shorter one.
		const defaultTestTTL = 5 * time.Second

		testCases := []struct {
			name            string
			shards          []contracts.RegistryShard
			setupProcessors func(t *testing.T, h *testHarness)
			// requestTTL overrides the default TTL for time-sensitive tests.
			requestTTL      time.Duration
			expectedOutcome types.QueueOutcome
			expectErr       bool
			expectErrIs     error
		}{
			{
				name:   "SubmitSucceeds_NonBlocking_WithSingleActiveShard",
				shards: []contracts.RegistryShard{newMockShard("shard-A").build()},
				setupProcessors: func(t *testing.T, h *testHarness) {
					h.mockProcessorFactory.processors["shard-A"] = &mockShardProcessor{
						SubmitFunc: func(item *internal.FlowItem) error {
							// Simulate asynchronous processing and successful dispatch.
							go item.FinalizeWithOutcome(types.QueueOutcomeDispatched, nil)
							return nil
						},
					}
				},
				expectedOutcome: types.QueueOutcomeDispatched,
			},
			{
				name: "DistributesToLeastLoadedShard_WithMultipleActiveShards",
				shards: []contracts.RegistryShard{
					newMockShard("shard-A").withByteSize(1000).build(), // More loaded
					newMockShard("shard-B").withByteSize(100).build(),  // Least loaded
				},
				setupProcessors: func(t *testing.T, h *testHarness) {
					h.mockProcessorFactory.processors["shard-A"] = &mockShardProcessor{
						SubmitFunc: func(_ *internal.FlowItem) error {
							t.Error("Submit was called on the more loaded shard (shard-A); JSQ-Bytes algorithm failed")
							return internal.ErrProcessorBusy
						},
					}
					h.mockProcessorFactory.processors["shard-B"] = &mockShardProcessor{
						SubmitFunc: func(item *internal.FlowItem) error {
							item.SetHandle(&typesmocks.MockQueueItemHandle{})
							go item.FinalizeWithOutcome(types.QueueOutcomeDispatched, nil)
							return nil
						},
					}
				},
				expectedOutcome: types.QueueOutcomeDispatched,
			},
			{
				name: "SubmitSucceeds_AfterBlocking_WithAllProcessorsBusy",
				shards: []contracts.RegistryShard{
					newMockShard("shard-A").withByteSize(1000).build(),
					newMockShard("shard-B").withByteSize(100).build(),
				},
				setupProcessors: func(t *testing.T, h *testHarness) {
					// Both processors reject the initial non-blocking Submit.
					h.mockProcessorFactory.processors["shard-A"] = &mockShardProcessor{
						SubmitFunc: func(_ *internal.FlowItem) error { return internal.ErrProcessorBusy },
					}
					// Shard-B is the least loaded, so it should receive the blocking fallback (SubmitOrBlock).
					h.mockProcessorFactory.processors["shard-B"] = &mockShardProcessor{
						SubmitFunc: func(_ *internal.FlowItem) error { return internal.ErrProcessorBusy },
						SubmitOrBlockFunc: func(_ context.Context, item *internal.FlowItem) error {
							// The blocking call succeeds.
							go item.FinalizeWithOutcome(types.QueueOutcomeDispatched, nil)
							return nil
						},
					}
				},
				expectedOutcome: types.QueueOutcomeDispatched,
			},
			{
				// Validates the scenario where the request's TTL expires while the controller is blocked waiting for capacity.
				// NOTE: This relies on real time passing, as context.WithDeadline timers cannot be controlled by FakeClock.
				name:       "Rejects_AfterBlocking_WhenTTL_Expires",
				shards:     []contracts.RegistryShard{newMockShard("shard-A").build()},
				requestTTL: 50 * time.Millisecond, // Short TTL to keep the test fast.
				setupProcessors: func(t *testing.T, h *testHarness) {
					h.mockProcessorFactory.processors["shard-A"] = &mockShardProcessor{
						// Reject the non-blocking attempt.
						SubmitFunc: func(_ *internal.FlowItem) error { return internal.ErrProcessorBusy },
						// Block the fallback attempt until the context (carrying the TTL deadline) expires.
						SubmitOrBlockFunc: func(ctx context.Context, _ *internal.FlowItem) error {
							<-ctx.Done()
							return ctx.Err()
						},
					}
				},
				// No runActions needed; we rely on the real-time timer to expire.
				// When the blocking call fails due to context expiry, the outcome is RejectedOther.
				expectedOutcome: types.QueueOutcomeRejectedOther,
				expectErr:       true,
				// The error must reflect the specific cause of the context cancellation (ErrTTLExpired).
				expectErrIs: types.ErrTTLExpired,
			},
			{
				name:   "Rejects_OnProcessorShutdownDuringSubmit",
				shards: []contracts.RegistryShard{newMockShard("shard-A").build()},
				setupProcessors: func(t *testing.T, h *testHarness) {
					h.mockProcessorFactory.processors["shard-A"] = &mockShardProcessor{
						// Simulate the processor shutting down during the non-blocking handoff.
						SubmitFunc: func(_ *internal.FlowItem) error { return types.ErrFlowControllerNotRunning },
						SubmitOrBlockFunc: func(_ context.Context, _ *internal.FlowItem) error {
							return types.ErrFlowControllerNotRunning
						},
					}
				},
				expectedOutcome: types.QueueOutcomeRejectedOther,
				expectErr:       true,
				expectErrIs:     types.ErrFlowControllerNotRunning,
			},
			{
				name:   "Rejects_OnProcessorShutdownDuringSubmitOrBlock",
				shards: []contracts.RegistryShard{newMockShard("shard-A").build()},
				setupProcessors: func(t *testing.T, h *testHarness) {
					h.mockProcessorFactory.processors["shard-A"] = &mockShardProcessor{
						SubmitFunc: func(_ *internal.FlowItem) error { return internal.ErrProcessorBusy },
						// Simulate the processor shutting down during the blocking handoff.
						SubmitOrBlockFunc: func(_ context.Context, _ *internal.FlowItem) error {
							return types.ErrFlowControllerNotRunning
						},
					}
				},
				expectedOutcome: types.QueueOutcomeRejectedOther,
				expectErr:       true,
				expectErrIs:     types.ErrFlowControllerNotRunning,
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()

				// Arrange
				mockRegistry := &mockRegistryClient{}

				// Configure the harness with the appropriate TTL.
				harnessConfig := Config{DefaultRequestTTL: defaultTestTTL}
				if tc.requestTTL > 0 {
					harnessConfig.DefaultRequestTTL = tc.requestTTL
				}
				h := newUnitHarness(t, t.Context(), harnessConfig, mockRegistry)

				// Configure the registry to return the specified shards.
				mockRegistry.WithConnectionFunc = func(
					_ types.FlowKey,
					fn func(conn contracts.ActiveFlowConnection) error,
				) error {
					return fn(&mockActiveFlowConnection{ActiveShardsV: tc.shards})
				}
				tc.setupProcessors(t, h)

				// Act
				var outcome types.QueueOutcome
				var err error

				startTime := time.Now() // Capture real start time for duration checks.
				// Use a background context for the parent; the request lifecycle is governed by the config/derived context.
				outcome, err = h.fc.EnqueueAndWait(context.Background(), newTestRequest(defaultFlowKey))

				// Assert
				if tc.expectErr {
					require.Error(t, err, "expected an error during EnqueueAndWait but got nil")
					assert.ErrorIs(t, err, tc.expectErrIs, "error should wrap the expected underlying cause")
					// All failures during the distribution phase (capacity, timeout, shutdown) should result in a rejection.
					assert.ErrorIs(t, err, types.ErrRejected, "rejection errors must wrap types.ErrRejected")

					// Specific assertion for real-time TTL tests.
					if errors.Is(tc.expectErrIs, types.ErrTTLExpired) {
						duration := time.Since(startTime)
						// Ensure the test didn't return instantly. Use a tolerance for CI environments.
						// This validates that the real-time wait actually occurred.
						assert.GreaterOrEqual(t, duration, tc.requestTTL-30*time.Millisecond,
							"EnqueueAndWait returned faster than the TTL allows, indicating the timer did not function correctly")
					}

				} else {
					require.NoError(t, err, "expected no error during EnqueueAndWait but got: %v", err)
				}
				assert.Equal(t, tc.expectedOutcome, outcome, "outcome did not match expected value")
			})
		}
	})

	t.Run("Retry", func(t *testing.T) {
		t.Parallel()

		// This test specifically validates the behavior when the request context is cancelled externally while the
		// controller is blocked in the SubmitOrBlock phase.
		t.Run("Rejects_OnRequestContextCancelledWhileBlocking", func(t *testing.T) {
			t.Parallel()
			mockRegistry := &mockRegistryClient{
				WithConnectionFunc: func(
					_ types.FlowKey,
					fn func(conn contracts.ActiveFlowConnection,
					) error) error {
					return fn(&mockActiveFlowConnection{
						ActiveShardsV: []contracts.RegistryShard{newMockShard("shard-A").build()},
					})
				},
			}
			// Use a long TTL to ensure the failure is due to cancellation, not timeout.
			h := newUnitHarness(t, t.Context(), Config{DefaultRequestTTL: 10 * time.Second}, mockRegistry)
			h.mockProcessorFactory.processors["shard-A"] = &mockShardProcessor{
				// Reject non-blocking attempt.
				SubmitFunc: func(_ *internal.FlowItem) error { return internal.ErrProcessorBusy },
				// Block the fallback attempt until the context is cancelled.
				SubmitOrBlockFunc: func(ctx context.Context, _ *internal.FlowItem) error {
					<-ctx.Done()
					return ctx.Err()
				},
			}

			// Create a cancellable context for the request.
			reqCtx, cancelReq := context.WithCancel(context.Background())
			// Cancel the request shortly after starting the operation.
			// We use real time sleep here as we are testing external cancellation signals interacting with the context.
			go func() { time.Sleep(10 * time.Millisecond); cancelReq() }()

			outcome, err := h.fc.EnqueueAndWait(reqCtx, newTestRequest(defaultFlowKey))

			require.Error(t, err, "EnqueueAndWait must fail when context is cancelled during a blocking submit")
			assert.ErrorIs(t, err, types.ErrRejected, "error should wrap ErrRejected")
			assert.ErrorIs(t, err, context.Canceled, "error should wrap the underlying ctx.Err() (context.Canceled)")
			assert.Equal(t, types.QueueOutcomeRejectedOther, outcome,
				"outcome should be QueueOutcomeRejectedOther when cancelled during distribution")
		})

		// This test validates the retry mechanism when a processor reports that its shard is draining.
		t.Run("RetriesAndSucceeds_OnProcessorReportsShardDraining", func(t *testing.T) {
			t.Parallel()
			var callCount atomic.Int32
			mockRegistry := &mockRegistryClient{
				WithConnectionFunc: func(
					_ types.FlowKey,
					fn func(conn contracts.ActiveFlowConnection) error,
				) error {
					attempt := callCount.Add(1)
					shardA := newMockShard("shard-A").withByteSize(100).build()
					shardB := newMockShard("shard-B").withByteSize(1000).build()

					if attempt == 1 {
						// Attempt 1: Shard A is the least loaded and is selected.
						return fn(&mockActiveFlowConnection{ActiveShardsV: []contracts.RegistryShard{shardA, shardB}})
					}
					// Attempt 2 (Retry): Assume Shard A is now draining and removed from the active set by the registry.
					return fn(&mockActiveFlowConnection{ActiveShardsV: []contracts.RegistryShard{shardB}})
				},
			}
			// Use a long TTL to ensure retries don't time out.
			h := newUnitHarness(t, t.Context(), Config{DefaultRequestTTL: 10 * time.Second}, mockRegistry)

			// Configure Shard A's processor to reject the request due to draining.
			h.mockProcessorFactory.processors["shard-A"] = &mockShardProcessor{
				SubmitFunc: func(item *internal.FlowItem) error {
					// The processor accepts the item but then asynchronously finalizes it with ErrShardDraining.
					item.SetHandle(&typesmocks.MockQueueItemHandle{})
					go item.FinalizeWithOutcome(types.QueueOutcomeRejectedOther, contracts.ErrShardDraining)
					return nil
				},
			}
			// Configure Shard B's processor to successfully dispatch the request on the retry.
			h.mockProcessorFactory.processors["shard-B"] = &mockShardProcessor{
				SubmitFunc: func(item *internal.FlowItem) error {
					go item.FinalizeWithOutcome(types.QueueOutcomeDispatched, nil)
					return nil
				},
			}

			// Act
			outcome, err := h.fc.EnqueueAndWait(context.Background(), newTestRequest(defaultFlowKey))

			// Assert
			require.NoError(t, err, "EnqueueAndWait must succeed after retrying on a healthy shard")
			assert.Equal(t, types.QueueOutcomeDispatched, outcome, "outcome should be QueueOutcomeDispatched")
			assert.Equal(t, int32(2), callCount.Load(), "registry must be consulted for Active shards on each retry attempt")
		})
	})

	// Lifecycle covers the post-distribution phase, focusing on how the controller handles context cancellation and TTL
	// expiry while the request is buffered or queued by the processor (Asynchronous Finalization).
	t.Run("Lifecycle", func(t *testing.T) {
		t.Parallel()

		// Validates that the controller correctly initiates asynchronous finalization when the request context is cancelled
		// after ownership has been transferred to the processor.
		t.Run("OnReqCtxCancelledAfterDistribution", func(t *testing.T) {
			t.Parallel()
			// Use a long TTL to ensure the failure is due to cancellation.
			h := newUnitHarness(t, t.Context(), Config{DefaultRequestTTL: 10 * time.Second}, nil)

			shardA := newMockShard("shard-A").build()
			h.mockRegistry.WithConnectionFunc = func(_ types.FlowKey, fn func(_ contracts.ActiveFlowConnection) error) error {
				return fn(&mockActiveFlowConnection{ActiveShardsV: []contracts.RegistryShard{shardA}})
			}

			// Channel for synchronization.
			itemSubmitted := make(chan *internal.FlowItem, 1)

			// Configure the processor to accept the item but never finalize it, simulating a queued request.
			h.mockProcessorFactory.processors["shard-A"] = &mockShardProcessor{
				SubmitFunc: func(item *internal.FlowItem) error {
					item.SetHandle(&typesmocks.MockQueueItemHandle{})
					itemSubmitted <- item
					return nil
				},
			}

			reqCtx, cancelReq := context.WithCancel(context.Background())
			req := newTestRequest(defaultFlowKey)

			var outcome types.QueueOutcome
			var err error
			done := make(chan struct{})
			go func() {
				outcome, err = h.fc.EnqueueAndWait(reqCtx, req)
				close(done)
			}()

			// 1. Wait for the item to be successfully distributed.
			var item *internal.FlowItem
			select {
			case item = <-itemSubmitted:
				// Success. Ownership has transferred. EnqueueAndWait is now in the select loop.
			case <-time.After(1 * time.Second):
				t.Fatal("timed out waiting for item to be submitted to the processor")
			}

			// 2. Cancel the request context.
			cancelReq()

			// 3. Wait for EnqueueAndWait to return.
			select {
			case <-done:
				// Success. The controller detected the cancellation and unblocked the caller.
			case <-time.After(1 * time.Second):
				t.Fatal("timed out waiting for EnqueueAndWait to return after cancellation")
			}

			// 4. Assertions for EnqueueAndWait's return values.
			require.Error(t, err, "EnqueueAndWait should return an error when the request is cancelled post-distribution")
			// The outcome should be Evicted (as the handle was set).
			assert.ErrorIs(t, err, types.ErrEvicted, "error should wrap ErrEvicted")
			// The underlying cause must be propagated.
			assert.ErrorIs(t, err, types.ErrContextCancelled, "error should wrap ErrContextCancelled")
			assert.Equal(t, types.QueueOutcomeEvictedContextCancelled, outcome, "outcome should be EvictedContextCancelled")

			// 5. Assert that the FlowItem itself was indeed finalized by the controller.
			finalState := item.FinalState()
			require.NotNil(t, finalState, "Item should have been finalized asynchronously by the controller")
			assert.Equal(t, types.QueueOutcomeEvictedContextCancelled, finalState.Outcome,
				"Item's internal outcome must match the returned outcome")
		})

		// Validates the asynchronous finalization path due to TTL expiry.
		// Note: This relies on real time passing, as context.WithDeadline timers cannot be controlled by FakeClock.
		t.Run("OnReqCtxTimeoutAfterDistribution", func(t *testing.T) {
			t.Parallel()
			// Configure a short TTL to keep the test reasonably fast.
			const requestTTL = 50 * time.Millisecond
			h := newUnitHarness(t, t.Context(), Config{DefaultRequestTTL: requestTTL}, nil)

			shardA := newMockShard("shard-A").build()
			h.mockRegistry.WithConnectionFunc = func(_ types.FlowKey, fn func(_ contracts.ActiveFlowConnection) error) error {
				return fn(&mockActiveFlowConnection{ActiveShardsV: []contracts.RegistryShard{shardA}})
			}

			itemSubmitted := make(chan *internal.FlowItem, 1)

			// Configure the processor to accept the item but never finalize it.
			h.mockProcessorFactory.processors["shard-A"] = &mockShardProcessor{
				SubmitFunc: func(item *internal.FlowItem) error {
					item.SetHandle(&typesmocks.MockQueueItemHandle{})
					itemSubmitted <- item
					return nil
				},
			}

			req := newTestRequest(defaultFlowKey)
			// Use a context for the call itself that won't time out independently.
			enqueueCtx, enqueueCancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer enqueueCancel()

			var outcome types.QueueOutcome
			var err error
			done := make(chan struct{})

			startTime := time.Now() // Capture start time to validate duration.
			go func() {
				outcome, err = h.fc.EnqueueAndWait(enqueueCtx, req)
				close(done)
			}()

			// 1. Wait for the item to be submitted.
			var item *internal.FlowItem
			select {
			case item = <-itemSubmitted:
			case <-time.After(1 * time.Second):
				t.Fatal("timed out waiting for item to be submitted to the processor")
			}

			// 2.Wait for the TTL to expire (Real time). We do NOT call Step().
			// Wait for EnqueueAndWait to return due to the TTL expiry.
			select {
			case <-done:
				// Success. Now validate that enough time actually passed.
				duration := time.Since(startTime)
				assert.GreaterOrEqual(t, duration, requestTTL-30*time.Millisecond, // tolerance for CI environments
					"EnqueueAndWait returned faster than the TTL allows, indicating the timer did not function correctly")
			case <-time.After(1 * time.Second):
				t.Fatal("timed out waiting for EnqueueAndWait to return after TTL expiry")
			}

			// 4. Assertions for EnqueueAndWait's return values.
			require.Error(t, err, "EnqueueAndWait should return an error when TTL expires post-distribution")
			assert.ErrorIs(t, err, types.ErrEvicted, "error should wrap ErrEvicted")
			assert.ErrorIs(t, err, types.ErrTTLExpired, "error should wrap the underlying cause (types.ErrTTLExpired)")
			assert.Equal(t, types.QueueOutcomeEvictedTTL, outcome, "outcome should be EvictedTTL")

			// 5. Assert FlowItem final state.
			finalState := item.FinalState()
			require.NotNil(t, finalState, "Item should have been finalized asynchronously by the controller")
			assert.Equal(t, types.QueueOutcomeEvictedTTL, finalState.Outcome,
				"Item's internal outcome must match the returned outcome")
		})
	})
}

// TestFlowController_WorkerManagement covers the lifecycle of the shard processors (workers), including startup,
// reconciliation (garbage collection), and shutdown.
func TestFlowController_WorkerManagement(t *testing.T) {
	t.Parallel()

	// Reconciliation validates that the controller correctly identifies and shuts down workers whose shards no longer
	// exist in the registry.
	t.Run("Reconciliation", func(t *testing.T) {
		t.Parallel()

		// Setup: A registry that initially knows about "shard-A" and "stale-shard", but later only reports "shard-A".
		mockRegistry := &mockRegistryClient{
			ShardStatsFunc: func() []contracts.ShardStats {
				// The current state of the world according to the registry.
				return []contracts.ShardStats{{ID: "shard-A"}}
			}}
		h := newUnitHarness(t, t.Context(), Config{}, mockRegistry)

		// Pre-populate the controller with initial workers, simulating a previous state.
		initialShards := []string{"shard-A", "stale-shard"}
		for _, shardID := range initialShards {
			currentShardID := shardID
			// Initialize the processor mocks with the channel needed to synchronize startup.
			h.mockProcessorFactory.processors[currentShardID] = &mockShardProcessor{runStarted: make(chan struct{})}
			shard := &mocks.MockRegistryShard{IDFunc: func() string { return currentShardID }}
			// Start the worker using the internal mechanism.
			h.fc.getOrStartWorker(shard)
		}
		require.Len(t, h.mockProcessorFactory.processors, 2, "pre-condition: initial workers not set up correctly")

		// Wait for all worker goroutines to have started and captured their contexts.
		for id, p := range h.mockProcessorFactory.processors {
			proc := p
			select {
			case <-proc.runStarted:
				// Worker is running.
			case <-time.After(2 * time.Second):
				t.Fatalf("timed out waiting for worker %s to start", id)
			}
		}

		// Act: Manually trigger the reconciliation logic.
		h.fc.reconcileProcessors()

		t.Run("StaleWorkerIsCancelled", func(t *testing.T) {
			staleProc := h.mockProcessorFactory.processors["stale-shard"]
			require.NotNil(t, staleProc.Context(), "precondition: stale processor context should have been captured")
			// The context of the removed worker must be cancelled to signal shutdown.
			select {
			case <-staleProc.Context().Done():
				// Success: Context was cancelled.
			case <-time.After(100 * time.Millisecond):
				t.Error("context of the stale worker was not cancelled during reconciliation")
			}
		})

		t.Run("ActiveWorkerIsNotCancelled", func(t *testing.T) {
			activeProc := h.mockProcessorFactory.processors["shard-A"]
			require.NotNil(t, activeProc.Context(), "precondition: active processor context should have been captured")
			// The context of an active worker must remain open.
			select {
			case <-activeProc.Context().Done():
				t.Error("context of the active worker was incorrectly cancelled during reconciliation")
			default:
				// Success: Context is still active.
			}
		})

		t.Run("WorkerMapIsUpdated", func(t *testing.T) {
			// The stale worker must be removed from the controller's concurrent map.
			_, ok := h.fc.workers.Load("stale-shard")
			assert.False(t, ok, "stale worker must be deleted from the controller's map")
			_, ok = h.fc.workers.Load("shard-A")
			assert.True(t, ok, "active worker must remain in the controller's map")
		})
	})

	// Validates that the reconciliation loop runs periodically based on the configured interval.
	t.Run("Reconciliation_IsTriggeredByTicker", func(t *testing.T) {
		t.Parallel()
		const reconciliationInterval = 10 * time.Second
		mockRegistry := &mockRegistryClient{}

		// Count the number of times the reconciliation logic (which calls ShardStats) runs.
		var reconcileCount atomic.Int32
		mockRegistry.ShardStatsFunc = func() []contracts.ShardStats {
			reconcileCount.Add(1)
			return nil
		}

		h := newUnitHarness(t, t.Context(), Config{ProcessorReconciliationInterval: reconciliationInterval}, mockRegistry)
		// Ensure we are using the FakeClock specifically for this test, as we need Step/HasWaiters.
		require.NotNil(t, h.mockClock, "This test requires the harness to be using FakeClock")

		// Wait for the reconciliation loop to start and create the ticker.
		// This prevents a race where the clock is stepped before the ticker is registered with the FakeClock.
		require.Eventually(t, h.mockClock.HasWaiters, time.Second, 10*time.Millisecond,
			"reconciliation ticker was not created")

		// Advance the clock to trigger the first reconciliation.
		h.mockClock.Step(reconciliationInterval)

		assert.Eventually(t, func() bool {
			return reconcileCount.Load() == 1
		}, time.Second, 10*time.Millisecond, "reconciliation was not triggered by the first ticker event")

		// Advance the clock again to ensure it continues to fire.
		h.mockClock.Step(reconciliationInterval)
		assert.Eventually(t, func() bool {
			return reconcileCount.Load() == 2
		}, time.Second, 10*time.Millisecond, "reconciliation did not fire on the second ticker event")
	})

	// Validates the atomicity of worker creation and ensures resource cleanup for the loser of the race.
	t.Run("WorkerCreationRace", func(t *testing.T) {
		t.Parallel()

		// This test orchestrates a deterministic race condition.
		factoryEntered := make(chan *mockShardProcessor, 2)
		continueFactory := make(chan struct{})
		// Map to store the construction context for each processor instance, allowing us to verify cleanup.
		constructionContexts := sync.Map{}

		h := newUnitHarness(t, t.Context(), Config{}, nil)

		// Inject a custom factory to control the timing of worker creation.
		h.fc.shardProcessorFactory = func(
			ctx context.Context, // The context created by getOrStartWorker for the potential new processor.
			shard contracts.RegistryShard,
			_ contracts.SaturationDetector,
			_ contracts.PodLocator,
			_ clock.WithTicker,
			_ time.Duration,
			_ int,
			_ logr.Logger,
		) shardProcessor {
			// This function is called by getOrStartWorker before the LoadOrStore check.
			proc := &mockShardProcessor{runStarted: make(chan struct{})}
			constructionContexts.Store(proc, ctx) // Capture the construction context.

			// Signal entry and then block, allowing another goroutine to enter.
			factoryEntered <- proc
			<-continueFactory
			return proc
		}

		shard := newMockShard("race-shard").build()
		var wg sync.WaitGroup
		wg.Add(2)

		// Start two goroutines that will race to create the same worker.
		go func() {
			defer wg.Done()
			h.fc.getOrStartWorker(shard)
		}()
		go func() {
			defer wg.Done()
			h.fc.getOrStartWorker(shard)
		}()

		// 1. Wait for both goroutines to enter the factory and create their respective processor instances.
		proc1 := <-factoryEntered
		proc2 := <-factoryEntered

		// 2. Unblock both goroutines, allowing them to race to workers.LoadOrStore.
		close(continueFactory)
		wg.Wait()

		// 3. Identify the winner and the loser.
		actual, ok := h.fc.workers.Load("race-shard")
		require.True(t, ok, "a worker must have been successfully stored in the map")

		storedWorker := actual.(*managedWorker)
		winnerProc := storedWorker.processor.(*mockShardProcessor)

		var loserProc *mockShardProcessor
		if winnerProc == proc1 {
			loserProc = proc2
		} else {
			loserProc = proc1
		}

		// 4. Validate the state of the winning processor.
		// Wait for the Run method to be called on the winner (only the winner should start).
		select {
		case <-winnerProc.runStarted:
			// Success.
		case <-time.After(1 * time.Second):
			t.Fatal("timed out waiting for the winning worker's Run method to be called")
		}

		// The winning processor's context must remain active.
		require.NotNil(t, winnerProc.Context(), "winner's context should not be nil (Run was called)")
		select {
		case <-winnerProc.Context().Done():
			t.Error("context of the winning worker should not be cancelled")
		default:
			// Success
		}

		// 5. Validate the state of the losing processor and resource cleanup.
		// The losing processor's Run method must NOT be called.
		select {
		case <-loserProc.runStarted:
			t.Error("Run was incorrectly called on the losing worker")
		default:
			// Success
		}

		// Verify the context created for the loser during construction was cancelled by getOrStartWorker.
		loserCtxRaw, ok := constructionContexts.Load(loserProc)
		require.True(t, ok, "loser processor construction context should have been captured")
		loserCtx := loserCtxRaw.(context.Context)

		select {
		case <-loserCtx.Done():
			// Success: Context was cancelled, preventing resource leaks.
		case <-time.After(100 * time.Millisecond):
			t.Error("context of the losing worker was not cancelled, this will leak resources")
		}
	})
}

// Helper function to create a realistic mock registry environment for integration/concurrency tests.
func setupRegistryForConcurrency(t *testing.T, numShards int, flowKey types.FlowKey) *mockRegistryClient {
	t.Helper()
	mockRegistry := &mockRegistryClient{}
	shards := make([]contracts.RegistryShard, numShards)

	// Configure the shards and their dependencies required by the real ShardProcessor implementation.
	for i := range numShards {
		// Capture loop variables for closures.
		shardID := fmt.Sprintf("shard-%d", i)
		// Use high-fidelity mock queues (MockManagedQueue) that implement the necessary interfaces and synchronization.
		currentQueue := &mocks.MockManagedQueue{FlowKeyV: flowKey}

		shards[i] = &mocks.MockRegistryShard{
			IDFunc: func() string { return shardID },
			ManagedQueueFunc: func(_ types.FlowKey) (contracts.ManagedQueue, error) {
				return currentQueue, nil
			},
			// Configuration required for ShardProcessor initialization and dispatch logic.
			AllOrderedPriorityLevelsFunc: func() []int { return []int{flowKey.Priority} },
			PriorityBandAccessorFunc: func(priority int) (framework.PriorityBandAccessor, error) {
				if priority == flowKey.Priority {
					return &frameworkmocks.MockPriorityBandAccessor{
						PriorityV: priority,
						IterateQueuesFunc: func(f func(framework.FlowQueueAccessor) bool) {
							f(currentQueue.FlowQueueAccessor())
						},
					}, nil
				}
				return nil, fmt.Errorf("unexpected priority %d", priority)
			},
			// Configure dispatch policies (FIFO).
			IntraFlowDispatchPolicyFunc: func(_ types.FlowKey) (framework.IntraFlowDispatchPolicy, error) {
				return &frameworkmocks.MockIntraFlowDispatchPolicy{
					SelectItemFunc: func(qa framework.FlowQueueAccessor) (types.QueueItemAccessor, error) {
						return qa.PeekHead(), nil
					},
				}, nil
			},
			InterFlowDispatchPolicyFunc: func(_ int) (framework.InterFlowDispatchPolicy, error) {
				return &frameworkmocks.MockInterFlowDispatchPolicy{
					SelectQueueFunc: func(band framework.PriorityBandAccessor) (framework.FlowQueueAccessor, error) {
						return currentQueue.FlowQueueAccessor(), nil
					},
				}, nil
			},
			// Configure stats reporting based on the live state of the mock queues.
			StatsFunc: func() contracts.ShardStats {
				return contracts.ShardStats{
					ID:            shardID,
					TotalLen:      uint64(currentQueue.Len()),
					TotalByteSize: currentQueue.ByteSize(),
					PerPriorityBandStats: map[int]contracts.PriorityBandStats{
						flowKey.Priority: {
							Len:           uint64(currentQueue.Len()),
							ByteSize:      currentQueue.ByteSize(),
							CapacityBytes: 1e9, // Effectively unlimited capacity to ensure dispatch success.
						},
					},
				}
			},
		}
	}

	// Configure the registry connection.
	mockRegistry.WithConnectionFunc = func(_ types.FlowKey, fn func(conn contracts.ActiveFlowConnection) error) error {
		return fn(&mockActiveFlowConnection{ActiveShardsV: shards})
	}
	mockRegistry.ShardStatsFunc = func() []contracts.ShardStats {
		stats := make([]contracts.ShardStats, len(shards))
		for i, shard := range shards {
			stats[i] = shard.Stats()
		}
		return stats
	}
	return mockRegistry
}

// TestFlowController_Concurrency_Distribution performs an integration test under high contention, using real
// ShardProcessors.
// It validates the thread-safety of the distribution logic and the overall system throughput.
func TestFlowController_Concurrency_Distribution(t *testing.T) {
	const (
		numShards     = 4
		numGoroutines = 50
		numRequests   = 200
	)

	// Arrange
	mockRegistry := setupRegistryForConcurrency(t, numShards, defaultFlowKey)

	// Initialize the integration harness with real ShardProcessors.
	h := newIntegrationHarness(t, t.Context(), Config{
		// Use a generous buffer to focus the test on distribution logic rather than backpressure.
		EnqueueChannelBufferSize: numRequests,
		DefaultRequestTTL:        5 * time.Second,
		ExpiryCleanupInterval:    100 * time.Millisecond,
	}, mockRegistry)

	// Act: Hammer the controller concurrently.
	var wg sync.WaitGroup
	wg.Add(numGoroutines)
	outcomes := make(chan types.QueueOutcome, numRequests)

	for i := range numGoroutines {
		goroutineID := i
		go func() {
			defer wg.Done()
			for j := range numRequests / numGoroutines {
				req := newTestRequest(defaultFlowKey)
				req.IDV = fmt.Sprintf("req-distrib-%d-%d", goroutineID, j)

				// Use a reasonable timeout for the individual request context.
				reqCtx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
				defer cancel()

				ctx := logr.NewContext(reqCtx, logr.Discard())
				outcome, err := h.fc.EnqueueAndWait(ctx, req)
				if err != nil {
					// Use t.Errorf for concurrent tests to report failures without halting execution.
					t.Errorf("EnqueueAndWait failed unexpectedly under load: %v", err)
				}
				outcomes <- outcome
			}
		}()
	}

	// Wait for all requests to complete.
	wg.Wait()
	close(outcomes)

	// Assert: All requests should be successfully dispatched.
	successCount := 0
	for outcome := range outcomes {
		if outcome == types.QueueOutcomeDispatched {
			successCount++
		}
	}
	require.Equal(t, numRequests, successCount,
		"all concurrent requests must be dispatched successfully without errors or data races")
}

// TestFlowController_Concurrency_Backpressure specifically targets the blocking submission path (SubmitOrBlock) by
// configuring the processors with zero buffer capacity.
func TestFlowController_Concurrency_Backpressure(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping concurrency integration test in short mode.")
	}
	t.Parallel()

	const (
		numShards     = 2
		numGoroutines = 20
		// Fewer requests than the distribution test, as the blocking path is inherently slower.
		numRequests = 40
	)

	// Arrange: Set up the registry environment.
	mockRegistry := setupRegistryForConcurrency(t, numShards, defaultFlowKey)

	// Use the integration harness with a configuration designed to induce backpressure.
	h := newIntegrationHarness(t, t.Context(), Config{
		// Zero buffer forces immediate use of SubmitOrBlock if the processor loop is busy.
		EnqueueChannelBufferSize: 0,
		// Generous TTL to ensure timeouts are not the cause of failure.
		DefaultRequestTTL:     10 * time.Second,
		ExpiryCleanupInterval: 100 * time.Millisecond,
	}, mockRegistry)

	// Act: Concurrently submit requests.
	var wg sync.WaitGroup
	wg.Add(numGoroutines)
	outcomes := make(chan types.QueueOutcome, numRequests)

	for i := range numGoroutines {
		goroutineID := i
		go func() {
			defer wg.Done()
			for j := range numRequests / numGoroutines {
				req := newTestRequest(defaultFlowKey)
				req.IDV = fmt.Sprintf("req-backpressure-%d-%d", goroutineID, j)

				// Use a reasonable timeout for the individual request context to ensure the test finishes promptly if a
				// deadlock occurs.
				reqCtx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
				defer cancel()

				outcome, err := h.fc.EnqueueAndWait(logr.NewContext(reqCtx, logr.Discard()), req)
				if err != nil {
					t.Errorf("EnqueueAndWait failed unexpectedly under backpressure for request %s: %v", req.ID(), err)
				}
				outcomes <- outcome
			}
		}()
	}
	wg.Wait()
	close(outcomes)

	// Assert: Verify successful dispatch despite high contention and zero buffer.
	successCount := 0
	for outcome := range outcomes {
		if outcome == types.QueueOutcomeDispatched {
			successCount++
		}
	}
	require.Equal(t, numRequests, successCount,
		"all concurrent requests should be dispatched successfully even under high contention and zero buffer capacity")
}
