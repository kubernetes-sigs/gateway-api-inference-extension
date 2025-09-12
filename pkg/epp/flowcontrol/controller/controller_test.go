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
	fc                   *FlowController
	cfg                  Config
	mockRegistry         *mockRegistryClient
	mockDetector         *mocks.MockSaturationDetector
	mockClock            *testclock.FakeClock
	mockProcessorFactory *mockShardProcessorFactory
}

// newUnitHarness creates a test environment with a mock processor factory, suitable for focused unit tests of the
// controller's logic. It starts the controller's run loop and returns a cancel function to stop it.
func newUnitHarness(t *testing.T, ctx context.Context, cfg Config, registry *mockRegistryClient) *testHarness {
	t.Helper()
	mockDetector := &mocks.MockSaturationDetector{}
	mockClock := testclock.NewFakeClock(time.Now())
	mockProcessorFactory := &mockShardProcessorFactory{
		processors: make(map[string]*mockShardProcessor),
	}
	opts := []flowControllerOption{
		withRegistryClient(registry),
		withClock(mockClock),
		withShardProcessorFactory(mockProcessorFactory.new),
	}
	fc, err := NewFlowController(ctx, cfg, registry, mockDetector, logr.Discard(), opts...)
	require.NoError(t, err, "failed to create FlowController for unit test harness")

	h := &testHarness{
		fc:                   fc,
		cfg:                  cfg,
		mockRegistry:         registry,
		mockDetector:         mockDetector,
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
	mockClock := testclock.NewFakeClock(time.Now())
	opts := []flowControllerOption{
		withRegistryClient(registry),
		withClock(mockClock),
	}
	fc, err := NewFlowController(ctx, cfg, registry, mockDetector, logr.Discard(), opts...)
	require.NoError(t, err, "failed to create FlowController for integration test harness")

	h := &testHarness{
		fc:           fc,
		cfg:          cfg,
		mockRegistry: registry,
		mockDetector: mockDetector,
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
	runCtx            context.Context
	runCtxMu          sync.RWMutex
	runStarted        chan struct{}
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
	<-ctx.Done()
}

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

func (f *mockShardProcessorFactory) new(
	shard contracts.RegistryShard,
	_ internal.BandFilter,
	_ clock.Clock,
	_ time.Duration,
	_ int,
	_ logr.Logger,
) shardProcessor {
	f.mu.Lock()
	defer f.mu.Unlock()
	if proc, ok := f.processors[shard.ID()]; ok {
		return proc
	}
	// Return a default mock processor if one is not registered.
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

func newTestRequest(ctx context.Context, key types.FlowKey) *typesmocks.MockFlowControlRequest {
	return &typesmocks.MockFlowControlRequest{
		Ctx:       ctx,
		FlowKeyV:  key,
		ByteSizeV: 100,
		IDV:       "req-" + key.ID,
	}
}

// --- Test Cases ---

func TestNewFlowController(t *testing.T) {
	t.Parallel()

	t.Run("ErrorOnInvalidConfig", func(t *testing.T) {
		t.Parallel()
		invalidCfg := Config{ProcessorReconciliationInterval: -1 * time.Second}
		_, err := NewFlowController(
			context.Background(),
			invalidCfg,
			&mockRegistryClient{},
			&mocks.MockSaturationDetector{},
			logr.Discard(),
		)
		require.Error(t, err, "NewFlowController must return an error for invalid configuration")
	})
}

func TestFlowController_EnqueueAndWait(t *testing.T) {
	t.Parallel()

	t.Run("Rejections", func(t *testing.T) {
		t.Parallel()

		t.Run("OnNilRequest", func(t *testing.T) {
			t.Parallel()
			h := newUnitHarness(t, t.Context(), Config{}, &mockRegistryClient{})

			outcome, err := h.fc.EnqueueAndWait(nil)
			require.Error(t, err, "EnqueueAndWait must reject a nil request")
			assert.Equal(t, "request cannot be nil", err.Error())
			assert.Equal(t, types.QueueOutcomeRejectedOther, outcome, "outcome should be QueueOutcomeRejectedOther")
		})

		t.Run("OnControllerShutdown", func(t *testing.T) {
			t.Parallel()
			ctx, cancel := context.WithCancel(t.Context())
			h := newUnitHarness(t, ctx, Config{}, &mockRegistryClient{})
			cancel() // Immediately stop the controller.

			req := newTestRequest(context.Background(), defaultFlowKey)
			outcome, err := h.fc.EnqueueAndWait(req)
			require.Error(t, err, "EnqueueAndWait must reject requests if controller is not running")
			assert.ErrorIs(t, err, types.ErrRejected, "error should wrap ErrRejected")
			assert.ErrorIs(t, err, types.ErrFlowControllerNotRunning, "error should wrap ErrFlowControllerNotRunning")
			assert.Equal(t, types.QueueOutcomeRejectedOther, outcome, "outcome should be QueueOutcomeRejectedOther")
		})

		t.Run("OnNoShardsAvailable", func(t *testing.T) {
			t.Parallel()
			h := newUnitHarness(t, t.Context(), Config{}, &mockRegistryClient{})

			req := newTestRequest(context.Background(), defaultFlowKey)
			outcome, err := h.fc.EnqueueAndWait(req)
			require.Error(t, err, "EnqueueAndWait must reject requests if no shards are available")
			assert.ErrorIs(t, err, types.ErrRejected, "error should wrap ErrRejected")
			assert.Equal(t, types.QueueOutcomeRejectedCapacity, outcome, "outcome should be QueueOutcomeRejectedCapacity")
		})

		t.Run("OnRegistryConnectionError", func(t *testing.T) {
			t.Parallel()
			mockRegistry := &mockRegistryClient{}
			h := newUnitHarness(t, t.Context(), Config{}, mockRegistry)

			expectedErr := errors.New("connection failed")
			mockRegistry.WithConnectionFunc = func(
				_ types.FlowKey,
				_ func(conn contracts.ActiveFlowConnection) error,
			) error {
				return expectedErr
			}

			req := newTestRequest(context.Background(), defaultFlowKey)
			outcome, err := h.fc.EnqueueAndWait(req)
			require.Error(t, err, "EnqueueAndWait must reject requests if registry connection fails")
			assert.ErrorIs(t, err, types.ErrRejected, "error should wrap ErrRejected")
			assert.ErrorIs(t, err, expectedErr, "error should wrap the underlying connection error")
			assert.Equal(t, types.QueueOutcomeRejectedOther, outcome, "outcome should be QueueOutcomeRejectedOther")
		})

		t.Run("PanicsOnManagedQueueError", func(t *testing.T) {
			t.Parallel()
			mockRegistry := &mockRegistryClient{}
			h := newUnitHarness(t, t.Context(), Config{}, mockRegistry)

			faultyShard := &mocks.MockRegistryShard{
				IDFunc: func() string { return "faulty-shard" },
				ManagedQueueFunc: func(_ types.FlowKey) (contracts.ManagedQueue, error) {
					return nil, errors.New("queue retrieval failed")
				},
			}
			mockRegistry.WithConnectionFunc = func(
				_ types.FlowKey,
				fn func(conn contracts.ActiveFlowConnection) error,
			) error {
				return fn(&mockActiveFlowConnection{ActiveShardsV: []contracts.RegistryShard{faultyShard}})
			}

			req := newTestRequest(context.Background(), defaultFlowKey)
			assert.Panics(t, func() {
				_, _ = h.fc.EnqueueAndWait(req)
			}, "EnqueueAndWait did not panic as expected on a ManagedQueue error")
		})
	})

	t.Run("Distribution", func(t *testing.T) {
		t.Parallel()

		testCases := []struct {
			name            string
			shards          []contracts.RegistryShard
			setupProcessors func(t *testing.T, h *testHarness)
			expectedOutcome types.QueueOutcome
			expectErr       bool
		}{
			{
				name:   "SubmitSucceeds_NonBlocking_WithSingleActiveShard",
				shards: []contracts.RegistryShard{newMockShard("shard-A").build()},
				setupProcessors: func(t *testing.T, h *testHarness) {
					h.mockProcessorFactory.processors["shard-A"] = &mockShardProcessor{
						SubmitFunc: func(item *internal.FlowItem) error {
							go item.Finalize(types.QueueOutcomeDispatched, nil)
							return nil
						},
					}
				},
				expectedOutcome: types.QueueOutcomeDispatched,
			},
			{
				name: "DistributesToLeastLoadedShard_WithMultipleActiveShards",
				shards: []contracts.RegistryShard{
					newMockShard("shard-A").withByteSize(1000).build(),
					newMockShard("shard-B").withByteSize(100).build(),
				},
				setupProcessors: func(t *testing.T, h *testHarness) {
					h.mockProcessorFactory.processors["shard-A"] = &mockShardProcessor{
						SubmitFunc: func(_ *internal.FlowItem) error {
							t.Error("Submit was called on the more loaded shard (shard-A), which is incorrect")
							return internal.ErrProcessorBusy
						},
					}
					h.mockProcessorFactory.processors["shard-B"] = &mockShardProcessor{
						SubmitFunc: func(item *internal.FlowItem) error {
							go item.Finalize(types.QueueOutcomeDispatched, nil)
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
					h.mockProcessorFactory.processors["shard-A"] = &mockShardProcessor{
						SubmitFunc: func(_ *internal.FlowItem) error { return internal.ErrProcessorBusy },
					}
					h.mockProcessorFactory.processors["shard-B"] = &mockShardProcessor{
						SubmitFunc: func(_ *internal.FlowItem) error { return internal.ErrProcessorBusy },
						SubmitOrBlockFunc: func(_ context.Context, item *internal.FlowItem) error {
							go item.Finalize(types.QueueOutcomeDispatched, nil)
							return nil
						},
					}
				},
				expectedOutcome: types.QueueOutcomeDispatched,
			},
			{
				name:   "Rejects_AfterBlocking_WithAllProcessorsRemainingBusy",
				shards: []contracts.RegistryShard{newMockShard("shard-A").build()},
				setupProcessors: func(t *testing.T, h *testHarness) {
					h.mockProcessorFactory.processors["shard-A"] = &mockShardProcessor{
						SubmitFunc:        func(_ *internal.FlowItem) error { return internal.ErrProcessorBusy },
						SubmitOrBlockFunc: func(_ context.Context, _ *internal.FlowItem) error { return context.DeadlineExceeded },
					}
				},
				expectedOutcome: types.QueueOutcomeRejectedCapacity,
				expectErr:       true,
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				mockRegistry := &mockRegistryClient{}
				h := newUnitHarness(t, t.Context(), Config{}, mockRegistry)
				mockRegistry.WithConnectionFunc = func(
					_ types.FlowKey,
					fn func(conn contracts.ActiveFlowConnection) error,
				) error {
					return fn(&mockActiveFlowConnection{ActiveShardsV: tc.shards})
				}
				tc.setupProcessors(t, h)
				outcome, err := h.fc.EnqueueAndWait(newTestRequest(context.Background(), defaultFlowKey))
				if tc.expectErr {
					require.Error(t, err, "expected an error but got nil")
				} else {
					require.NoError(t, err, "expected no error but got: %v", err)
				}
				assert.Equal(t, tc.expectedOutcome, outcome, "outcome did not match expected value")
			})
		}
	})

	t.Run("Retry", func(t *testing.T) {
		t.Parallel()

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
			h := newUnitHarness(t, t.Context(), Config{}, mockRegistry)
			h.mockProcessorFactory.processors["shard-A"] = &mockShardProcessor{
				SubmitFunc:        func(_ *internal.FlowItem) error { return internal.ErrProcessorBusy },
				SubmitOrBlockFunc: func(ctx context.Context, _ *internal.FlowItem) error { <-ctx.Done(); return ctx.Err() },
			}
			reqCtx, cancelReq := context.WithCancel(context.Background())
			go func() { time.Sleep(50 * time.Millisecond); cancelReq() }()

			outcome, err := h.fc.EnqueueAndWait(newTestRequest(reqCtx, defaultFlowKey))

			require.Error(t, err, "EnqueueAndWait must fail when context is cancelled during a blocking submit")
			assert.ErrorIs(t, err, types.ErrRejected, "error should wrap ErrRejected")
			assert.ErrorIs(t, err, context.Canceled, "error should wrap the underlying ctx.Err()")
			assert.Equal(t, types.QueueOutcomeRejectedCapacity, outcome, "outcome should be QueueOutcomeRejectedCapacity")
		})

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
						return fn(&mockActiveFlowConnection{ActiveShardsV: []contracts.RegistryShard{shardA, shardB}})
					}
					return fn(&mockActiveFlowConnection{ActiveShardsV: []contracts.RegistryShard{shardB}})
				},
			}
			h := newUnitHarness(t, t.Context(), Config{}, mockRegistry)
			h.mockProcessorFactory.processors["shard-A"] = &mockShardProcessor{
				SubmitFunc: func(item *internal.FlowItem) error {
					go item.Finalize(types.QueueOutcomeRejectedOther, contracts.ErrShardDraining)
					return nil
				},
			}
			h.mockProcessorFactory.processors["shard-B"] = &mockShardProcessor{
				SubmitFunc: func(item *internal.FlowItem) error {
					go item.Finalize(types.QueueOutcomeDispatched, nil)
					return nil
				},
			}

			outcome, err := h.fc.EnqueueAndWait(newTestRequest(context.Background(), defaultFlowKey))
			require.NoError(t, err, "EnqueueAndWait must succeed after retrying on a healthy shard")
			assert.Equal(t, types.QueueOutcomeDispatched, outcome, "outcome should be QueueOutcomeDispatched")
			assert.Equal(t, int32(2), callCount.Load(), "registry must be consulted for Active shards on each retry attempt")
		})
	})
}

func TestFlowController_Lifecycle(t *testing.T) {
	t.Parallel()

	t.Run("Reconciliation", func(t *testing.T) {
		t.Parallel()

		mockRegistry := &mockRegistryClient{
			// Configure the mock registry to report the new state without the stale shard.
			ShardStatsFunc: func() []contracts.ShardStats {
				return []contracts.ShardStats{{ID: "shard-A"}}
			}}
		h := newUnitHarness(t, t.Context(), Config{}, mockRegistry)

		// Pre-populate the controller with initial workers.
		initialShards := []string{"shard-A", "stale-shard"}
		for _, shardID := range initialShards {
			currentShardID := shardID
			h.mockProcessorFactory.processors[currentShardID] = &mockShardProcessor{runStarted: make(chan struct{})}
			shard := &mocks.MockRegistryShard{IDFunc: func() string { return currentShardID }}
			h.fc.getOrStartWorker(shard)
		}
		require.Len(t, h.mockProcessorFactory.processors, 2, "pre-condition: initial workers not set up correctly")

		// Wait for all workers to have started and set their contexts before proceeding with the test.
		for id, p := range h.mockProcessorFactory.processors {
			proc := p
			select {
			case <-proc.runStarted:
				// Success
			case <-time.After(2 * time.Second):
				t.Fatalf("timed out waiting for worker %s to start", id)
			}
		}

		// Manually trigger the reconciliation loop logic.
		h.fc.reconcileProcessors()

		t.Run("StaleWorkerIsCancelled", func(t *testing.T) {
			staleProc := h.mockProcessorFactory.processors["stale-shard"]
			require.NotNil(t, staleProc.Context(), "precondition: stale processor context should have been captured")
			select {
			case <-staleProc.Context().Done():
				// Success
			case <-time.After(100 * time.Millisecond):
				t.Error("context of removed worker must be cancelled")
			}
		})

		t.Run("ActiveWorkerIsNotCancelled", func(t *testing.T) {
			activeProc := h.mockProcessorFactory.processors["shard-A"]
			require.NotNil(t, activeProc.Context(), "precondition: active processor context should have been captured")
			select {
			case <-activeProc.Context().Done():
				t.Error("context of remaining worker must not be cancelled")
			default:
				// Success
			}
		})

		t.Run("WorkerMapIsUpdated", func(t *testing.T) {
			_, ok := h.fc.workers.Load("stale-shard")
			assert.False(t, ok, "stale worker must be removed from the controller's map")
			_, ok = h.fc.workers.Load("shard-A")
			assert.True(t, ok, "active worker must remain in the controller's map")
		})
	})

	t.Run("Reconciliation_IsTriggeredByTicker", func(t *testing.T) {
		t.Parallel()
		reconciliationInterval := 10 * time.Second
		mockRegistry := &mockRegistryClient{}

		var reconcileCount atomic.Int32
		mockRegistry.ShardStatsFunc = func() []contracts.ShardStats {
			reconcileCount.Add(1)
			return nil
		}

		h := newUnitHarness(t, t.Context(), Config{ProcessorReconciliationInterval: reconciliationInterval}, mockRegistry)

		// Wait for the reconciliation loop to start and create the ticker.
		// This prevents a race where the clock is stepped before the ticker is registered.
		require.Eventually(t, h.mockClock.HasWaiters, time.Second, 10*time.Millisecond, "ticker was not created")

		// Advance the clock to trigger the next reconciliation.
		h.mockClock.Step(reconciliationInterval)

		assert.Eventually(t, func() bool {
			return reconcileCount.Load() == 1
		}, time.Second, 10*time.Millisecond, "reconciliation was not triggered by the ticker")

		// Advance the clock again to ensure it continues to fire.
		h.mockClock.Step(reconciliationInterval)
		assert.Eventually(t, func() bool {
			return reconcileCount.Load() == 2
		}, time.Second, 10*time.Millisecond, "reconciliation did not fire on the second tick")
	})

	t.Run("WorkerCreationRace", func(t *testing.T) {
		t.Parallel()

		// This test requires manual control over the shard processor factory to deterministically create a race.
		factoryEntered := make(chan *mockShardProcessor, 2)
		continueFactory := make(chan struct{})

		h := newUnitHarness(t, t.Context(), Config{}, &mockRegistryClient{})
		h.fc.shardProcessorFactory = func(
			shard contracts.RegistryShard,
			_ internal.BandFilter,
			_ clock.Clock,
			_ time.Duration,
			_ int,
			_ logr.Logger,
		) shardProcessor {
			// This factory function will be called by `startNewWorker`.
			// We use channels to pause execution here, allowing two goroutines to enter this function before one "wins"
			// the `LoadOrStore` race.
			proc := &mockShardProcessor{runStarted: make(chan struct{})}
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

		// Wait for both goroutines to have entered the factory and created a processor.
		// This confirms they both missed the initial `workers.Load` check.
		proc1 := <-factoryEntered
		proc2 := <-factoryEntered

		// Unblock both goroutines, allowing them to race to `workers.LoadOrStore`.
		close(continueFactory)
		wg.Wait()

		// One processor "won" and was stored, the other "lost" and should have been cancelled.
		actual, ok := h.fc.workers.Load("race-shard")
		require.True(t, ok, "a worker should have been stored in the map")

		storedWorker := actual.(*managedWorker)
		winnerProc := storedWorker.processor.(*mockShardProcessor)

		var loserProc *mockShardProcessor
		if winnerProc == proc1 {
			loserProc = proc2
		} else {
			loserProc = proc1
		}

		// Wait for the `Run` method to be called on the winning processor to ensure its context is available.
		select {
		case <-winnerProc.runStarted:
		// Success.
		case <-time.After(1 * time.Second):
			t.Fatal("timed out waiting for winning worker to start")
		}

		// The winning processor's context must remain active.
		require.NotNil(t, winnerProc.Context(), "winner's context should not be nil")
		select {
		case <-winnerProc.Context().Done():
			t.Error("context of the winning worker should not be cancelled")
		default:
			// Success
		}

		// The losing processor's `Run` method must not be called, and its context should be nil.
		select {
		case <-loserProc.runStarted:
			t.Error("Run was called on the losing worker, but it should not have been")
		default:
			// Success
		}
		assert.Nil(t, loserProc.Context(), "loser's context should be nil as Run is never called")
	})
}

func TestFlowController_Concurrency(t *testing.T) {
	const (
		numShards     = 4
		numGoroutines = 50
		numRequests   = 200
	)

	// Set up a realistic registry that vends real components to the processor.
	mockRegistry := &mockRegistryClient{}
	shards := make([]contracts.RegistryShard, numShards)
	queues := make(map[string]contracts.ManagedQueue)
	for i := range numShards {
		shardID := fmt.Sprintf("shard-%d", i)
		queues[shardID] = &mocks.MockManagedQueue{FlowKeyV: defaultFlowKey} // Use the high-fidelity mock queue.
		shards[i] = &mocks.MockRegistryShard{
			IDFunc: func() string { return shardID },
			ManagedQueueFunc: func(_ types.FlowKey) (contracts.ManagedQueue, error) {
				return queues[shardID], nil
			},
			AllOrderedPriorityLevelsFunc: func() []int { return []int{100} },
			PriorityBandAccessorFunc: func(priority int) (framework.PriorityBandAccessor, error) {
				if priority == 100 {
					return &frameworkmocks.MockPriorityBandAccessor{
						PriorityNameV: "high",
						PriorityV:     100,
						IterateQueuesFunc: func(f func(framework.FlowQueueAccessor) bool) {
							f(queues[shardID].FlowQueueAccessor())
						},
					}, nil
				}
				return nil, fmt.Errorf("unexpected priority %d", priority)
			},
			IntraFlowDispatchPolicyFunc: func(_ types.FlowKey) (framework.IntraFlowDispatchPolicy, error) {
				return &frameworkmocks.MockIntraFlowDispatchPolicy{
					SelectItemFunc: func(qa framework.FlowQueueAccessor) (types.QueueItemAccessor, error) {
						return qa.PeekHead()
					},
				}, nil
			},
			InterFlowDispatchPolicyFunc: func(_ int) (framework.InterFlowDispatchPolicy, error) {
				return &frameworkmocks.MockInterFlowDispatchPolicy{
					SelectQueueFunc: func(band framework.PriorityBandAccessor) (framework.FlowQueueAccessor, error) {
						return queues[shardID].FlowQueueAccessor(), nil
					},
				}, nil
			},
			StatsFunc: func() contracts.ShardStats {
				return contracts.ShardStats{
					ID:            shardID,
					TotalLen:      uint64(queues[shardID].Len()),
					TotalByteSize: queues[shardID].ByteSize(),
					PerPriorityBandStats: map[int]contracts.PriorityBandStats{
						100: {
							Len:           uint64(queues[shardID].Len()),
							ByteSize:      queues[shardID].ByteSize(),
							CapacityBytes: 1e9, // Effectively unlimited capacity
						},
					},
				}
			},
		}
	}
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
	h := newIntegrationHarness(t, t.Context(), Config{
		// Use a generous buffer to prevent flakes in the test due to transient queuing delays.
		EnqueueChannelBufferSize: numRequests,
		DefaultRequestTTL:        1 * time.Second,
	}, mockRegistry)

	var wg sync.WaitGroup
	wg.Add(numGoroutines)
	outcomes := make(chan types.QueueOutcome, numRequests)
	for range numGoroutines {
		go func() {
			defer wg.Done()
			for range numRequests / numGoroutines {
				req := newTestRequest(logr.NewContext(context.Background(), logr.Discard()), defaultFlowKey)
				outcome, err := h.fc.EnqueueAndWait(req)
				if err != nil {
					// Use `t.Errorf` for concurrent tests to avoid halting execution on a single failure.
					t.Errorf("EnqueueAndWait failed unexpectedly: %v", err)
				}
				outcomes <- outcome
			}
		}()
	}
	wg.Wait()
	close(outcomes)

	successCount := 0
	for outcome := range outcomes {
		if outcome == types.QueueOutcomeDispatched {
			successCount++
		}
	}
	require.Equal(t, numRequests, successCount, "all concurrent requests should be dispatched successfully")
}
