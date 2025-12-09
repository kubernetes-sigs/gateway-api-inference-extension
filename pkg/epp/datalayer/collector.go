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
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// TODO:
// currently the data store is expected to manage the state of multiple
// Collectors (e.g., using sync.Map mapping pod to its Collector). Alternatively,
// this can be encapsulated in this file, providing the data store with an interface
// to only update on endpoint addition/change and deletion. This can also be used
// to centrally track statistics such errors, active routines, etc.

const (
	defaultCollectionTimeout = time.Second
)

// Ticker implements a time source for periodic invocation.
// The Ticker is passed in as parameter a Collector to allow control over time
// progress in tests, ensuring tests are deterministic and fast.
type Ticker interface {
	Channel() <-chan time.Time
	Stop()
}

// TimeTicker implements a Ticker based on time.Ticker.
type TimeTicker struct {
	*time.Ticker
}

// NewTimeTicker returns a new time.Ticker with the configured duration.
func NewTimeTicker(d time.Duration) Ticker {
	return &TimeTicker{
		Ticker: time.NewTicker(d),
	}
}

// Channel exposes the ticker's channel.
func (t *TimeTicker) Channel() <-chan time.Time {
	return t.C
}

// Collector runs the data collection for a single endpoint.
type Collector struct {
	// per-endpoint context and cancellation
	ctx    context.Context
	cancel context.CancelFunc

	// goroutine management
	startOnce sync.Once
	stopOnce  sync.Once

	// TODO: optional metrics tracking collection (e.g., errors, invocations, ...)
}

// NewCollector returns a new collector.
func NewCollector() *Collector {
	return &Collector{}
}

// Start initiates data source collection for the endpoint.
// TODO: pass PoolInfo for backward compatibility
func (c *Collector) Start(ctx context.Context, ticker Ticker, ep Endpoint, sources []DataSource) error {
	var ready chan struct{}
	started := false

	c.startOnce.Do(func() {
		logger := log.FromContext(ctx).WithValues("endpoint", ep.GetMetadata().GetIPAddress())
		c.ctx, c.cancel = context.WithCancel(ctx)
		started = true
		ready = make(chan struct{})

		go func(endpoint Endpoint, sources []DataSource) {
			logger.V(logging.DEFAULT).Info("starting collection")

			defer func() {
				logger.V(logging.DEFAULT).Info("terminating collection")
				ticker.Stop()
			}()

			close(ready) // signal ready to accept ticks

			for {
				select {
				case <-c.ctx.Done(): // per endpoint context cancelled
					return
				case <-ticker.Channel():
					// TODO: do not collect if there's no pool specified?
					for _, src := range sources {
						ctx, cancel := context.WithTimeout(c.ctx, defaultCollectionTimeout)
						_ = src.Collect(ctx, endpoint) // TODO: track errors per collector?
						cancel()                       // release the ctx timeout resources
					}
				}
			}
		}(ep, sources)
	})

	if !started {
		return errors.New("collector start called multiple times")
	}

	// Wait for goroutine to signal readiness.
	// The use of ready channel is mostly to make the function testable, by ensuring
	// synchronous order of events. Ignoring test requirements, one could let the
	// go routine start at some arbitrary point in the future, possibly after this
	// function has returned.
	select {
	case <-ready:
		return nil
	case <-ctx.Done():
		if c.cancel != nil {
			c.cancel() // ensure clean up
		}
		return ctx.Err()
	}
}

// Stop terminates the collector.
func (c *Collector) Stop() error {
	if c.ctx == nil || c.cancel == nil {
		return errors.New("collector stop called before start")
	}

	stopped := false
	c.stopOnce.Do(func() {
		stopped = true
		c.cancel()
	})

	if !stopped {
		return errors.New("collector stop called multiple times")
	}
	return nil
}
