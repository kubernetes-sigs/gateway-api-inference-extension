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
	"sync"
	"time"
)

// TODO:
// currently the data store is expected to manage the state of multiple
// Collectors (e.g., using sync.Map mapping pod to its Collector). Alternatively,
// this can be encapsulated in this file, providing the data store with an interface
// to only update on endpoint addition/change and deletion. This can also be used
// to centrally track statistics such errors, active routines, etc.

// Ticker implements a time source for periodic invocation.
// Defined as an interface to allow mocking in tests.
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
	ticker Ticker
	ctx    context.Context
	cancel context.CancelFunc

	// goroutine management
	startOnce sync.Once
	stopOnce  sync.Once
	done      chan struct{}
	// wg sync.WaitGroup needed? Add on start, Wait on stop

	// TODO: optional metrics tracking collection (e.g., errors, invocations, ...)
}

// NewCollector returns a new collector.
func NewCollector() *Collector {
	return &Collector{}
}

// Start initiates data source collection for the endpoint.
func (c *Collector) Start(ctx context.Context, tick Ticker, ep Endpoint, registry *DataSourceRegistry) {
	c.startOnce.Do(func() {
		c.ctx, c.cancel = context.WithCancel(ctx)
		c.done = make(chan struct{})
		c.ticker = tick
		// c.wg.Add(1)

		// run the collection go routine
		datasources := registry.GetSources()
		go func(endpoint Endpoint, sources []DataSource) {
			defer func() {
				// TODO: defer completion functions (e.g., end of collection log, wg.Done())
			}()

			for {
				select {
				case <-ctx.Done(): // global context cancelled (TODO: needed?)
					return
				case <-c.ctx.Done(): // per endpoint context cancelled
					return
				case <-c.done: // explicit stop signal
					return
				case <-c.ticker.Channel():
					for _, src := range sources {
						// TODO: track errors, add context input and error return?
						src.Collect(endpoint)
					}
				}
			}
		}(ep, datasources)
	})
}

func (c *Collector) Stop() {
	c.stopOnce.Do(func() {
		c.cancel()
		close(c.done)
		c.ticker.Stop()
		// c.wg.Wait() TODO: wait for goroutine to finish?
	})
}
