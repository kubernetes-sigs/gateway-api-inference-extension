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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
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

	// Sources and extractors for data collection
	// TODO: remove. Only used in Start
	pollers    []fwkdl.PollingDataSource
	extractors []fwkdl.Extractor

	// goroutine management
	startOnce sync.Once
	stopOnce  sync.Once

	// TODO: optional metrics tracking collection (e.g., errors, invocations, ...)
}

// NewCollector returns a new collector.
func NewCollector() *Collector {
	return &Collector{}
}

// NewCollectorWithExtractors returns a new collector with the given sources and extractors.
func NewCollectorWithExtractors(pollers []fwkdl.PollingDataSource, extractors []fwkdl.Extractor) *Collector {
	return &Collector{
		pollers:    pollers,
		extractors: extractors,
	}
}

// Start initiates data source collection for the endpoint.
// All sources must implement PollingDataSource. Validation is performed by the caller.
// If sources is provided, uses them; otherwise uses pollers/extractors from struct (set via NewCollectorWithExtractors).
// TODO: pass in collectors and for each set of extractors (for example: sources []fwkdl.DataSource, extractors map[string][]fwkdl.Extractor)
func (c *Collector) Start(ctx context.Context, ticker Ticker, ep fwkdl.Endpoint, sources ...[]fwkdl.DataSource) error {
	// Determine pollers to use
	var pollers []fwkdl.PollingDataSource

	switch { // TODO always use passed in sources and extractors, there is no fields on struct after removal.
	case len(sources) > 0 && sources[0] != nil && len(sources[0]) > 0:
		// Use provided sources
		for _, src := range sources[0] {
			if src == nil {
				return errors.New("cannot add nil data source")
			}
			pollers = append(pollers, src.(fwkdl.PollingDataSource))
		}
	case len(c.pollers) > 0:
		// Use struct's pollers
		pollers = c.pollers
	default:
		return errors.New("cannot start collector with no pollers")
	}

	var extractors []fwkdl.Extractor
	if len(c.extractors) > 0 {
		extractors = c.extractors
	}

	return c.startCollection(ctx, ticker, ep, pollers, extractors)
}

func (c *Collector) startCollection(ctx context.Context, ticker Ticker, ep fwkdl.Endpoint, pollers []fwkdl.PollingDataSource, extractors []fwkdl.Extractor) error {
	var ready chan struct{}
	started := false

	c.startOnce.Do(func() {
		logger := log.FromContext(ctx).WithValues("endpoint", ep.GetMetadata().GetIPAddress())
		c.ctx, c.cancel = context.WithCancel(ctx)
		started = true
		ready = make(chan struct{})

		go func(endpoint fwkdl.Endpoint, sources []fwkdl.PollingDataSource, exts []fwkdl.Extractor) {
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
						data, err := src.Poll(ctx, endpoint)
						if err != nil {
							logger.Error(err, "poll failed", "source", src.TypedName())
							cancel()
							continue
						}
						cancel()
						// If extractors provided, call them with the data
						if exts != nil && data != nil {
							for _, ext := range exts {
								if err := ext.Extract(ctx, data, endpoint); err != nil {
									logger.Error(err, "extract failed", "extractor", ext.TypedName())
								}
							}
						}
					}
				}
			}
		}(ep, pollers, extractors)
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
