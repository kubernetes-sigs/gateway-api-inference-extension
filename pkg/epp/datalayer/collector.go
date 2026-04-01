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

	"github.com/go-logr/logr"
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

	// goroutine management
	startOnce sync.Once
	stopOnce  sync.Once

	// done is closed when the collection goroutine exits, allowing Stop to block until it is safe
	// to inspect internal state (e.g., in tests).
	done chan struct{}

	// lastPollErrors and lastExtractErrors track the last error per source/extractor
	// for change-only logging. Only accessed from the collection goroutine — no synchronization required.
	lastPollErrors    map[string]error
	lastExtractErrors map[string]error
}

// NewCollector returns a new collector.
func NewCollector() *Collector {
	return &Collector{
		done:              make(chan struct{}),
		lastPollErrors:    make(map[string]error),
		lastExtractErrors: make(map[string]error),
	}
}

// Start initiates data source collection for the endpoint.
// All sources must implement PollingDataSource. Validation is performed by the caller.
func (c *Collector) Start(ctx context.Context, ticker Ticker, ep fwkdl.Endpoint, pollers []fwkdl.PollingDataSource, extractors map[string][]fwkdl.Extractor) error {
	// Validate sources slice is not empty
	if len(pollers) == 0 {
		return errors.New("cannot start collector with empty sources")
	}

	for _, src := range pollers {
		if src == nil {
			return errors.New("cannot add nil data source")
		}
	}

	return c.startCollection(ctx, ticker, ep, pollers, extractors)
}

func (c *Collector) startCollection(ctx context.Context, ticker Ticker, ep fwkdl.Endpoint, pollers []fwkdl.PollingDataSource, extractors map[string][]fwkdl.Extractor) error {
	var ready chan struct{}
	started := false

	c.startOnce.Do(func() {
		logger := log.FromContext(ctx).WithValues("endpoint", ep.GetMetadata().GetIPAddress())
		c.ctx, c.cancel = context.WithCancel(ctx)
		started = true
		ready = make(chan struct{})

		go func(endpoint fwkdl.Endpoint, sources []fwkdl.PollingDataSource, exts map[string][]fwkdl.Extractor) {
			logger.V(logging.DEFAULT).Info("starting collection")

			defer func() {
				logger.V(logging.DEFAULT).Info("terminating collection")
				ticker.Stop()
				close(c.done)
			}()

			close(ready) // signal ready to accept ticks

			for {
				select {
				case <-c.ctx.Done(): // per endpoint context cancelled
					return
				case <-ticker.Channel():
					for _, src := range sources {
						tn := src.TypedName()
						key := tn.String()

						ctx, cancel := context.WithTimeout(c.ctx, defaultCollectionTimeout)
						data, err := src.Poll(ctx, endpoint)
						cancel()

						logErrorTransition(logger, c.lastPollErrors, key, "poll", "source", err)
						if err != nil {
							continue
						}

						if srcExtractors, ok := exts[tn.Name]; ok && data != nil {
							for _, ext := range srcExtractors {
								extKey := ext.TypedName().String()
								extErr := ext.Extract(ctx, data, endpoint)
								logErrorTransition(logger, c.lastExtractErrors, extKey, "extract", "extractor", extErr)
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

// Stop terminates the collector and waits for the collection goroutine to exit.
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
	<-c.done
	return nil
}

// logErrorTransition logs only when the error state transitions (nil→non-nil or non-nil→nil).
// errs is updated in place. verb is the operation name ("poll"/"extract"); fieldName is the log key.
func logErrorTransition(logger logr.Logger, errs map[string]error, key, verb, fieldName string, err error) {
	prev, seen := errs[key]
	if (err != nil) != (seen && prev != nil) {
		if err != nil {
			logger.Error(err, verb+" failed", fieldName, key)
		} else {
			logger.V(logging.DEFAULT).Info(verb+" recovered", fieldName, key)
		}
		errs[key] = err
	}
}
