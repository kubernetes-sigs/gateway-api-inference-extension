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

package common

import (
	"context"
	"crypto/tls"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/fsnotify/fsnotify"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

// debounceDelay wait for events to settle before reloading
const debounceDelay = 250 * time.Millisecond

type CertReloader struct {
	cert *atomic.Pointer[tls.Certificate]
}

func NewCertReloader(ctx context.Context, path string, init *tls.Certificate) (*CertReloader, error) {
	certPtr := &atomic.Pointer[tls.Certificate]{}
	certPtr.Store(init)

	w, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, fmt.Errorf("failed to create cert watcher: %w", err)
	}

	logger := log.FromContext(ctx).
		WithName("cert-reloader").
		WithValues("path", path)
	traceLogger := logger.V(logutil.TRACE)

	if err := w.Add(path); err != nil {
		_ = w.Close() // Clean up watcher before returning
		return nil, fmt.Errorf("failed to watch %q: %w", path, err)
	}

	go func() {
		defer w.Close()

		var debounceTimer *time.Timer

		for {
			select {
			case ev := <-w.Events:
				traceLogger.Info("Cert changed", "event", ev)

				if ev.Op&(fsnotify.Write|fsnotify.Create) == 0 {
					continue
				}

				// Debounce: reset the timer if we get another event
				if debounceTimer != nil {
					debounceTimer.Stop()
				}

				debounceTimer = time.AfterFunc(debounceDelay, func() {
					// This runs after the delay with no new events
					cert, err := tls.LoadX509KeyPair(path+"/tls.crt", path+"/tls.key")
					if err != nil {
						logger.Error(err, "Failed to reload TLS certificate")
						return
					}
					certPtr.Store(&cert)
					traceLogger.Info("Reloaded TLS certificate")
				})

			case err := <-w.Errors:
				if err != nil {
					logger.Error(err, "cert watcher failed")
				}
			case <-ctx.Done():
				return
			}
		}
	}()

	return &CertReloader{cert: certPtr}, nil
}

func (r *CertReloader) Get() *tls.Certificate {
	return r.cert.Load()
}
