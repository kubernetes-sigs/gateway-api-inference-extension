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

package metrics

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"sync"
	"time"

	"github.com/prometheus/common/expfmt"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
)

// Client is an interface for retrieving the metrics from an endpoint URL.
type Client interface {
	Get(ctx context.Context, target *url.URL, ep datalayer.Addressable) (PrometheusMetricMap, error)
}

// ClientFactory returns a Client suitable for an endpoint.
// Implementations may return a new Client each time, or use a cached
// copy for optimized retrieval.
type ClientFactory interface {
	GetClientForEndpoint(ep datalayer.Addressable) (Client, error)
}

// GetDefaultClientFactory returns a default implementation of the
// ClientFactory, which caches and reuses client across calls.
func GetDefaultClientFactory() ClientFactory {
	return defaultClientFactory
}

// -- package implementations --
var (
	cleanupTick          = 30 * time.Second
	maxIdleTime          = time.Minute
	defaultClientFactory = newClientFactory()
)

type client struct {
	cl       *http.Client
	lastUsed time.Time
}

type clientmap struct {
	cleanupOnce sync.Once
	clients     sync.Map // key: target (Pod) IP address, value: (cached) HTTP client
}

func newClientFactory() *clientmap {
	clm := &clientmap{}
	clm.startCleanupGoroutine()
	return clm
}

func (clm *clientmap) startCleanupGoroutine() {
	clm.cleanupOnce.Do(func() {
		go func() {
			for {
				time.Sleep(cleanupTick)
				now := time.Now()

				clm.clients.Range(func(key, value any) bool {
					entry := value.(*client)
					if now.Sub(entry.lastUsed) > maxIdleTime {
						clm.clients.Delete(key)
					}
					return true
				})
			}
		}()
	})
}

func (clm *clientmap) GetClientForEndpoint(ep datalayer.Addressable) (Client, error) {
	id := ep.GetIPAddress()

	if value, found := clm.clients.Load(id); found {
		cl, ok := value.(*client)
		if !ok {
			return nil, fmt.Errorf("invalid client stored for %s(%s)", id, ep.GetNamespacedName().String())
		}
		return cl, nil
	}

	value, _ := clm.clients.LoadOrStore(id, newClient()) // if stored, will return the new value
	cl, ok := value.(*client)
	if !ok {
		return nil, fmt.Errorf("invalid client stored for %s(%s)", id, ep.GetNamespacedName().String())
	}
	return cl, nil
}

func newClient() *client {
	return &client{
		cl: &http.Client{
			Timeout: 10 * time.Second,
			// TODO: set additional timeouts, transport options, etc.
		},
		lastUsed: time.Now(),
	}
}

func (cl *client) Get(ctx context.Context, target *url.URL, ep datalayer.Addressable) (PrometheusMetricMap, error) {
	cl.lastUsed = time.Now()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, target.String(), nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}
	resp, err := cl.cl.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch metrics from %s: %w", ep.GetNamespacedName(), err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code from %s: %v", ep.GetNamespacedName(), resp.StatusCode)
	}

	parser := expfmt.TextParser{}
	metricFamilies, err := parser.TextToMetricFamilies(resp.Body)
	if err != nil {
		return nil, err
	}
	return metricFamilies, err
}
