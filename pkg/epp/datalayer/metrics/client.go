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
	"time"

	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
)

// Client is an interface for retrieving the metrics from an endpoint URL.
type Client interface {
	Get(ctx context.Context, target *url.URL, ep datalayer.Addressable) (PrometheusMetricMap, error)
}

const (
	// the maximum idle connection count is shared by all endpoints. The value is
	// set high to ensure the number of idle connection is above the expected
	// endpoint count. Setting it too low would cause thrashing of the idle connection
	// pool and incur higher overheads for every GET (e.g., socket initiation, certificate
	// exchange, connections in timed wait state, etc.).
	maxIdleConnections = 5000
	maxIdleTime        = 10 * time.Second // once a endpoint goes down, allow closing.
	timeout            = 10 * time.Second // mostly guard against unresponsive endpoints.
	// allow some grace when connections are not made idle immediately (e.g., parsing
	// and updating might take some time). This allows maintaining up to two idle connections
	// per endpoint (defined as scheme://host:port).
	maxIdleConnsPerHost = 2
)

var (
	baseTransport = &http.Transport{
		MaxIdleConns:        maxIdleConnections,
		MaxIdleConnsPerHost: maxIdleConnsPerHost,
		// TODO: set additional timeouts, transport options, etc.
	}
	defaultClient = &client{
		Client: http.Client{
			Timeout:   timeout,
			Transport: baseTransport,
		},
	}
)

type client struct {
	http.Client
}

func (cl *client) Get(ctx context.Context, target *url.URL, ep datalayer.Addressable) (PrometheusMetricMap, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, target.String(), nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}
	resp, err := defaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch metrics from %s: %w", ep.GetNamespacedName(), err)
	}
	defer func() {
		_ = resp.Body.Close()
	}()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code from %s: %v", ep.GetNamespacedName(), resp.StatusCode)
	}

	parser := expfmt.NewTextParser(model.LegacyValidation)
	metricFamilies, err := parser.TextToMetricFamilies(resp.Body)
	if err != nil {
		return nil, err
	}
	return metricFamilies, err
}
