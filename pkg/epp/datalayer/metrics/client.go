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

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
)

// Client is an interface for retrieving the metrics from an endpoint URL.
type Client interface {
	Get(ctx context.Context, target *url.URL, ep datalayer.Addressable) (PrometheusMetricMap, error)
}

// -- package implementations --
const (
	maxIdleConnections = 5000
	maxIdleTime        = 10 * time.Second
	timeout            = 10 * time.Second
)

var (
	defaultClient = &client{
		Client: http.Client{
			Timeout: timeout,
			Transport: &http.Transport{
				MaxIdleConns:        maxIdleConnections,
				MaxIdleConnsPerHost: 4, // host is defined as scheme://host:port
			},
			// TODO: set additional timeouts, transport options, etc.
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

	parser := expfmt.TextParser{}
	metricFamilies, err := parser.TextToMetricFamilies(resp.Body)
	if err != nil {
		return nil, err
	}
	return metricFamilies, err
}
