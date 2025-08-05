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
	"errors"
	"fmt"
	"net"
	"net/url"
	"strconv"
	"sync"
	"sync/atomic"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
)

const (
	dataSourceName = "metrics-data-source"
)

// DataSource is a Model Server Protocol (MSP) compliant metrics data source,
// returning Prometheus formatted metrics for an endpoint.
type DataSource struct {
	metricsScheme string                 // scheme to use in metrics URL
	metricsPort   atomic.Pointer[string] // target port to use in metrics URL
	metricsPath   string                 // path to use in metrics URL

	client     Client   // client (e.g. a wrapped http.Client) used to get metrics
	extractors sync.Map // key: name, value: extractor
}

// NewDataSource returns a new MSP compliant metrics data source, configured with the provided
// client factory. If ClientFactory is nil, a default factory is used.
func NewDataSource(metricsScheme string, metricsPort int32, metricsPath string, cl Client) *DataSource {
	if cl == nil {
		cl = defaultClient
	}

	dataSrc := &DataSource{
		metricsScheme: metricsScheme,
		metricsPath:   metricsPath,
		client:        cl,
	}
	dataSrc.SetPort(metricsPort)
	return dataSrc
}

// SetPort updates the port used for metrics scraping.
func (dataSrc *DataSource) SetPort(metricsPort int32) {
	port := strconv.Itoa(int(metricsPort))
	dataSrc.metricsPort.Store(&port)
}

// Name returns the metrics data source name.
func (dataSrc *DataSource) Name() string {
	return dataSourceName
}

// AddExtractor adds an extractor to the data source, validating it can process
// the metrics' data source output type.
func (dataSrc *DataSource) AddExtractor(extractor datalayer.Extractor) error {
	if err := datalayer.ValidateExtractorType(PrometheusMetricType, extractor.ExpectedInputType()); err != nil {
		return err
	}
	if _, loaded := dataSrc.extractors.LoadOrStore(extractor.Name(), extractor); loaded {
		return fmt.Errorf("attempt to add extractor with duplicate name %s to %s", extractor.Name(), dataSrc.Name())
	}
	return nil
}

// Collect is triggered by the data layer framework to fetch potentially new
// MSP metrics data for an endpoint.
func (dataSrc *DataSource) Collect(ctx context.Context, ep datalayer.Endpoint) error {
	target := dataSrc.getMetricsEndpoint(ep.GetPod())
	families, err := dataSrc.client.Get(ctx, target, ep.GetPod())

	if err != nil {
		return err
	}

	var errs []error
	dataSrc.extractors.Range(func(_, val any) bool {
		if ex, ok := val.(datalayer.Extractor); ok {
			if err = ex.Extract(ctx, families, ep); err != nil {
				errs = append(errs, err)
			}
		}
		return true // continue iteration
	})

	if len(errs) != 0 {
		return errors.Join(errs...)
	}
	return nil
}

func (dataSrc *DataSource) getMetricsEndpoint(ep datalayer.Addressable) *url.URL {
	return &url.URL{
		Scheme: dataSrc.metricsScheme,
		Host:   net.JoinHostPort(ep.GetIPAddress(), *dataSrc.metricsPort.Load()),
		Path:   dataSrc.metricsPath,
	}
}
