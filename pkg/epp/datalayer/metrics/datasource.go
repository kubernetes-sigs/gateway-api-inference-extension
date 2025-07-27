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
	"net"
	"net/url"
	"strconv"
	"sync"
	"sync/atomic"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
)

const (
	datasourceName = "metrics-data-source"
)

// DataSource is the metrics data source, returning Prometheus formatted
// metrics from an endpoint.
type DataSource struct {
	mspScheme string
	mspPort   atomic.Pointer[string]
	mspPath   string

	clients    ClientFactory
	extractors sync.Map // key: name, value: extractor
}

// NewDataSource returns a new metrics data source, configured with the provided
// client factory. If ClientFactory is nil, a default factory is used.
func NewDataSource(metricsScheme string, metricsPort int32, metricsPath string, clf ClientFactory) *DataSource {
	if clf == nil {
		clf = GetDefaultClientFactory()
	}
	ds := &DataSource{
		mspScheme: metricsScheme,
		mspPath:   metricsPath,
		clients:   clf,
	}
	ds.SetPort(metricsPort)
	return ds
}

// SetPort updates the port used for metrics scraping.
func (ds *DataSource) SetPort(metricsPort int32) {
	port := strconv.Itoa(int(metricsPort))
	ds.mspPort.Store(&port)
}

// Name returns the metrics data source name.
func (ds *DataSource) Name() string {
	return datasourceName
}

// AddExtractor adds an extractor to the data source, validating it can process
// the metrics' data source output type.
func (ds *DataSource) AddExtractor(extractor datalayer.Extractor) error {
	if err := datalayer.ValidateExtractorType(PrometheusMetricType, extractor.ExpectedInputType()); err != nil {
		return err
	}
	if _, loaded := ds.extractors.LoadOrStore(extractor.Name(), extractor); loaded {
		return fmt.Errorf("attempt to add extractor with duplicate name %s to %s", extractor.Name(), ds.Name())
	}
	return nil
}

// Collect is triggered by the data layer framework to fetch potentially new
// metrics data for an endpoint.
//
// TODO: context.Context input (e.g., for logger); error return?
func (ds *DataSource) Collect(ep datalayer.Endpoint) {
	cl := ds.clients.GetClientForEndpoint(ep.GetPod())
	if cl == nil {
		// log error and return
		return
	}

	target := ds.getMetricsEndpoint(ep.GetPod())
	families, err := cl.Get(context.TODO(), target, ep.GetPod())

	if err != nil {
		// log error and return
		return
	}
	ds.extractors.Range(func(_, val any) bool {
		if ex, ok := val.(Extractor); ok {
			ex.Extract(families, ep) // TODO: provide context.Context, track errors?
		}
		return true // continue iteration
	})
}

func (ds *DataSource) getMetricsEndpoint(ep datalayer.Addressable) *url.URL {
	return &url.URL{
		Scheme: ds.mspScheme,
		Host:   net.JoinHostPort(ep.GetIPAddress(), *ds.mspPort.Load()),
		Path:   ds.mspPath,
	}
}
