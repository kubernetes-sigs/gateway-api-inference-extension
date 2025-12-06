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
	"crypto/tls"
	"errors"
	"fmt"
	"net/url"
	"sync"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
)

// DataSource is a Model Server Protocol (MSP) compliant metrics data source,
// returning Prometheus formatted metrics for an endpoint.
type DataSource struct {
	typedName     plugins.TypedName
	metricsScheme string // scheme to use in metrics URL
	metricsPath   string // path to use in metrics URL

	client     Client   // client (e.g. a wrapped http.Client) used to get metrics
	extractors sync.Map // key: name, value: extractor
}

// NewMetricsDataSource returns a new MSP compliant metrics data source, configured with
// the provided scheme, path and certificate verification parameters.
func NewMetricsDataSource(metricsScheme string, metricsPath string, skipCertVerification bool) *DataSource {
	if metricsScheme == "https" {
		httpsTransport := baseTransport.Clone()
		httpsTransport.TLSClientConfig = &tls.Config{
			InsecureSkipVerify: skipCertVerification,
		}
		defaultClient.Transport = httpsTransport
	}

	dataSrc := &DataSource{
		typedName: plugins.TypedName{
			Type: MetricsDataSourceType,
			Name: MetricsDataSourceType,
		},
		metricsScheme: metricsScheme,
		metricsPath:   metricsPath,
		client:        defaultClient,
	}
	return dataSrc
}

// TypedName returns the metrics data source type and name.
func (dataSrc *DataSource) TypedName() plugins.TypedName {
	return dataSrc.typedName
}

// Extractors returns a list of registered Extractor names.
func (dataSrc *DataSource) Extractors() []string {
	extractors := []string{}
	dataSrc.extractors.Range(func(_, val any) bool {
		if ex, ok := val.(datalayer.Extractor); ok {
			extractors = append(extractors, ex.TypedName().String())
		}
		return true // continue iteration
	})
	return extractors
}

// AddExtractor adds an extractor to the data source, validating it can process
// the metrics' data source output type.
func (dataSrc *DataSource) AddExtractor(extractor datalayer.Extractor) error {
	if err := datalayer.ValidateExtractorType(PrometheusMetricType, extractor.ExpectedInputType()); err != nil {
		return err
	}
	if _, loaded := dataSrc.extractors.LoadOrStore(extractor.TypedName().Name, extractor); loaded {
		return fmt.Errorf("attempt to add duplicate extractor %s to %s", extractor.TypedName(), dataSrc.TypedName())
	}
	return nil
}

// Collect is triggered by the data layer framework to fetch potentially new
// MSP metrics data for an endpoint.
func (dataSrc *DataSource) Collect(ctx context.Context, ep datalayer.Endpoint) error {
	target := dataSrc.getMetricsEndpoint(ep.GetMetadata())
	families, err := dataSrc.client.Get(ctx, target, ep.GetMetadata())

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
		Host:   ep.GetMetricsHost(),
		Path:   dataSrc.metricsPath,
	}
}
