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

package http

import (
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"net"
	"net/url"
	"reflect"
	"strconv"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// HTTPDataSource is a data source that receives its data using HTTP client.
type HTTPDataSource struct {
	typedName   fwkplugin.TypedName
	scheme      string // scheme to use
	path        string // path to use
	metricsPort int    // when non-zero, overrides the port in MetricsHost for scraping

	client     Client // client (e.g. a wrapped http.Client) used to get data
	parser     func(io.Reader) (any, error)
	outputType reflect.Type
}

// NewHTTPDataSource returns a new data source configured with the given scheme, path,
// and certificate verification. metricsPort overrides the port in MetricsHost when
// non-zero; pass 0 to use MetricsHost as-is.
func NewHTTPDataSource(scheme string, path string, skipCertVerification bool, metricsPort int, pluginType string,
	pluginName string, parser func(io.Reader) (any, error), outputType reflect.Type) (*HTTPDataSource, error) {
	if scheme != "http" && scheme != "https" {
		return nil, fmt.Errorf("unsupported scheme: %s", scheme)
	}
	if scheme == "https" {
		httpsTransport := baseTransport.Clone()
		httpsTransport.TLSClientConfig = &tls.Config{
			InsecureSkipVerify: skipCertVerification,
		}
		defaultClient.Transport = httpsTransport
	}

	dataSrc := &HTTPDataSource{
		typedName: fwkplugin.TypedName{
			Type: pluginType,
			Name: pluginName,
		},
		scheme:      scheme,
		path:        path,
		metricsPort: metricsPort,
		client:      defaultClient,
		parser:      parser,
		outputType:  outputType,
	}
	return dataSrc, nil
}

// TypedName returns the data source type and name.
func (dataSrc *HTTPDataSource) TypedName() fwkplugin.TypedName {
	return dataSrc.typedName
}

// OutputType returns the type of data this DataSource produces.
func (dataSrc *HTTPDataSource) OutputType() reflect.Type {
	return dataSrc.outputType
}

// ExtractorType returns the type of Extractor this DataSource expects.
func (dataSrc *HTTPDataSource) ExtractorType() reflect.Type {
	return fwkdl.ExtractorType
}

// Poll fetches data for an endpoint and returns it.
func (dataSrc *HTTPDataSource) Poll(ctx context.Context, ep fwkdl.Endpoint) (any, error) {
	target := dataSrc.getEndpoint(ep.GetMetadata())
	return dataSrc.client.Get(ctx, target, ep.GetMetadata(), dataSrc.parser)
}

func (dataSrc *HTTPDataSource) getEndpoint(ep Addressable) *url.URL {
	host := ep.GetMetricsHost()
	if dataSrc.metricsPort != 0 {
		ip, _, err := net.SplitHostPort(host)
		if err == nil {
			host = net.JoinHostPort(ip, strconv.Itoa(dataSrc.metricsPort))
		}
		// If SplitHostPort fails (e.g. host has no port), use MetricsHost unchanged
		// so we still attempt a scrape rather than silently dropping the endpoint.
	}
	return &url.URL{
		Scheme: dataSrc.scheme,
		Host:   host,
		Path:   dataSrc.path,
	}
}

var _ fwkdl.DataSource = (*HTTPDataSource)(nil)
var _ fwkdl.PollingDataSource = (*HTTPDataSource)(nil)
