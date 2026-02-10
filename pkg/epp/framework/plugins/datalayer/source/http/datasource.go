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
	"errors"
	"fmt"
	"io"
	"net/url"
	"reflect"
	"sync"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// HTTPDataSource is a data source that receives its data using HTTP client.
type HTTPDataSource struct {
	typedName fwkplugin.TypedName
	scheme    string // scheme to use
	path      string // path to use

	client     Client // client (e.g. a wrapped http.Client) used to get data
	parser     func(io.Reader) (any, error)
	outputType reflect.Type
	extractors sync.Map // key: name, value: extractor
}

// NewHTTPDataSource returns a new data source, configured with
// the provided scheme, path and certificate verification parameters.
func NewHTTPDataSource(scheme string, path string, skipCertVerification bool, pluginType string,
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
		scheme:     scheme,
		path:       path,
		client:     defaultClient,
		parser:     parser,
		outputType: outputType,
	}
	return dataSrc, nil
}

// TypedName returns the data source type and name.
func (dataSrc *HTTPDataSource) TypedName() fwkplugin.TypedName {
	return dataSrc.typedName
}

// Extractors returns a list of registered Extractor names.
func (dataSrc *HTTPDataSource) Extractors() []string {
	extractors := []string{}
	dataSrc.extractors.Range(func(_, val any) bool {
		if ex, ok := val.(fwkdl.Extractor); ok {
			extractors = append(extractors, ex.TypedName().String())
		}
		return true // continue iteration
	})
	return extractors
}

// AddExtractor adds an extractor to the data source, validating it can process
// the data source output type.
func (dataSrc *HTTPDataSource) AddExtractor(extractor fwkdl.Extractor) error {
	if err := datalayer.ValidateExtractorType(dataSrc.outputType, extractor.ExpectedInputType()); err != nil {
		return err
	}
	if _, loaded := dataSrc.extractors.LoadOrStore(extractor.TypedName().Name, extractor); loaded {
		return fmt.Errorf("attempt to add duplicate extractor %s to %s", extractor.TypedName(), dataSrc.TypedName())
	}
	return nil
}

// Collect is triggered by the data layer framework to fetch potentially new
// data for an endpoint.
func (dataSrc *HTTPDataSource) Collect(ctx context.Context, ep fwkdl.Endpoint) error {
	target := dataSrc.getEndpoint(ep.GetMetadata())
	data, err := dataSrc.client.Get(ctx, target, ep.GetMetadata(), dataSrc.parser)

	if err != nil {
		return err
	}

	var errs []error
	dataSrc.extractors.Range(func(_, val any) bool {
		if ex, ok := val.(fwkdl.Extractor); ok {
			if err = ex.Extract(ctx, data, ep); err != nil {
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

func (dataSrc *HTTPDataSource) getEndpoint(ep Addressable) *url.URL {
	return &url.URL{
		Scheme: dataSrc.scheme,
		Host:   ep.GetMetricsHost(),
		Path:   dataSrc.path,
	}
}

var _ fwkdl.DataSource = (*HTTPDataSource)(nil)
