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
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/source/http"
)

func TestDatasource(t *testing.T) {
	_, err := http.NewHTTPDataSource("invalid", "/metrics", true, 0, MetricsDataSourceType,
		"metrics-data-source", parseMetrics, PrometheusMetricType)
	assert.NotNil(t, err, "expected to fail with invalid scheme")

	source, err := http.NewHTTPDataSource("https", "/metrics", true, 0, MetricsDataSourceType,
		"metrics-data-source", parseMetrics, PrometheusMetricType)
	assert.Nil(t, err, "failed to create HTTP datasource")

	dsType := source.TypedName().Type
	assert.Equal(t, MetricsDataSourceType, dsType)

	ctx := context.Background()
	endpoint := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		NamespacedName: types.NamespacedName{
			Name:      "pod1",
			Namespace: "default",
		},
		Address: "1.2.3.4:5678",
	}, nil)
	_, err = source.Poll(ctx, endpoint)
	assert.NotNil(t, err, "expected to fail polling for metrics")
}

func TestMetricsDataSourceFactory_MetricsPortOverride(t *testing.T) {
	params, err := json.Marshal(map[string]any{
		"scheme":      "http",
		"metricsPort": 9090,
	})
	require.NoError(t, err)

	plugin, err := MetricsDataSourceFactory("test-ds", params, nil)
	require.NoError(t, err)

	ds, ok := plugin.(fwkdl.PollingDataSource)
	require.True(t, ok, "expected MetricsDataSourceFactory to return a PollingDataSource")

	// Poll will fail (no real server), but the error must reference port 9090.
	// If metricsPort were ignored, EPP would dial :8000 instead.
	endpoint := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "pod1", Namespace: "default"},
		MetricsHost:    "1.2.3.4:8000",
	}, nil)
	_, err = ds.Poll(context.Background(), endpoint)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "9090", "expected scrape target to use metricsPort 9090, not inference port 8000")
}
