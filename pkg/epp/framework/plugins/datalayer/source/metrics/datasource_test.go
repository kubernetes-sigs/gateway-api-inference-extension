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
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/source/http"
)

func TestDatasource(t *testing.T) {
	_, err := http.NewHTTPDataSource("invalid", "/metrics", true, MetricsDataSourceType,
		"metrics-data-source", parseMetrics, PrometheusMetricType)
	assert.NotNil(t, err, "expected to fail with invalid scheme")

	source, err := http.NewHTTPDataSource("https", "/metrics", true, MetricsDataSourceType,
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
	err = source.Collect(ctx, endpoint)
	assert.NotNil(t, err, "expected to fail to collect metrics")
}
