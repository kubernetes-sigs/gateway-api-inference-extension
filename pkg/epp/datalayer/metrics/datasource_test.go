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
	"time"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
)

func TestDatasource(t *testing.T) {
	source := NewMetricsDataSource("https", "/metrics", true)
	extractor, err := NewModelServerExtractor(defaultTotalQueuedRequestsMetric, "", "", "", "")
	assert.Nil(t, err, "failed to create extractor")

	dsType := source.TypedName().Type
	assert.Equal(t, MetricsDataSourceType, dsType)

	err = source.AddExtractor(extractor)
	assert.Nil(t, err, "failed to add extractor")

	err = source.AddExtractor(extractor)
	assert.NotNil(t, err, "expected to fail to add the same extractor twice")

	extractors := source.Extractors()
	assert.Len(t, extractors, 1)
	assert.Equal(t, extractor.TypedName().String(), extractors[0])

	err = datalayer.RegisterSource(source)
	assert.Nil(t, err, "failed to register")

	ctx := context.Background()
	factory := datalayer.NewEndpointFactory([]datalayer.DataSource{source}, 100*time.Millisecond)
	pod := &datalayer.EndpointMetadata{
		NamespacedName: types.NamespacedName{
			Name:      "pod1",
			Namespace: "default",
		},
		Address: "1.2.3.4:5678",
	}
	endpoint := factory.NewEndpoint(ctx, pod, nil)
	assert.NotNil(t, endpoint, "failed to create endpoint")

	err = source.Collect(ctx, endpoint)
	assert.NotNil(t, err, "expected to fail to collect metrics")
}
