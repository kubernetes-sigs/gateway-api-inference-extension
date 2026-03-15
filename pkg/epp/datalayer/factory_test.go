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

package datalayer

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
)

func TestFactory(t *testing.T) {
	runtime := NewTestRuntime(t, 100*time.Millisecond)

	pod1 := &fwkdl.EndpointMetadata{
		NamespacedName: types.NamespacedName{
			Name:      "pod1",
			Namespace: "default",
		},
		Address: "1.2.3.4:5678",
	}
	endpoint1 := runtime.NewEndpoint(context.Background(), pod1, nil)
	assert.NotNil(t, endpoint1, "failed to create endpoint")

	dup := runtime.NewEndpoint(context.Background(), pod1, nil)
	assert.Nil(t, dup, "expected to fail to create a duplicate collector")

	pod2 := &fwkdl.EndpointMetadata{
		NamespacedName: types.NamespacedName{
			Name:      "pod2",
			Namespace: "default",
		},
		Address: "1.2.3.4:5679",
	}
	endpoint2 := runtime.NewEndpoint(context.Background(), pod2, nil)
	assert.NotNil(t, endpoint2, "failed to create endpoint")

	runtime.ReleaseEndpoint(endpoint1)

	// use Eventually for async processing - need to access mocks.MetricsDataSource via runtime
	// This test would need modification to verify polling - for now just verify it works
	_ = endpoint2
}
