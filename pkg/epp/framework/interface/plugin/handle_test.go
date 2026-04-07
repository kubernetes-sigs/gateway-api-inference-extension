/*
Copyright 2026 The Kubernetes Authors.

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

package plugin_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	plugmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin/mocks"
)

func TestNewEppHandle_Metrics(t *testing.T) {
	recorder := &plugmocks.NoopMetricsRecorder{}
	handle := plugin.NewEppHandle(t.Context(), func() []types.NamespacedName { return nil }, recorder)

	assert.NotNil(t, handle.Metrics())
	assert.Equal(t, recorder, handle.Metrics())
}

func TestNewEppHandle_MetricsNil(t *testing.T) {
	handle := plugin.NewEppHandle(t.Context(), func() []types.NamespacedName { return nil }, nil)

	assert.Nil(t, handle.Metrics())
}
