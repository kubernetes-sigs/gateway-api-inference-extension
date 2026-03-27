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

package latencypredictor

import (
	"testing"

	"github.com/stretchr/testify/assert"

	attrlatency "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/latency"
	attrprefix "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

func TestProducesConsumes(t *testing.T) {
	pl := NewPredictedLatency(DefaultConfig, nil)

	produces := pl.Produces()
	assert.Contains(t, produces, attrlatency.LatencyPredictionInfoKey)

	consumes := pl.Consumes()
	assert.Contains(t, consumes, attrprefix.PrefixCacheMatchInfoKey)
}
