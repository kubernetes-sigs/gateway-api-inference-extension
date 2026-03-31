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

package prefixcacheaffinity

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrlatency "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/latency"
	attrprefix "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

// makeEndpoint creates a test endpoint with the given prefix cache match ratio
// (prefixMatch out of 100 total blocks) and predicted TTFT.
func makeEndpoint(name string, prefixMatch int, ttft float64) framework.Endpoint {
	meta := &fwkdl.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: name, Namespace: "default"},
	}
	ep := framework.NewEndpoint(meta, &fwkdl.Metrics{}, fwkdl.NewAttributes())
	if prefixMatch >= 0 {
		ep.Put(attrprefix.PrefixCacheMatchInfoKey, attrprefix.NewPrefixCacheMatchInfo(prefixMatch, 100, 16))
	}
	if ttft >= 0 {
		ep.Put(attrlatency.LatencyPredictionInfoKey, attrlatency.NewLatencyPredictionInfo(true, true, 0, 0, ttft, 0))
	}
	return ep
}

func TestFilter_TauDisabled(t *testing.T) {
	p := &Plugin{config: Config{Tau: 0}}
	endpoints := []framework.Endpoint{
		makeEndpoint("a", 0, 10),
		makeEndpoint("b", 90, 20),
	}
	result := p.Filter(context.Background(), nil, nil, endpoints)
	assert.Equal(t, 2, len(result), "tau=0 should return all")
}

func TestFilter_SingleEndpoint(t *testing.T) {
	p := &Plugin{config: Config{Tau: 0.80}}
	endpoints := []framework.Endpoint{makeEndpoint("a", 90, 10)}
	result := p.Filter(context.Background(), nil, nil, endpoints)
	assert.Equal(t, 1, len(result), "single endpoint should always pass")
}

func TestFilter_NoStickyEndpoints(t *testing.T) {
	p := &Plugin{config: Config{Tau: 0.80, EpsilonExplore: 0}}
	endpoints := []framework.Endpoint{
		makeEndpoint("a", 10, 10),
		makeEndpoint("b", 20, 20),
		makeEndpoint("c", 50, 30),
	}
	result := p.Filter(context.Background(), nil, nil, endpoints)
	assert.Equal(t, 3, len(result), "no sticky endpoints should return all")
}

func TestFilter_NarrowToSticky(t *testing.T) {
	p := &Plugin{config: Config{Tau: 0.80, EpsilonExplore: 0, MaxTTFTPenaltyMs: 5000}}
	endpoints := []framework.Endpoint{
		makeEndpoint("a", 90, 100),
		makeEndpoint("b", 85, 120),
		makeEndpoint("c", 10, 50),
	}
	result := p.Filter(context.Background(), nil, nil, endpoints)
	assert.Equal(t, 2, len(result), "should narrow to sticky endpoints")
}

func TestFilter_TTFTPenaltyBreaksStickiness(t *testing.T) {
	p := &Plugin{config: Config{Tau: 0.80, EpsilonExplore: 0, MaxTTFTPenaltyMs: 100}}
	endpoints := []framework.Endpoint{
		makeEndpoint("a", 90, 500),
		makeEndpoint("b", 10, 50),
	}
	result := p.Filter(context.Background(), nil, nil, endpoints)
	assert.Equal(t, 2, len(result), "TTFT penalty should break stickiness")
}

func TestFilter_EpsilonExplore(t *testing.T) {
	p := &Plugin{config: Config{Tau: 0.80, EpsilonExplore: 1.0}}
	endpoints := []framework.Endpoint{
		makeEndpoint("a", 90, 100),
		makeEndpoint("b", 10, 50),
	}
	result := p.Filter(context.Background(), nil, nil, endpoints)
	assert.Equal(t, 2, len(result), "epsilon=1.0 should always skip gate")
}

func TestFactory_ValidConfig(t *testing.T) {
	plugin, err := Factory("test", nil, nil)
	assert.NoError(t, err)
	assert.NotNil(t, plugin)
	assert.Equal(t, PluginType, plugin.TypedName().Type)
}

func TestFactory_InvalidTau(t *testing.T) {
	_, err := Factory("test", []byte(`{"tau": 1.5}`), nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "tau must be <= 1.0")
}

func TestFactory_InvalidEpsilon(t *testing.T) {
	_, err := Factory("test", []byte(`{"epsilonExplore": -0.1}`), nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "epsilonExplore must be in [0, 1]")
}
