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

func TestFilter_AffinityThresholdDisabled(t *testing.T) {
	p := &Plugin{config: Config{AffinityThreshold: 0}}
	endpoints := []framework.Endpoint{
		makeEndpoint("a", 0, 10),
		makeEndpoint("b", 90, 20),
	}
	result := p.Filter(context.Background(), nil, nil, endpoints)
	assert.Equal(t, 2, len(result), "affinityThreshold=0 should return all")
}

func TestFilter_SingleEndpoint(t *testing.T) {
	p := &Plugin{config: Config{AffinityThreshold: 0.80}}
	endpoints := []framework.Endpoint{makeEndpoint("a", 90, 10)}
	result := p.Filter(context.Background(), nil, nil, endpoints)
	assert.Equal(t, 1, len(result), "single endpoint should always pass")
}

func TestFilter_NoStickyEndpoints(t *testing.T) {
	p := &Plugin{config: Config{AffinityThreshold: 0.80, ExplorationProbability: 0}}
	endpoints := []framework.Endpoint{
		makeEndpoint("a", 10, 10),
		makeEndpoint("b", 20, 20),
		makeEndpoint("c", 50, 30),
	}
	result := p.Filter(context.Background(), nil, nil, endpoints)
	assert.Equal(t, 3, len(result), "no sticky endpoints should return all")
}

func TestFilter_NarrowToSticky(t *testing.T) {
	p := &Plugin{config: Config{AffinityThreshold: 0.80, ExplorationProbability: 0, MaxTTFTPenaltyMs: 5000}}
	endpoints := []framework.Endpoint{
		makeEndpoint("a", 90, 100),
		makeEndpoint("b", 85, 120),
		makeEndpoint("c", 10, 50),
	}
	result := p.Filter(context.Background(), nil, nil, endpoints)
	assert.Equal(t, 2, len(result), "should narrow to sticky endpoints")
}

func TestFilter_TTFTPenaltyBreaksStickiness(t *testing.T) {
	p := &Plugin{config: Config{AffinityThreshold: 0.80, ExplorationProbability: 0, MaxTTFTPenaltyMs: 100}}
	endpoints := []framework.Endpoint{
		makeEndpoint("a", 90, 500),
		makeEndpoint("b", 10, 50),
	}
	result := p.Filter(context.Background(), nil, nil, endpoints)
	assert.Equal(t, 2, len(result), "TTFT penalty should break stickiness")
}

func TestFilter_ExplorationProbability(t *testing.T) {
	p := &Plugin{config: Config{AffinityThreshold: 0.80, ExplorationProbability: 1.0}}
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

func TestFactory_PartialConfigPreservesDefaults(t *testing.T) {
	// Setting only affinityThreshold should preserve defaults for other params.
	plugin, err := Factory("test", []byte(`{"affinityThreshold": 0.95}`), nil)
	assert.NoError(t, err)
	p := plugin.(*Plugin)
	assert.Equal(t, 0.95, p.config.AffinityThreshold)
	assert.Equal(t, DefaultConfig.ExplorationProbability, p.config.ExplorationProbability)
	assert.Equal(t, DefaultConfig.MaxTTFTPenaltyMs, p.config.MaxTTFTPenaltyMs)

	// Setting only explorationProbability should preserve defaults for other params.
	plugin, err = Factory("test", []byte(`{"explorationProbability": 0.05}`), nil)
	assert.NoError(t, err)
	p = plugin.(*Plugin)
	assert.Equal(t, DefaultConfig.AffinityThreshold, p.config.AffinityThreshold)
	assert.Equal(t, 0.05, p.config.ExplorationProbability)
	assert.Equal(t, DefaultConfig.MaxTTFTPenaltyMs, p.config.MaxTTFTPenaltyMs)

	// Setting only maxTTFTPenaltyMs should preserve defaults for other params.
	plugin, err = Factory("test", []byte(`{"maxTTFTPenaltyMs": 10000}`), nil)
	assert.NoError(t, err)
	p = plugin.(*Plugin)
	assert.Equal(t, DefaultConfig.AffinityThreshold, p.config.AffinityThreshold)
	assert.Equal(t, DefaultConfig.ExplorationProbability, p.config.ExplorationProbability)
	assert.Equal(t, float64(10000), p.config.MaxTTFTPenaltyMs)
}

func TestFactory_InvalidAffinityThreshold(t *testing.T) {
	_, err := Factory("test", []byte(`{"affinityThreshold": 1.5}`), nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "affinityThreshold must be <= 1.0")
}

func TestFactory_InvalidExplorationProbability(t *testing.T) {
	_, err := Factory("test", []byte(`{"explorationProbability": -0.1}`), nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "explorationProbability must be in [0, 1]")
}
