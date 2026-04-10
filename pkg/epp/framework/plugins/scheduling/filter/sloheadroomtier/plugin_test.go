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

package sloheadroomtier

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	framework "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrlatency "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/latency"
)

func makeEndpoint(name string, ttftHeadroom, tpotHeadroom float64, hasPrediction bool) framework.Endpoint {
	meta := &fwkdl.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: name, Namespace: "default"},
	}
	ep := framework.NewEndpoint(meta, &fwkdl.Metrics{}, fwkdl.NewAttributes())
	if hasPrediction {
		ep.Put(attrlatency.LatencyPredictionInfoKey, attrlatency.NewLatencyPredictionInfo(
			ttftHeadroom >= 0, tpotHeadroom >= 0, ttftHeadroom, tpotHeadroom, 100, 10, 0))
	}
	return ep
}

func TestFilter_SingleEndpoint(t *testing.T) {
	p := &Plugin{config: DefaultConfig}
	endpoints := []framework.Endpoint{makeEndpoint("a", 10, 5, true)}
	result := p.Filter(context.Background(), nil, nil, endpoints)
	assert.Equal(t, 1, len(result))
}

func TestFilter_NoPredictions(t *testing.T) {
	p := &Plugin{config: DefaultConfig}
	endpoints := []framework.Endpoint{
		makeEndpoint("a", 0, 0, false),
		makeEndpoint("b", 0, 0, false),
	}
	result := p.Filter(context.Background(), nil, nil, endpoints)
	assert.Equal(t, 2, len(result), "no predictions should keep all")
}

func TestFilter_AllPositive(t *testing.T) {
	p := &Plugin{config: Config{EpsilonExploreNeg: 0}}
	endpoints := []framework.Endpoint{
		makeEndpoint("a", 100, 50, true),
		makeEndpoint("b", 200, 80, true),
	}
	result := p.Filter(context.Background(), nil, nil, endpoints)
	assert.Equal(t, 2, len(result), "only positive tier, keep all")
}

func TestFilter_AllNegative(t *testing.T) {
	p := &Plugin{config: Config{EpsilonExploreNeg: 0}}
	endpoints := []framework.Endpoint{
		makeEndpoint("a", -100, -50, true),
		makeEndpoint("b", -200, -80, true),
	}
	result := p.Filter(context.Background(), nil, nil, endpoints)
	assert.Equal(t, 2, len(result), "only negative tier, keep all")
}

func TestFilter_BothTiers_SelectPositive(t *testing.T) {
	p := &Plugin{config: Config{EpsilonExploreNeg: 0}} // never explore
	endpoints := []framework.Endpoint{
		makeEndpoint("pos1", 100, 50, true),
		makeEndpoint("pos2", 200, 80, true),
		makeEndpoint("neg1", -100, -50, true),
	}
	result := p.Filter(context.Background(), nil, nil, endpoints)
	assert.Equal(t, 2, len(result), "should select positive tier")
	assert.Equal(t, "pos1", result[0].GetMetadata().NamespacedName.Name)
}

func TestFilter_BothTiers_EpsilonExploreNeg(t *testing.T) {
	p := &Plugin{config: Config{EpsilonExploreNeg: 1.0}} // always explore
	endpoints := []framework.Endpoint{
		makeEndpoint("pos1", 100, 50, true),
		makeEndpoint("neg1", -100, -50, true),
	}
	result := p.Filter(context.Background(), nil, nil, endpoints)
	assert.Equal(t, 1, len(result), "should select negative tier")
	assert.Equal(t, "neg1", result[0].GetMetadata().NamespacedName.Name)
}

func TestFilter_NoPredictionGoesToNegative(t *testing.T) {
	p := &Plugin{config: Config{EpsilonExploreNeg: 1.0}} // always explore neg
	endpoints := []framework.Endpoint{
		makeEndpoint("pos1", 100, 50, true),
		makeEndpoint("nopred", 0, 0, false),
	}
	result := p.Filter(context.Background(), nil, nil, endpoints)
	// nopred goes to negative, epsilon selects negative
	assert.Equal(t, 1, len(result))
	assert.Equal(t, "nopred", result[0].GetMetadata().NamespacedName.Name)
}

// Note: Deficit bucketing tests are in the slo-deficit-bucket-filter package.

func TestFactory_ValidConfig(t *testing.T) {
	plugin, err := Factory("test", nil, nil)
	assert.NoError(t, err)
	assert.NotNil(t, plugin)
}

func TestFactory_InvalidEpsilon(t *testing.T) {
	_, err := Factory("test", []byte(`{"epsilonExploreNeg": 1.5}`), nil)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "epsilonExploreNeg must be in [0, 1]")
}
