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

package approximateprefix

import (
	"context"
	"testing"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwksched "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	attrprefix "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

func TestPrepareRequestData(t *testing.T) {
	config := Config{
		BlockSizeTokens:        1,
		MaxPrefixBlocksToMatch: DefaultMaxPrefixBlocks,
		LRUCapacityPerServer:   DefaultLRUCapacityPerServer,
	}
	// Test the "initialize if nil" pattern
	p, err := NewPrepareData(context.Background(), config, nil, nil, nil)
	assert.NoError(t, err)
	assert.NotNil(t, p.PluginState())

	endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1"}}, fwkdl.NewMetrics(), fwkdl.NewAttributes())
	endpoint2 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod2"}}, fwkdl.NewMetrics(), fwkdl.NewAttributes())
	endpoints := []fwksched.Endpoint{endpoint1, endpoint2}

	// First request to populate cache.
	req1 := &fwksched.LLMRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &fwksched.LLMRequestBody{
			Completions: &fwksched.CompletionsRequest{
				Prompt: "aaaabbbb",
			},
		},
	}

	// We need to simulate the PreRequest logic since PrepareRequestData only reads from the indexer.
	// But first let's see if PrepareRequestData correctly handles an empty indexer.
	err = p.PrepareRequestData(context.Background(), req1, endpoints)
	assert.NoError(t, err)

	// Verify state was written to PluginState
	state, err := plugin.ReadPluginStateKey[*SchedulingContextState](p.PluginState(), req1.RequestId, plugin.StateKey(attrprefix.PrefixCachePluginType))
	assert.NoError(t, err)
	assert.NotNil(t, state)
	assert.Equal(t, 2, len(state.PrefixHashes)) // "aaaabbbb" with blockSize 4 (1 token * 4 chars) -> 2 blocks
}

func TestPreRequest(t *testing.T) {
	config := Config{
		BlockSizeTokens:        1,
		MaxPrefixBlocksToMatch: DefaultMaxPrefixBlocks,
		LRUCapacityPerServer:   DefaultLRUCapacityPerServer,
	}
	p, _ := NewPrepareData(context.Background(), config, nil, nil, nil)

	endpoint1 := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod1", Namespace: "default"}}, fwkdl.NewMetrics(), fwkdl.NewAttributes())
	req1 := &fwksched.LLMRequest{
		RequestId:   uuid.NewString(),
		TargetModel: "test-model1",
		Body: &fwksched.LLMRequestBody{
			Completions: &fwksched.CompletionsRequest{
				Prompt: "aaaabbbb",
			},
		},
	}

	// 1. Prepare data (this saves state)
	_ = p.PrepareRequestData(context.Background(), req1, []fwksched.Endpoint{endpoint1})

	// 2. Simulate scheduling result
	res := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {
				TargetEndpoints: []fwksched.Endpoint{endpoint1},
			},
		},
	}

	// 3. Call PreRequest
	p.PreRequest(context.Background(), req1, res)

	// Wait for async update
	p.wg.Wait()

	// 4. Verify indexer was updated
	hashes := HashPrompt(context.Background(), req1, 4, DefaultMaxPrefixBlocks)
	for _, hash := range hashes {
		pods := p.indexer.Get(hash)
		assert.Contains(t, pods, ServerID(endpoint1.GetMetadata().NamespacedName))
	}
}

func TestPrepareDataValidation(t *testing.T) {
	validConfigs := []Config{{
		AutoTune:        false,
		BlockSizeTokens: 1,
	}, {
		AutoTune:        false,
		BlockSize:       1,
		BlockSizeTokens: 1,
	}, {
		AutoTune:        true,
		BlockSizeTokens: 0,
	}}
	invalidConfigs := []Config{{
		AutoTune:  false,
		BlockSize: 1,
	}, {
		AutoTune:        false,
		BlockSizeTokens: 0,
	}}

	for _, config := range validConfigs {
		_, err := NewPrepareData(context.Background(), config, nil, nil, nil)
		assert.NoError(t, err)
	}

	for _, config := range invalidConfigs {
		_, err := NewPrepareData(context.Background(), config, nil, nil, nil)
		assert.Error(t, err)
	}
}
