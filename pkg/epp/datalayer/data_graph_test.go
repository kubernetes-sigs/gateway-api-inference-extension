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
	"maps"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkfcmocks "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol/mocks"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkrc "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	fwksch "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

const mockProducedDataKey = "mockProducedData"

type mockPrepareRequestDataP struct {
	name     string
	produces map[string]any
	consumes map[string]any
}

type mockProducedDataType struct {
	value int
}

func (m *mockProducedDataType) Clone() fwkdl.Cloneable {
	return &mockProducedDataType{value: m.value}
}

func (m *mockPrepareRequestDataP) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Name: m.name, Type: "mock"}
}

func (m *mockPrepareRequestDataP) Produces() map[string]any {
	return m.produces
}

func (m *mockPrepareRequestDataP) Consumes() map[string]any {
	return m.consumes
}

func (m *mockPrepareRequestDataP) PrepareRequestData(ctx context.Context, request *fwksch.InferenceRequest, endpoints []fwksch.Endpoint) error {
	endpoints[0].Put(mockProducedDataKey, &mockProducedDataType{value: 42})
	return nil
}

type MockConsumerFairnessPolicy struct {
	fwkfcmocks.MockFairnessPolicy
	consumes map[string]any
}

func (m *MockConsumerFairnessPolicy) Consumes() map[string]any {
	return m.consumes
}

type MockSchedulingPlugin struct {
	fwksch.Scorer
	consumes map[string]any
}

func (m *MockSchedulingPlugin) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Name: "MockSchedulingPlugin", Type: "mock"}
}

func (m *MockSchedulingPlugin) Consumes() map[string]any {
	return m.consumes
}

func TestValidatePluginExecutionOrder(t *testing.T) {
	// Request control plugin that produces data.
	pluginA := &mockPrepareRequestDataP{name: "A", produces: map[string]any{"keyA": nil}}
	// Flow control plugin.
	consumerFairnessPolicyPlugin := MockConsumerFairnessPolicy{consumes: map[string]any{"keyA": nil}}
	// Scheduling plugin.
	consumerSchedulingPlugin := MockSchedulingPlugin{consumes: map[string]any{"keyA": nil}}
	if _, ok := any(pluginA).(fwkrc.PrepareDataPlugin); !ok {
		t.Fatalf("pluginA should implement PrepareDataPlugin")
	}

	testCases := []struct {
		name        string
		plugins     []fwkplugin.Plugin
		expectedErr string
	}{
		{
			name:        "Plugins with no dependencies",
			plugins:     []fwkplugin.Plugin{pluginA},
			expectedErr: "",
		},
		{
			name:        "FC depends on a request control plugin (invalid layer execution order)",
			plugins:     []fwkplugin.Plugin{pluginA, &consumerFairnessPolicyPlugin},
			expectedErr: "invalid plugin layer execution order",
		},
		{
			name:        "Scheduling plugin depends on a request control plugin",
			plugins:     []fwkplugin.Plugin{pluginA, &consumerSchedulingPlugin},
			expectedErr: "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := ValidateAndOrderDataDependencies(tc.plugins)
			if tc.expectedErr != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tc.expectedErr)
				return
			}
			assert.NoError(t, err)
		})
	}
}

func TestDAGAndTopologicalOrder(t *testing.T) {
	pluginA := &mockPrepareRequestDataP{name: "A", produces: map[string]any{"keyA": nil}}
	pluginB := &mockPrepareRequestDataP{name: "B", consumes: map[string]any{"keyA": nil}, produces: map[string]any{"keyB": nil}}
	pluginC := &mockPrepareRequestDataP{name: "C", consumes: map[string]any{"keyB": nil}}
	pluginD := &mockPrepareRequestDataP{name: "D", consumes: map[string]any{"keyA": nil}}
	pluginE := &mockPrepareRequestDataP{name: "E"} // No dependencies

	// Cycle plugins
	pluginX := &mockPrepareRequestDataP{name: "X", produces: map[string]any{"keyX": nil}, consumes: map[string]any{"keyY": nil}}
	pluginY := &mockPrepareRequestDataP{name: "Y", produces: map[string]any{"keyY": nil}, consumes: map[string]any{"keyX": nil}}

	// Data type mismatch plugin.
	pluginZ1 := &mockPrepareRequestDataP{name: "Z1", produces: map[string]any{"keyZ": int(0)}}
	pluginZ2 := &mockPrepareRequestDataP{name: "Z2", consumes: map[string]any{"keyZ": string("")}}

	// Same type different pointers.
	pluginP1 := &mockPrepareRequestDataP{name: "P1", produces: map[string]any{"keyP": &mockProducedDataType{}}}
	pluginP2 := &mockPrepareRequestDataP{name: "P2", consumes: map[string]any{"keyP": &mockProducedDataType{}}}

	testCases := []struct {
		name        string
		plugins     []fwkrc.PrepareDataPlugin
		expectedDAG map[string][]string
		expectedErr string
	}{
		{
			name:        "No plugins",
			plugins:     []fwkrc.PrepareDataPlugin{},
			expectedDAG: map[string][]string{},
			expectedErr: "",
		},
		{
			name:    "Plugins with no dependencies",
			plugins: []fwkrc.PrepareDataPlugin{pluginA, pluginE},
			expectedDAG: map[string][]string{
				"A/mock": {},
				"E/mock": {},
			},
			expectedErr: "",
		},
		{
			name:    "Simple linear dependency (C -> B -> A)",
			plugins: []fwkrc.PrepareDataPlugin{pluginA, pluginB, pluginC},
			expectedDAG: map[string][]string{
				"A/mock": {},
				"B/mock": {"A/mock"},
				"C/mock": {"B/mock"},
			},
			expectedErr: "",
		},
		{
			name:    "DAG with multiple dependencies (B -> A, D -> A, E independent)",
			plugins: []fwkrc.PrepareDataPlugin{pluginA, pluginB, pluginD, pluginE},
			expectedDAG: map[string][]string{
				"A/mock": {},
				"B/mock": {"A/mock"},
				"D/mock": {"A/mock"},
				"E/mock": {},
			},
			expectedErr: "",
		},
		{
			name:        "Graph with a cycle (X -> Y, Y -> X)",
			plugins:     []fwkrc.PrepareDataPlugin{pluginX, pluginY},
			expectedDAG: nil,
			expectedErr: "cycle detected",
		},
		{
			name:        "Data type mismatch between produced and consumed data",
			plugins:     []fwkrc.PrepareDataPlugin{pluginZ1, pluginZ2},
			expectedDAG: nil,
			expectedErr: "data type mismatch between produced and consumed data",
		},
		{
			name:    "Same type different pointers (should succeed)",
			plugins: []fwkrc.PrepareDataPlugin{pluginP1, pluginP2},
			expectedDAG: map[string][]string{
				"P1/mock": {},
				"P2/mock": {"P1/mock"},
			},
			expectedErr: "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			producers := make(map[string]fwkplugin.ProducerPlugin)
			consumers := make(map[string]fwkplugin.ConsumerPlugin)
			for _, p := range tc.plugins {
				if pp, ok := p.(fwkplugin.ProducerPlugin); ok {
					producers[p.TypedName().String()] = pp
				}
				if cp, ok := p.(fwkplugin.ConsumerPlugin); ok {
					consumers[p.TypedName().String()] = cp
				}
			}
			dag, err := buildDAG(producers, consumers)
			if err != nil {
				if tc.expectedErr != "" {
					assert.Error(t, err)
					assert.Contains(t, err.Error(), tc.expectedErr)
					return
				}
				assert.NoError(t, err)
			}
			orderedPlugins, err := topologicalSort(dag)

			if tc.expectedErr != "" {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tc.expectedErr)
				return
			}
			assert.NoError(t, err)

			// Normalize the slices in the maps for consistent comparison
			normalizedDAG := make(map[string][]string)
			maps.Copy(normalizedDAG, dag)
			normalizedExpectedDAG := make(map[string][]string)
			maps.Copy(normalizedExpectedDAG, tc.expectedDAG)

			if diff := cmp.Diff(normalizedExpectedDAG, normalizedDAG); diff != "" {
				t.Errorf("prepareDataGraph() mismatch (-want +got):\n%s", diff)
			}

			assertTopologicalOrder(t, dag, orderedPlugins)
		})
	}
}

func assertTopologicalOrder(t *testing.T, dag map[string][]string, ordered []string) {
	t.Helper()
	positions := make(map[string]int)
	for i, p := range ordered {
		positions[p] = i
	}

	for node, dependencies := range dag {
		for _, dep := range dependencies {
			assert.Less(t, positions[dep], positions[node], "Dependency %s should come before %s", dep, node)
		}
	}
}
