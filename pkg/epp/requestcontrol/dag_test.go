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

package requestcontrol

import (
	"context"
	"maps"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

type mockPrepareRequestDataP struct {
	name     string
	produces map[string]any
	consumes map[string]any
}

func (m *mockPrepareRequestDataP) TypedName() plugins.TypedName {
	return plugins.TypedName{Name: m.name, Type: "mock"}
}

func (m *mockPrepareRequestDataP) Produces() map[string]any {
	return m.produces
}

func (m *mockPrepareRequestDataP) Consumes() map[string]any {
	return m.consumes
}

func (m *mockPrepareRequestDataP) PrepareRequestData(ctx context.Context, request *types.LLMRequest, pods []types.Pod) error {
	pods[0].Put(mockProducedDataKey, mockProducedDataType{value: 42})
	return nil
}

func TestPrepareDataGraph(t *testing.T) {
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

	testCases := []struct {
		name        string
		plugins     []PrepareDataPlugin
		expectedDAG map[string][]string
		expectError bool
	}{
		{
			name:        "No plugins",
			plugins:     []PrepareDataPlugin{},
			expectedDAG: map[string][]string{},
			expectError: false,
		},
		{
			name:    "Plugins with no dependencies",
			plugins: []PrepareDataPlugin{pluginA, pluginE},
			expectedDAG: map[string][]string{
				"A/mock": {},
				"E/mock": {},
			},
			expectError: false,
		},
		{
			name:    "Simple linear dependency (C -> B -> A)",
			plugins: []PrepareDataPlugin{pluginA, pluginB, pluginC},
			expectedDAG: map[string][]string{
				"A/mock": {},
				"B/mock": {"A/mock"},
				"C/mock": {"B/mock"},
			},
			expectError: false,
		},
		{
			name:    "DAG with multiple dependencies (B -> A, D -> A, E independent)",
			plugins: []PrepareDataPlugin{pluginA, pluginB, pluginD, pluginE},
			expectedDAG: map[string][]string{
				"A/mock": {},
				"B/mock": {"A/mock"},
				"D/mock": {"A/mock"},
				"E/mock": {},
			},
			expectError: false,
		},
		{
			name:        "Graph with a cycle (X -> Y, Y -> X)",
			plugins:     []PrepareDataPlugin{pluginX, pluginY},
			expectedDAG: nil,
			expectError: true,
		},
		{
			name:        "Data type mismatch between produced and consumed data",
			plugins:     []PrepareDataPlugin{pluginZ1, pluginZ2},
			expectedDAG: nil,
			expectError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			dag, err := buildDAG(tc.plugins)
			if err != nil {
				if tc.expectError {
					assert.Error(t, err)
					return
				}
			}
			orderedPlugins, err := sortPlugins(dag, tc.plugins)

			if tc.expectError {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "cycle detected")
				return
			}
			assert.NoError(t, err)

			// Normalize the slices in the maps for consistent comparison
			normalizedDAG := make(map[string][]string)
			maps.Copy(normalizedDAG, dag)
			normalizedExpectedDAG := make(map[string][]string)
			for k, v := range tc.expectedDAG {
				normalizedExpectedDAG[k] = v
			}

			if diff := cmp.Diff(normalizedExpectedDAG, normalizedDAG); diff != "" {
				t.Errorf("prepareDataGraph() mismatch (-want +got):\n%s", diff)
			}

			orderedPluginNames := make([]string, len(orderedPlugins))
			for i, p := range orderedPlugins {
				orderedPluginNames[i] = p.TypedName().String()
			}
			assertTopologicalOrder(t, dag, orderedPlugins)
		})
	}
}

func assertTopologicalOrder(t *testing.T, dag map[string][]string, ordered []PrepareDataPlugin) {
	t.Helper()
	positions := make(map[string]int)
	for i, p := range ordered {
		positions[p.TypedName().String()] = i
	}

	for node, dependencies := range dag {
		for _, dep := range dependencies {
			assert.Less(t, positions[dep], positions[node], "Dependency %s should come before %s", dep, node)
		}
	}
}
