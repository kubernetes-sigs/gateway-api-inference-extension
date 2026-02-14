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
	"errors"
	"slices"

	fwkfc "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/flowcontrol"
	fwkrq "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requestcontrol"
	fwksch "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

// ValidateAndOrderDataDependencies validates that the data dependencies among the given plugins are acyclic
// and returns a topologically sorted order of plugin names based on their data dependencies.
// Further, it validates that the plugins are ordered in a way that respects the layer execution order.
func ValidateAndOrderDataDependencies(plugins []plugin.Plugin) ([]string, error) {
	pluginMap := make(map[string]plugin.Plugin)
	for _, p := range plugins {
		pluginMap[p.TypedName().String()] = p
	}
	for _, p := range plugins {
		pluginMap[p.TypedName().String()] = p
	}
	producers := make(map[string]plugin.ProducerPlugin)
	consumers := make(map[string]plugin.ConsumerPlugin)
	for name, p := range pluginMap {
		if producer, ok := p.(plugin.ProducerPlugin); ok {
			producers[name] = producer
		}
		if consumer, ok := p.(plugin.ConsumerPlugin); ok {
			consumers[name] = consumer
		}
	}
	dag, err := buildDAG(producers, consumers)
	if err != nil {
		return nil, err
	}
	// Topologically sort the DAG to determine the order of plugin execution.
	pluginNames, err := topologicalSort(dag)
	if err != nil {
		return nil, err
	}

	return pluginNames, nil
}

// Define constants for layer execution order. Lower value means earlier execution.
const (
	FlowControlLayer    = 0
	RequestControlLayer = 1
	SchedulingLayer     = 2
	DefaultLayer        = -1 // For plugins that don't fit into a known layer
)

func pluginToLayerExecutionOrder(plugin plugin.Plugin) int {
	// Flow control plugins
	if _, ok := plugin.(fwkfc.FairnessPolicy); ok {
		return FlowControlLayer
	}
	if _, ok := plugin.(fwkfc.OrderingPolicy); ok {
		return FlowControlLayer
	}

	// Request control plugins
	if _, ok := plugin.(fwkrq.PrepareDataPlugin); ok {
		return RequestControlLayer
	}
	if _, ok := plugin.(fwkrq.AdmissionPlugin); ok {
		return RequestControlLayer
	}
	if _, ok := plugin.(fwkrq.PreRequest); ok {
		return RequestControlLayer
	}
	if _, ok := plugin.(fwkrq.ResponseReceived); ok {
		return RequestControlLayer
	}

	// Scheduling plugins
	if _, ok := plugin.(fwksch.ProfileHandler); ok {
		return SchedulingLayer
	}
	if _, ok := plugin.(fwksch.Filter); ok {
		return SchedulingLayer
	}
	if _, ok := plugin.(fwksch.Scorer); ok {
		return SchedulingLayer
	}
	if _, ok := plugin.(fwksch.Picker); ok {
		return SchedulingLayer
	}

	// If the plugin doesn't match any known layer, return -1.
	return DefaultLayer
}

// buildDAG builds a dependency graph among data preparation plugins based on their
// produced and consumed data keys.
func buildDAG(producers map[string]plugin.ProducerPlugin, consumers map[string]plugin.ConsumerPlugin) (map[string][]string, error) {
	dag := make(map[string][]string)
	// Create dependency graph as a DAG.
	for _, producer := range producers {
		dag[producer.TypedName().String()] = []string{}
	}
	for _, consumer := range consumers {
		dag[consumer.TypedName().String()] = []string{}
	}
	for pName, producer := range producers {
		for cName, consumer := range consumers {
			if pName == cName {
				continue
			}
			if producer.Produces() != nil && consumer.Consumes() != nil {
				for producedKey, producedData := range producer.Produces() {
					if consumedData, ok := consumer.Consumes()[producedKey]; ok {
						// Check types are same. Reflection is avoided here for simplicity.
						// TODO(#1985): Document this detail in IGW docs.
						if producedData != consumedData {
							return nil, errors.New("data type mismatch between produced and consumed data for key: " + producedKey)
						}
						if pluginToLayerExecutionOrder(producer) > pluginToLayerExecutionOrder(consumer) {
							return nil, errors.New("invalid plugin layer execution order: producer " + pName + " needs to be executed before consumer " + cName)
						}
						// Consumer depends on producer, so add an edge from consumer to producer.
						dag[cName] = append(dag[cName], pName)
						break
					}
				}
			}
		}
	}
	return dag, nil
}

// TopologicalSort performs Kahn's Algorithm on a DAG.
// It returns the sorted order or an error if a cycle is detected.
func topologicalSort(graph map[string][]string) ([]string, error) {
	// 1. Initialize in-degree map
	inDegree := make(map[string]int)

	// Ensure all nodes are present in the inDegree map, even those with no dependencies
	for u, neighbors := range graph {
		if _, ok := inDegree[u]; !ok {
			inDegree[u] = 0
		}
		for _, v := range neighbors {
			inDegree[v]++ // Increment in-degree for the destination node
		}
	}

	// 2. Initialize the queue with nodes having 0 in-degree
	var queue []string
	for node, degree := range inDegree {
		if degree == 0 {
			queue = append(queue, node)
		}
	}

	var result []string

	// 3. Process the queue
	for len(queue) > 0 {
		// Dequeue
		u := queue[0]
		queue = queue[1:]

		result = append(result, u)

		// Decrease in-degree of neighbors
		if neighbors, ok := graph[u]; ok {
			for _, v := range neighbors {
				inDegree[v]--
				if inDegree[v] == 0 {
					queue = append(queue, v)
				}
			}
		}
	}

	// 4. Check for cycles
	// If the result size != total nodes, there is a cycle
	if len(result) != len(inDegree) {
		return nil, errors.New("cycle detected: graph is not a DAG")
	}

	// Reverse to get the correct order since edges point from consumer to producer
	slices.Reverse(result)
	return result, nil
}
