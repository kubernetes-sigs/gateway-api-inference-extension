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
	"errors"
	"slices"
)

// buildDAG builds a dependency graph among data preparation plugins based on their
// produced and consumed data keys.
func buildDAG(plugins []PrepareDataPlugin) (map[string][]string, error) {
	dag := make(map[string][]string)
	for _, plugin := range plugins {
		dag[plugin.TypedName().String()] = []string{}
	}
	// Create dependency graph as a DAG.
	for i := range plugins {
		for j := range plugins {
			if i == j {
				continue
			}
			// Check whether plugin[i] produces something consumed by plugin[j]. In that case, j depends on i.
			if plugins[i].Produces() != nil && plugins[j].Consumes() != nil {
				for producedKey, producedData := range plugins[i].Produces() {
					// If plugin j consumes the produced key, then j depends on i. We can break after the first match.
					if consumedData, ok := plugins[j].Consumes()[producedKey]; ok {
						// Check types are same. Reflection is avoided here for simplicity.
						// TODO(#1985): Document this detail in IGW docs.
						if producedData != consumedData {
							return nil, errors.New("data type mismatch between produced and consumed data for key: " + producedKey)
						}
						iPluginName := plugins[i].TypedName().String()
						jPluginName := plugins[j].TypedName().String()
						dag[jPluginName] = append(dag[jPluginName], iPluginName)
						break
					}
				}
			}
		}
	}
	return dag, nil
}

// sortPlugins builds the dependency graph and returns the plugins ordered in topological order.
// If there is a cycle, it returns an error.
func sortPlugins(dag map[string][]string, plugins []PrepareDataPlugin) ([]PrepareDataPlugin, error) {
	nameToPlugin := map[string]PrepareDataPlugin{}
	for _, plugin := range plugins {
		nameToPlugin[plugin.TypedName().String()] = plugin
	}
	sortedPlugins, err := topologicalSort(dag)
	if err != nil {
		return nil, err
	}
	orderedPlugins := []PrepareDataPlugin{}
	for _, pluginName := range sortedPlugins {
		orderedPlugins = append(orderedPlugins, nameToPlugin[pluginName])
	}

	return orderedPlugins, err
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
	slices.Reverse(result)
	return result, nil
}
