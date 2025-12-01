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

package plugins

import "sigs.k8s.io/controller-runtime/pkg/manager"

// Plugin defines the interface for a plugin.
// This interface should be embedded in all plugins across the code.
type Plugin interface {
	// TypedName returns the type and name tuple of this plugin instance.
	TypedName() TypedName
}

// ConsumerPlugin defines the interface for a consumer.
type ConsumerPlugin interface {
	Plugin
	// Consumes returns data consumed by the plugin.
	// This is a map from data key (string) produced to
	// the data type of the key (represented as data with default value casted as any field).
	Consumes() map[string]any
}

// ProducerPlugin defines the interface for a producer.
type ProducerPlugin interface {
	Plugin
	// Produces returns data produced by the producer.
	// This is a map from data key (string) produced to
	// the data type of the key (represented as data with default value casted as any field).
	Produces() map[string]any
}

// RunnablePlugin defines the interface for a plugin that provides a controller-runtime Runnable.
type RunnablePlugin interface {
	GetRunnable() manager.Runnable
}
