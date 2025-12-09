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

// Package plugins provides the core extensibility framework for the Endpoint Picker (EPP).
// It enables the system to be composed of modular, pluggable components that handle everything from
// request scheduling to flow control and data enrichment.
//
// # Core Concepts
//
// 1. Registry & Lifecycle
//
// Plugins are registered globally via the Registry. The system supports two lifecycles:
//
//   - Singleton (Default): Instantiated once at startup.
//   - Transient: Instantiated on-demand at runtime via a Factory.
//
// 2. Configuration (Handle)
//
// The Handle serves as the bridge between the configuration (YAML) and the runtime.
// It holds "Blueprints" (config for Transient plugins) and references to "Instances" (active
// Singleton plugins).
//
// 3. Factory
//
// The PluginFactory is the mechanism for creating Transient plugins. It resolves a configuration
// blueprint from the Handle and instantiates a new, distinct plugin instance, allowing for unique
// runtime identities (e.g., "tenant-a-queue").
//
// Architectural Distinction: The DAG vs. Flow Control
//
// The EPP Request Control DAG operates exclusively on Singleton plugins.
// It relies on a static topological sort performed at startup.
//
// Transient plugins (LifecycleTransient) do NOT participate in the global Request Control DAG.
// They live inside specific subsystems (like the Flow Control Layer) and have their own independent
// lifecycles managed by that subsystem. They are not accessible via handle.GetAllPlugins() and
// cannot be configured as dependencies for PrepareData plugins.
package plugins
