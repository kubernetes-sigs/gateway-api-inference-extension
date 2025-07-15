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

// Package types defines the core data structures and service contracts for the Flow Control system.
//
// It establishes the "vocabulary" of the system, defining the objects that are passed between the main controller,
// policies, and queue plugins. The central data model revolves around the lifecycle of a request, which is
// progressively wrapped in interfaces that provide an enriched, read-only view of its state.
package types
