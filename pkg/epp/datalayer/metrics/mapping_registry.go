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

package metrics

import (
	"errors"
	"fmt"
	"sync"
)

const (
	// DefaultEngineType is the key for the fallback mapping.
	DefaultEngineType = "default"

	// EngineTypeLabelKey is the label on Pods that indicates the inference engine type.
	EngineTypeLabelKey = "inference.networking.k8s.io/engine-type"
)

// MappingRegistry holds multiple metric mappings for different inference engines.
type MappingRegistry struct {
	mappings sync.Map // key: engine type string, value: *Mapping
}

// NewMappingRegistry creates a new registry for metric mappings.
func NewMappingRegistry() *MappingRegistry {
	return &MappingRegistry{}
}

// Register adds a mapping for a specific engine type.
// It returns an error if the engine type is empty or if the mapping is nil.
func (r *MappingRegistry) Register(engineType string, mapping *Mapping) error {
	if engineType == "" {
		return errors.New("engine type cannot be empty")
	}
	if mapping == nil {
		return errors.New("mapping cannot be nil")
	}

	if _, loaded := r.mappings.LoadOrStore(engineType, mapping); loaded {
		return fmt.Errorf("mapping for engine type %q already exists", engineType)
	}
	return nil
}

// Get finds the mapping for the given engine type, falling back to "default" if not found.
func (r *MappingRegistry) Get(engineType string) (*Mapping, bool) {
	if engineType == "" {
		engineType = DefaultEngineType
	}

	// Direct lookup
	if val, ok := r.mappings.Load(engineType); ok {
		return val.(*Mapping), true
	}

	// Fallback to default
	if engineType != DefaultEngineType {
		if val, ok := r.mappings.Load(DefaultEngineType); ok {
			return val.(*Mapping), true
		}
	}

	return nil, false
}

// ListEngineTypes returns a list of all registered engine types.
func (r *MappingRegistry) ListEngineTypes() []string {
	var result []string
	r.mappings.Range(func(key, _ any) bool {
		result = append(result, key.(string))
		return true
	})
	return result
}
