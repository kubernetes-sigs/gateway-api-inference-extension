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
	"sync"
)

// Cloneable types support cloning of the value.
type Cloneable interface {
	Clone() Cloneable
}

// AttributeMap is used to store flexible metadata or traits
// across different aspects of an inference server.
// Stored values must be Cloneable.
type AttributeMap interface {
	Put(string, Cloneable)
	Get(string) (Cloneable, bool)
	Keys() []string
}

// Attributes provides a goroutine-safe implementation of AttributeMap.
type Attributes struct {
	data sync.Map // key: attribute name (string), value: attribute value (opaque, Cloneable)
}

// NewAttributes returns a new instance of Attributes.
func NewAttributes() *Attributes {
	return &Attributes{}
}

// Put adds or updates an attribute in the map.
func (a *Attributes) Put(key string, value Cloneable) {
	if value != nil {
		a.data.Store(key, value) // TODO: Clone into map to ensure isolation
	}
}

// Get retrieves an attribute by key, returning a cloned copy.
func (a *Attributes) Get(key string) (Cloneable, bool) {
	val, ok := a.data.Load(key)
	if !ok {
		return nil, false
	}
	if cloneable, ok := val.(Cloneable); ok {
		return cloneable.Clone(), true
	}
	return nil, false
}

// Keys returns all keys in the attribute map.
func (a *Attributes) Keys() []string {
	var keys []string
	a.data.Range(func(key, _ any) bool {
		if sk, ok := key.(string); ok {
			keys = append(keys, sk)
		}
		return true
	})
	return keys
}

// Clone creates a deep copy of the entire attribute map.
func (a *Attributes) Clone() *Attributes {
	clone := NewAttributes()
	a.data.Range(func(key, value any) bool {
		if sk, ok := key.(string); ok {
			if v, ok := value.(Cloneable); ok {
				clone.Put(sk, v)
			}
		}
		return true
	})
	return clone
}
