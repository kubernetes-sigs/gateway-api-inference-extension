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

// Attributes provides a goroutine safe implementation of AttributeMap.
type Attributes struct {
	data sync.Map
}

// NewAttributes return a new attribute map instance.
func NewAttributes() *Attributes {
	return &Attributes{
		data: sync.Map{},
	}
}

// Put adds (or updates) an attribute in the map.
func (a *Attributes) Put(key string, value Cloneable) {
	a.data.Store(key, value) // TODO: Clone into map?
}

// Get returns an attribute from the map.
func (a *Attributes) Get(key string) (Cloneable, bool) {
	val, ok := a.data.Load(key)
	if !ok {
		return nil, false
	}
	if cloneable, ok := val.(Cloneable); ok {
		return cloneable.Clone(), true
	}
	return nil, false // shouldn't happen since Put accepts Cloneables only
}

// Keys returns an array of all the names of attributes stored in the map.
func (a *Attributes) Keys() []string {
	keys := []string{}
	a.data.Range(func(key, _ any) bool {
		if k, ok := key.(string); ok {
			keys = append(keys, k)
		}
		return true // continue iteration
	})
	return keys
}

// Clone the attributes object itself.
func (a *Attributes) Clone() *Attributes {
	cloned := &Attributes{
		data: sync.Map{},
	}

	a.data.Range(func(k, v interface{}) bool {
		cloned.data.Store(k, v)
		return true
	})
	return cloned
}
