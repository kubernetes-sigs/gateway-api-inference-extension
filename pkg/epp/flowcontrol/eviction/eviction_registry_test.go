/*
Copyright 2026 The Kubernetes Authors.

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

package eviction

import (
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestEvictionRegistry_RegisterAndGet(t *testing.T) {
	t.Parallel()
	r := NewEvictionRegistry()

	ch := make(chan struct{})
	r.Register("req-1", ch)

	got := r.Get("req-1")
	assert.Equal(t, ch, got, "Get should return the registered channel")

	assert.Nil(t, r.Get("non-existent"), "Get for unregistered ID should return nil")
}

func TestEvictionRegistry_Deregister(t *testing.T) {
	t.Parallel()
	r := NewEvictionRegistry()

	ch := make(chan struct{})
	r.Register("req-1", ch)
	r.Deregister("req-1")

	assert.Nil(t, r.Get("req-1"), "Get after Deregister should return nil")

	// Deregister non-existent should not panic.
	r.Deregister("non-existent")
}

func TestEvictionRegistry_Concurrency(t *testing.T) {
	t.Parallel()
	r := NewEvictionRegistry()

	const goroutines = 10
	const opsPerGoroutine = 100

	var wg sync.WaitGroup
	wg.Add(goroutines)

	for g := range goroutines {
		go func(id int) {
			defer wg.Done()
			for i := range opsPerGoroutine {
				reqID := fmt.Sprintf("req-%d-%d", id, i)
				ch := make(chan struct{})

				switch i % 3 {
				case 0:
					r.Register(reqID, ch)
				case 1:
					r.Register(reqID, ch)
					r.Get(reqID)
					r.Deregister(reqID)
				case 2:
					r.Get(reqID)
				}
			}
		}(g)
	}

	wg.Wait()
}
