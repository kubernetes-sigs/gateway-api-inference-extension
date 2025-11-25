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
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
)

type dummy struct {
	Text string
}

func (d *dummy) Clone() Cloneable {
	return &dummy{Text: d.Text}
}

func TestExpectPutThenGetToMatch(t *testing.T) {
	attrs := NewAttributes()
	original := &dummy{"foo"}
	attrs.Put("a", original)

	got, ok := attrs.Get("a")
	assert.True(t, ok, "expected key to exist")
	assert.NotSame(t, original, got, "expected Get to return a clone, not original")

	dv, ok := got.(*dummy)
	assert.True(t, ok, "expected value to be of type *dummy")
	assert.Equal(t, "foo", dv.Text)

	_, ok = attrs.Get("b")
	assert.False(t, ok, "expected key not to exist")
}

func TestExpectKeysToMatchAdded(t *testing.T) {
	attrs := NewAttributes()
	attrs.Put("x", &dummy{"1"})
	attrs.Put("y", &dummy{"2"})

	keys := attrs.Keys()
	assert.Len(t, keys, 2)
	assert.ElementsMatch(t, keys, []string{"x", "y"})
}

func TestCloneReturnsCopy(t *testing.T) {
	original := NewAttributes()
	original.Put("k", &dummy{"value"})

	cloned := original.Clone()

	kOrig, _ := original.Get("k")
	kClone, _ := cloned.Get("k")

	assert.NotSame(t, kOrig, kClone, "expected cloned value to be a different instance")
	if diff := cmp.Diff(kOrig, kClone); diff != "" {
		t.Errorf("Unexpected output (-want +got): %v", diff)
	}
}
