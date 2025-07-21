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

package datalayer_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
)

type dummy struct {
	Text string
}

func (d *dummy) Clone() datalayer.Cloneable {
	return &dummy{Text: d.Text}
}

func TestExpectPutThenGetToMatch(t *testing.T) {
	attrs := datalayer.NewAttributes()
	original := &dummy{"foo"}
	attrs.Put("a", original)

	got, ok := attrs.Get("a")
	assert.True(t, ok, "expected key to exist")
	assert.NotSame(t, original, got, "expected Get to return a clone, not original")

	dv, ok := got.(*dummy)
	assert.True(t, ok, "expected value to be of type *dummy")
	assert.Equal(t, "foo", dv.Text)
}

func TestExpectKeysToMatchAdded(t *testing.T) {
	attrs := datalayer.NewAttributes()
	attrs.Put("x", &dummy{"1"})
	attrs.Put("y", &dummy{"2"})

	keys := attrs.Keys()
	assert.Len(t, keys, 2)

	found := map[string]bool{}
	for _, k := range keys {
		found[k] = true
	}

	assert.True(t, found["x"])
	assert.True(t, found["y"])
}

func TestCloneReturnsCopy(t *testing.T) {
	original := datalayer.NewAttributes()
	original.Put("k", &dummy{"value"})

	cloned := original.Clone()

	gotOrig, _ := original.Get("k")
	gotClone, _ := cloned.Get("k")

	assert.NotSame(t, gotOrig, gotClone, "expected cloned value to be a different instance")

	dv, ok := gotClone.(*dummy)
	assert.True(t, ok)
	assert.Equal(t, "value", dv.Text)
}
