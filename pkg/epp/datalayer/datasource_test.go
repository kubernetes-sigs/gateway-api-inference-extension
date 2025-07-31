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
	"context"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
)

type mockDataSource struct {
	name string
}

func (m *mockDataSource) Name() string                                { return m.name }
func (m *mockDataSource) AddExtractor(_ Extractor) error              { return nil }
func (m *mockDataSource) Collect(_ context.Context, _ Endpoint) error { return nil }

func TestRegisterAndGetSource(t *testing.T) {
	reg := DataSourceRegistry{}
	ds := &mockDataSource{name: "test"}

	err := reg.Register(ds)
	assert.NoError(t, err, "expected no error on first registration")

	// Duplicate registration
	err = reg.Register(ds)
	assert.Error(t, err, "expected error on duplicate registration")

	// Get by name
	got, found := reg.GetNamedSource("test")
	assert.True(t, found, "expected to find registered data source")
	assert.Equal(t, "test", got.Name())

	// Get all sources
	all := reg.GetSources()
	assert.Len(t, all, 1)
	assert.Equal(t, "test", all[0].Name())
}

func TestGetNamedSourceWhenNotFound(t *testing.T) {
	reg := DataSourceRegistry{}
	_, found := reg.GetNamedSource("missing")
	assert.False(t, found, "expected source to be missing")
}

func TestValidateExtractorType(t *testing.T) {
	type rawStruct struct{}
	type iface interface{ Foo() }

	tests := []struct {
		name   string
		output reflect.Type
		input  reflect.Type
		valid  bool
	}{
		{"exact match", typeOf(rawStruct{}), typeOf(rawStruct{}), true},
		{"input is interface{}", typeOf(rawStruct{}), typeOf((*any)(nil)), true},
		{"nil types are not allowed", typeOf(rawStruct{}), typeOf(nil), false},
		{"output does not implement input", typeOf(rawStruct{}), typeOf((*iface)(nil)), false},
	}

	for _, tt := range tests {
		err := ValidateExtractorType(tt.output, tt.input)
		if tt.valid {
			assert.NoError(t, err, "%s: expected valid extractor type", tt.name)
		} else {
			assert.Error(t, err, "%s: expected invalid extractor type", tt.name)
		}
	}
}

// typeOf returns the reflect.Type of a value.
// If the value is a pointer (e.g., (*iface)(nil)), it returns the pointed-to type (via Elem()).
// Otherwise, it returns the type as-is.
func typeOf(v any) reflect.Type {
	t := reflect.TypeOf(v)
	if t == nil {
		return nil
	}
	if t.Kind() == reflect.Ptr {
		return t.Elem()
	}
	return t
}
