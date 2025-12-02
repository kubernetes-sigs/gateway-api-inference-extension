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
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
)

const (
	testType = "test-ds-type"
)

func TestRegisterAndGetSource(t *testing.T) {
	reg := DataSourceRegistry{}
	ds := &FakeDataSource{typedName: &plugins.TypedName{Type: testType, Name: testType}}

	err := reg.Register(ds)
	assert.NoError(t, err, "expected no error on first registration")

	// Duplicate registration
	err = reg.Register(ds)
	assert.Error(t, err, "expected error on duplicate registration")

	err = reg.Register(nil)
	assert.Error(t, err, "expected error on nil")

	// Get all sources
	all := reg.GetSources()
	assert.Len(t, all, 1)
	assert.Equal(t, testType, all[0].TypedName().Type)

	// Default registry
	err = RegisterSource(ds)
	assert.NoError(t, err, "expected no error on registration")

	// Get all sources
	all = GetSources()
	assert.Len(t, all, 1)
	assert.Equal(t, testType, all[0].TypedName().Type)
}

func TestGetSourceWhenNoneAreRegistered(t *testing.T) {
	reg := DataSourceRegistry{}
	found := reg.GetSources()
	assert.Empty(t, found, "expected no sources to be returned")
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
