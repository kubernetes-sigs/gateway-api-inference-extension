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
)

func TestValidateExtractorType(t *testing.T) {
	type rawStruct struct{}
	type iface interface{ Foo() }

	tests := []struct {
		name   string
		output reflect.Type
		input  reflect.Type
		valid  bool
	}{
		{"exact match", typeOfTest(rawStruct{}), typeOfTest(rawStruct{}), true},
		{"input is interface{}", typeOfTest(rawStruct{}), typeOfTest((*any)(nil)), true},
		{"nil types are not allowed", typeOfTest(rawStruct{}), typeOfTest(nil), false},
		{"output does not implement input", typeOfTest(rawStruct{}), typeOfTest((*iface)(nil)), false},
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

func typeOfTest(v any) reflect.Type {
	t := reflect.TypeOf(v)
	if t == nil {
		return nil
	}
	if t.Kind() == reflect.Ptr {
		return t.Elem()
	}
	return t
}
