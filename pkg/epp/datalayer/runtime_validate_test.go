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

package datalayer

import (
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/runtime/schema"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/source/mocks"
)

var (
	extractorType             = reflect.TypeOf((*fwkdl.Extractor)(nil)).Elem()
	notificationextractorType = reflect.TypeOf((*fwkdl.NotificationExtractor)(nil)).Elem()
)

func TestValidateInputTypeCompatible(t *testing.T) {
	type rawStruct struct{}
	type iface interface{ Foo() }

	tests := []struct {
		name   string
		output reflect.Type
		input  reflect.Type
		valid  bool
	}{
		{"exact match", typeOfT(rawStruct{}), typeOfT(rawStruct{}), true},
		{"input is interface{}", typeOfT(rawStruct{}), typeOfT((*any)(nil)), true},
		{"nil types are not allowed", typeOfT(rawStruct{}), typeOfT(nil), false},
		{"output does not implement input", typeOfT(rawStruct{}), typeOfT((*iface)(nil)), false},
	}

	for _, tt := range tests {
		err := validateInputTypeCompatible(tt.output, tt.input)
		if tt.valid {
			assert.NoError(t, err, "%s: expected valid extractor type", tt.name)
		} else {
			assert.Error(t, err, "%s: expected invalid extractor type", tt.name)
		}
	}
}

func TestValidateExtractorCompatible(t *testing.T) {
	type notExtractor struct{}

	tests := []struct {
		name         string
		extType      reflect.Type
		expectedType reflect.Type
		valid        bool
		errContains  string
	}{
		{
			name:         "nil extractor type",
			extType:      nil,
			expectedType: extractorType,
			valid:        false,
			errContains:  "can't be nil",
		},
		{
			name:         "nil expected type",
			extType:      reflect.TypeOf(&notExtractor{}),
			expectedType: nil,
			valid:        false,
			errContains:  "can't be nil",
		},
		{
			name:         "expected type not interface",
			extType:      reflect.TypeOf(&notExtractor{}),
			expectedType: reflect.TypeOf("string"),
			valid:        false,
			errContains:  "must be an interface",
		},
		{
			name:         "does not implement interface",
			extType:      reflect.TypeOf(&notExtractor{}),
			expectedType: extractorType,
			valid:        false,
			errContains:  "does not implement interface",
		},
	}

	for _, tt := range tests {
		err := validateExtractorCompatible(tt.extType, tt.expectedType)
		if tt.valid {
			assert.NoError(t, err, "%s: expected valid", tt.name)
		} else {
			assert.Error(t, err, "%s: expected error", tt.name)
			if tt.errContains != "" {
				assert.Contains(t, err.Error(), tt.errContains, "%s: error should contain", tt.name)
			}
		}
	}
}

func TestTypeConstants(t *testing.T) {
	assert.True(t, extractorType.Kind() == reflect.Interface, "extractorType should be an interface")
	assert.True(t, notificationextractorType.Kind() == reflect.Interface, "notificationextractorType should be an interface")
}

func typeOfT(v any) reflect.Type {
	t := reflect.TypeOf(v)
	if t == nil {
		return nil
	}
	if t.Kind() == reflect.Ptr {
		return t.Elem()
	}
	return t
}

func TestRuntimeConfigureWithNilExtractor(t *testing.T) {
	logger := newTestLogger(t)
	r := NewRuntime(1)

	cfg := &Config{
		Sources: []DataSourceConfig{
			{
				Plugin:     &mocks.MetricsDataSource{},
				Extractors: nil, // nil extractors should be allowed
			},
		},
	}

	err := r.Configure(cfg, false, "", logger)
	assert.NoError(t, err, "Configure should succeed with nil extractors")
}

func TestRuntimeConfigureDuplicateGVKFails(t *testing.T) {
	logger := newTestLogger(t)
	r := NewRuntime(1)

	// Create two notification sources with the same GVK
	gvk := schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"}
	src1 := mocks.NewNotificationSource("test", "source1", gvk)
	src2 := mocks.NewNotificationSource("test", "source2", gvk)

	cfg := &Config{
		Sources: []DataSourceConfig{
			{Plugin: src1, Extractors: nil},
			{Plugin: src2, Extractors: nil},
		},
	}

	err := r.Configure(cfg, false, "", logger)
	assert.Error(t, err, "Configure should fail with duplicate GVK")
	assert.Contains(t, err.Error(), "duplicate", "Error should mention duplicate GVK")
}
