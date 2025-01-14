/*
Copyright 2024 The Kubernetes Authors.

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

package crd_test

import (
	"context"
	"os"
	"testing"

	"github.com/stretchr/testify/mock"
	"github.com/stretchr/testify/require"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/crd"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/crd/mocks"
)

func createTempFile(t *testing.T, dir, content, pattern string) {
	tempFile, err := os.CreateTemp(dir, pattern)
	require.NoError(t, err)

	_, err = tempFile.WriteString(content)
	require.NoError(t, err)
	tempFile.Close()
}

func TestInstallCRDs(t *testing.T) {
	ctx := context.TODO()
	mockClient := new(mocks.MockClient)

	// Mock calls for valid CRDs
	mockClient.On("Create", mock.Anything, mock.Anything, mock.Anything).Return(nil)

	// Valid CRD content
	validCRD := `apiVersion: apiextensions.k8s.io/v1
kind: CustomResourceDefinition
metadata:
  name: valid-crd.example.com
spec:
  group: example.com
  names:
    kind: ValidCrd
    listKind: ValidCrdList
    plural: validcrds
    singular: validcrd
  scope: Namespaced
  versions:
  - name: v1
    served: true
    storage: true`

	// Non-CRD YAML content
	nonCRDYAML := `apiVersion: v1
kind: ConfigMap
metadata:
  name: test-configmap`

	// Invalid file content
	invalidContent := "invalid content"

	tests := []struct {
		description string
		setup       func(dir string)
		expectError bool
	}{
		{
			description: "Directory with valid CRD file",
			setup: func(dir string) {
				createTempFile(t, dir, validCRD, "valid-*.yaml")
			},
			expectError: false,
		},
		{
			description: "Directory with invalid content file",
			setup: func(dir string) {
				createTempFile(t, dir, invalidContent, "invalid-*.yaml")
			},
			expectError: true,
		},
		{
			description: "Directory with non-CRD YAML file",
			setup: func(dir string) {
				createTempFile(t, dir, nonCRDYAML, "non-crd-*.yaml")
			},
			expectError: false,
		},
		{
			description: "Directory with mixed valid and invalid files",
			setup: func(dir string) {
				createTempFile(t, dir, validCRD, "valid-*.yaml")
				createTempFile(t, dir, invalidContent, "invalid-*.yaml")
			},
			expectError: true,
		},
		{
			description: "Empty directory",
			setup:       func(dir string) {}, // No files created
			expectError: false,
		},
	}

	for _, test := range tests {
		t.Run(test.description, func(t *testing.T) {
			tempDir := t.TempDir()
			test.setup(tempDir)

			err := crd.InstallCRDs(ctx, mockClient, tempDir)
			if test.expectError {
				require.Error(t, err, "Expected an error but got nil")
			} else {
				require.NoError(t, err, "Expected no error but got one")
			}
		})
	}

	// Assert mockClient expectations
	mockClient.AssertExpectations(t)
}
