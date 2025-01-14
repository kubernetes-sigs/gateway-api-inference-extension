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

package crd

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	apiextv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/client-go/util/retry"
	klog "k8s.io/klog/v2"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

// InstallCRDs reads YAML files containing CustomResourceDefinitions from the given directory path.
// It processes only valid YAML files with `apiVersion: apiextensions.k8s.io/v1` and `kind: CustomResourceDefinition`.
func InstallCRDs(ctx context.Context, cli client.Client, dirPath string) error {
	var aggregatedErrors []string

	// Read all files in the directory
	files, err := os.ReadDir(dirPath)
	if err != nil {
		return fmt.Errorf("failed to read directory %s: %v", dirPath, err)
	}

	for _, file := range files {
		// Skip directories and non-YAML files
		if file.IsDir() || (!strings.HasSuffix(file.Name(), ".yaml") && !strings.HasSuffix(file.Name(), ".yml")) {
			klog.Infof("Skipping non-YAML file: %s", file.Name())
			continue
		}

		// Construct full file path
		filePath := filepath.Join(dirPath, file.Name())

		// Read the file
		content, err := os.ReadFile(filePath)
		if err != nil {
			klog.Infof("Failed to read file %s: %v", filePath, err)
			aggregatedErrors = append(aggregatedErrors, err.Error())
			continue
		}

		// Decode YAML content and filter by `apiVersion` and `kind`
		klog.Infof("Processing file: %s", filePath)
		decoder := yaml.NewYAMLOrJSONDecoder(bytes.NewReader(content), 1024)
		for {
			obj := &apiextv1.CustomResourceDefinition{}
			err := decoder.Decode(obj)
			if err != nil {
				if errors.Is(err, io.EOF) {
					break // End of file
				}
				klog.Errorf("Failed to decode CRD object from file %s: %v", filePath, err)
				aggregatedErrors = append(aggregatedErrors, err.Error())
				break
			}

			// Check if the decoded object is a valid CRD
			if obj.APIVersion != "apiextensions.k8s.io/v1" || obj.Kind != "CustomResourceDefinition" {
				klog.Infof("Skipping invalid CRD object in file %s", filePath)
				continue
			}

			// Apply the CRD using the controller-runtime client
			err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
				_, err := controllerutil.CreateOrUpdate(ctx, cli, obj, func() error {
					return nil
				})
				return err
			})
			if err != nil {
				klog.Errorf("Failed to apply CRD %s: %v", obj.Name, err)
				aggregatedErrors = append(aggregatedErrors, err.Error())
			}
		}
	}

	if len(aggregatedErrors) > 0 {
		return errors.New(strings.Join(aggregatedErrors, "; "))
	}

	return nil
}
