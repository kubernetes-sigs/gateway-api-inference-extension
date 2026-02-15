//go:build ignore
// +build ignore

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

// validate-framework-imports validates that files under pkg/epp/framework
// only import other files from within pkg/epp/framework (or external
// dependencies).
package main

import (
	"fmt"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/pflag"
)

const (
	frameworkPath = "./pkg/epp/framework"
	repoModule    = "sigs.k8s.io/gateway-api-inference-extension"
)

// allowedExceptions lists additional import paths that are allowed
// for framework files beyond pkg/epp/framework itself.
// Add paths here that the framework is permitted to import from other
// parts of the repository.
var allowedExceptions = []string{
	"pkg/common/observability/logging",
	// "pkg/epp/util/error",
}

var (
	additionalAllowed []string
)

func main() {
	pflag.StringSliceVar(&additionalAllowed, "allow", []string{}, "Additional allowed import paths via command line (can be specified multiple times)")
	pflag.Parse()

	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

// isAllowedImport checks if the given import path starts with any allowed path
func isAllowedImport(importPath string, allowedPaths map[string]struct{}) bool {
	for allowedPath := range allowedPaths {
		if strings.HasPrefix(importPath, allowedPath) {
			return true
		}
	}
	return false
}

func run() error {
	violations := []string{}

	// build the list of allowed paths with deduplication
	allowedPathsMap := make(map[string]struct{})
	allowedPathsMap["pkg/epp/framework"] = struct{}{} // always allowed
	
	// add built-in exceptions
	for _, path := range allowedExceptions {
		allowedPathsMap[path] = struct{}{}
	}
	
	// add exceptions from command-line
	for _, path := range additionalAllowed {
		allowedPathsMap[path] = struct{}{}
	}

	// Convert map keys to slice for display purposes only
	allowedPathsList := make([]string, 0, len(allowedPathsMap))
	for path := range allowedPathsMap {
		allowedPathsList = append(allowedPathsList, path)
	}

	fmt.Printf("Validating imports in %s\n", frameworkPath)
	if len(allowedExceptions) > 0 {
		fmt.Printf("Allowed exceptions (hardcoded): %v\n", allowedExceptions)
	}
	if len(additionalAllowed) > 0 {
		fmt.Printf("Additional allowed paths (via flags): %v\n", additionalAllowed)
	}
	if len(allowedPathsList) > 1 {
		fmt.Printf("Total unique allowed paths: %v\n", allowedPathsList)
	}
	fmt.Println()

	// walk through all Go files in pkg/epp/framework
	err := filepath.Walk(frameworkPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info.IsDir() || !strings.HasSuffix(path, ".go") {
			return nil
		}

		// skip test files if desired (currently we check them too)
		// if strings.HasSuffix(path, "_test.go") {
		// 	return nil
		// }

		// parse the Go file
		fset := token.NewFileSet()
		node, err := parser.ParseFile(fset, path, nil, parser.ImportsOnly)
		if err != nil {
			return fmt.Errorf("failed to parse %s: %w", path, err)
		}

		// check imports
		for _, imp := range node.Imports {
			importPath := strings.Trim(imp.Path.Value, `"`)
			
			if strings.HasPrefix(importPath, repoModule) {
				relPath := strings.TrimPrefix(importPath, repoModule+"/")
				
				// Check if the import path starts with any allowed path
				allowed := isAllowedImport(relPath, allowedPathsMap)
				
				if !allowed {
					violations = append(violations, fmt.Sprintf(
						"%s: imports %s (not in allowed paths)",
						path, importPath,
					))
				}
			}
		}

		return nil
	})

	if err != nil {
		return fmt.Errorf("failed to walk directory: %w", err)
	}

	if len(violations) > 0 { // report results
		fmt.Printf("\nAllowed paths: %v\n", allowedPathsList)
		fmt.Println("❌ Found %d import violations:", len(violations))
		fmt.Println()
		for _, v := range violations {
			fmt.Println("  " + v)
		}
		return fmt.Errorf("import validation failed")
	}

	fmt.Printf("✅ All %s imports are valid!\n", frameworkPath)
	return nil
}
