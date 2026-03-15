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

// verify-component-imports verifies that EPP and BBR do not have cross imports.
// EPP code (pkg/epp/) must not import BBR code (pkg/bbr/), and vice versa.
// Both may import common code (pkg/common/) and external dependencies.
//
// Known violations are listed in currentCodeExceptionMap and reported as
// warnings. New violations (not in the map) cause a non-zero exit, blocking
// PRs that introduce additional cross-imports.
// Use --strict to treat all violations as errors (for when all exceptions
// have been resolved).
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
	repoModule = "sigs.k8s.io/gateway-api-inference-extension"
)

// componentPair defines a directional import boundary between two components.
type componentPair struct {
	sourcePath string // the component whose files are being checked
	blockedPkg string // the import path prefix that is forbidden
}

var (
	strictMode bool

	// componentBoundaries defines import rules. Each entry prevents source
	// from importing blocked.
	componentBoundaries = []componentPair{
		{sourcePath: "./pkg/bbr", blockedPkg: "pkg/epp"},
		{sourcePath: "./pkg/epp", blockedPkg: "pkg/bbr"},
	}
)

// currentCodeExceptionMap lists known cross-component imports that exist in
// the codebase today. These are reported as warnings but do not fail the
// check. As each violation is fixed, its entry should be removed so the list
// only shrinks over time.
var currentCodeExceptionMap = map[string][]string{
	"pkg/bbr/framework/plugins.go": {
		"pkg/epp/framework/interface/plugin",
	},
	"pkg/bbr/plugins/body_field_to_header.go": {
		"pkg/epp/framework/interface/plugin",
	},
	"pkg/bbr/plugins/default.go": {
		"pkg/epp/framework/interface/plugin",
	},
}

func init() {
	pflag.BoolVar(&strictMode, "strict", false, "Fail on all violations including allowed exceptions")
}

type violation struct {
	filePath   string
	importPath string
	component  string
	blocked    string
	isAllowed  bool
}

func (v violation) String() string {
	return fmt.Sprintf("%s: imports %s (component %s must not import %s)", v.filePath, v.importPath, v.component, v.blocked)
}

func main() {
	pflag.Parse()

	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	var allowedViolations []violation
	var newViolations []violation

	if strictMode {
		fmt.Println("Running in strict mode: all violations will fail")
	} else {
		fmt.Println("Running in permissive mode: allowed exceptions will be warned, new violations will fail")
	}
	fmt.Println()

	for _, boundary := range componentBoundaries {
		fmt.Printf("Checking %s does not import %s\n", boundary.sourcePath, boundary.blockedPkg)

		allowed, new, err := checkBoundary(boundary)
		if err != nil {
			return fmt.Errorf("failed checking %s: %w", boundary.sourcePath, err)
		}
		allowedViolations = append(allowedViolations, allowed...)
		newViolations = append(newViolations, new...)
	}

	if len(allowedViolations) > 0 {
		fmt.Printf("\n[WARNING] Allowed violations (current codebase): %d violations across %d files\n",
			len(allowedViolations), uniqueFileCount(allowedViolations))
		for _, v := range allowedViolations {
			fmt.Println("  " + v.String())
		}
		fmt.Println("\nShared code should be placed in pkg/common/ so both components can use it.")
	}

	if len(newViolations) > 0 {
		fmt.Printf("\n[ERROR] Found %d new cross-component import violations across %d files:\n",
			len(newViolations), uniqueFileCount(newViolations))
		for _, v := range newViolations {
			fmt.Println("  " + v.String())
		}
		return fmt.Errorf("cross-component import validation failed: %d new violations found", len(newViolations))
	}

	if strictMode && len(allowedViolations) > 0 {
		fmt.Printf("\n[ERROR] Found %d total import violations (strict mode):\n", len(allowedViolations))
		for _, v := range allowedViolations {
			fmt.Println("  " + v.String())
		}
		return fmt.Errorf("cross-component import validation failed (strict mode)")
	}

	if len(allowedViolations) > 0 {
		fmt.Printf("\n[PASS] No new violations. %d allowed exceptions exist in current codebase.\n", len(allowedViolations))
	} else {
		fmt.Printf("\n[PASS] No cross-component imports found between EPP and BBR.\n")
	}
	return nil
}

func checkBoundary(boundary componentPair) (allowed []violation, new []violation, err error) {
	err = filepath.Walk(boundary.sourcePath, func(path string, info os.FileInfo, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if info.IsDir() || !strings.HasSuffix(path, ".go") {
			return nil
		}

		relPath, relErr := filepath.Rel(".", path)
		if relErr != nil {
			relPath = path
		}

		fset := token.NewFileSet()
		node, parseErr := parser.ParseFile(fset, path, nil, parser.ImportsOnly)
		if parseErr != nil {
			return fmt.Errorf("failed to parse %s: %w", path, parseErr)
		}

		for _, imp := range node.Imports {
			importPath := strings.Trim(imp.Path.Value, `"`)

			if !strings.HasPrefix(importPath, repoModule) {
				continue
			}

			relImportPath := strings.TrimPrefix(importPath, repoModule+"/")
			if strings.HasPrefix(relImportPath, boundary.blockedPkg) {
				v := violation{
					filePath:   relPath,
					importPath: relImportPath,
					component:  filepath.Base(boundary.sourcePath),
					blocked:    boundary.blockedPkg,
				}

				if isException(relPath, relImportPath) && !strictMode {
					v.isAllowed = true
					allowed = append(allowed, v)
				} else {
					new = append(new, v)
				}
			}
		}
		return nil
	})

	return allowed, new, err
}

// isException returns true if the file+import pair is a known exception.
func isException(filePath, importPath string) bool {
	allowedImports, exists := currentCodeExceptionMap[filePath]
	if !exists {
		return false
	}
	for _, allowed := range allowedImports {
		if importPath == allowed {
			return true
		}
	}
	return false
}

func uniqueFileCount(violations []violation) int {
	files := make(map[string]bool)
	for _, v := range violations {
		files[v.filePath] = true
	}
	return len(files)
}
