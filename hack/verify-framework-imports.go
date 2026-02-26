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

var (
	additionalAllowed []string
	strictMode        bool
)

// globalExceptions are paths that are always allowed for all framework files.
// These are utility packages that the framework is permitted to import.
var globalExceptions = []string{
	"pkg/common/observability/logging",
}

// currentCodeExceptionMap maps existing violation in files to their allowed import exceptions.
// This should cleaned-up alongside the relevant files and their imports.
var currentCodeExceptionMap = map[string][]string{
	"pkg/epp/framework/plugins/requestcontrol/test/responsereceived/destination_endpoint_served_verifier.go": {
		"pkg/epp/metadata",
	},
	"pkg/epp/framework/plugins/requestcontrol/test/responsereceived/destination_endpoint_served_verifier_test.go": {
		"pkg/epp/metadata",
	},
	"pkg/epp/framework/plugins/scheduling/scorer/predictedlatency/headers.go": {
		"pkg/epp/util/error",
	},
	"pkg/epp/framework/plugins/scheduling/scorer/predictedlatency/latencypredictor_helper.go": {
		"pkg/epp/metrics",
		"pkg/epp/util/request",
		"sidecars/latencypredictorasync",
	},
	"pkg/epp/framework/plugins/scheduling/scorer/predictedlatency/latencypredictor_helper_test.go": {
		"sidecars/latencypredictorasync",
	},
	"pkg/epp/framework/plugins/scheduling/scorer/predictedlatency/prediction.go": {
		"sidecars/latencypredictorasync",
	},
	"pkg/epp/framework/plugins/scheduling/scorer/predictedlatency/requestcontrol_hooks.go": {
		"pkg/epp/metrics",
		"pkg/epp/util/request",
	},
	"pkg/epp/framework/plugins/scheduling/scorer/predictedlatency/requestcontrol_hooks_test.go": {
		"pkg/epp/util/request",
	},
	"pkg/epp/framework/plugins/scheduling/scorer/predictedlatency/scorer.go": {
		"sidecars/latencypredictorasync",
	},
	"pkg/epp/framework/plugins/scheduling/scorer/predictedlatency/scorer_helpers.go": {
		"pkg/epp/util/error",
	},
	"pkg/epp/framework/plugins/scheduling/scorer/predictedlatency/scorer_test.go": {
		"pkg/epp/util/request",
		"sidecars/latencypredictorasync",
		"test/utils",
	},
	"pkg/epp/framework/plugins/scheduling/scorer/prefix/indexer.go": {
		"pkg/epp/metrics",
	},
	"pkg/epp/framework/plugins/scheduling/scorer/prefix/plugin.go": {
		"pkg/epp/metrics",
	},
}

func init() {
	pflag.StringSliceVar(&additionalAllowed, "allow", []string{}, "Additional allowed import paths (can be specified multiple times)")
	pflag.BoolVar(&strictMode, "strict", false, "Fail on all violations (including allowed exceptions in current code)")
}

func main() {
	pflag.Parse()

	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

type violation struct {
	filePath   string
	importPath string
	isAllowed  bool
}

func (v violation) String() string {
	return fmt.Sprintf("%s: imports %s", v.filePath, v.importPath)
}

func uniqueFileCount(violations []violation) int {
	files := make(map[string]bool)
	for _, v := range violations {
		files[v.filePath] = true
	}
	return len(files)
}

func run() error {
	allowedViolations := []violation{}
	newViolations := []violation{}

	allowedBasePaths := map[string]struct{}{
		"pkg/epp/framework": {}, // framework itself is always allowed
	}
	for _, exc := range globalExceptions { // add the global exceptions (e.g., logging, metrics)
		allowedBasePaths[exc] = struct{}{}
	}
	for _, path := range additionalAllowed { // additional exceptions from command line
		allowedBasePaths[path] = struct{}{}
	}

	fmt.Printf("Validating imports in %s\n", frameworkPath)
	fmt.Printf("Globally allowed paths: %v\n", globalExceptions)
	if !strictMode {
		fmt.Printf("Running in permissive mode: allowed exceptions will be warned, new violations will fail\n")
	} else {
		fmt.Printf("Running in strict mode: all violations will fail\n")
	}
	if len(additionalAllowed) > 0 {
		fmt.Printf("Additional allowed paths (via flags): %v\n", additionalAllowed)
	}
	fmt.Println()

	// walk through all Go files in pkg/epp/framework
	err := filepath.Walk(frameworkPath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		relPath, err := filepath.Rel(".", path)
		if err != nil {
			relPath = path
		}

		if info.IsDir() || !strings.HasSuffix(path, ".go") {
			return nil
		}

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
				relImportPath := strings.TrimPrefix(importPath, repoModule+"/")

				// check if it's in allowed base paths
				allowed := false
				for basePath := range allowedBasePaths {
					if strings.HasPrefix(relImportPath, basePath) {
						allowed = true
						break
					}
				}

				if !allowed {
					isAllowed := false
					if _, hasException := currentCodeExceptionMap[relPath]; hasException {
						isAllowed = true
					}

					if strictMode { // in strict mode, treat all violations as errors
						isAllowed = false
					}

					v := violation{
						filePath:   relPath,
						importPath: relImportPath,
						isAllowed:  isAllowed,
					}
					if isAllowed {
						allowedViolations = append(allowedViolations, v)
					} else {
						newViolations = append(newViolations, v)
					}
				}
			}
		}

		return nil
	})

	if err != nil {
		return fmt.Errorf("failed to walk directory: %w", err)
	}

	// report results
	if len(allowedViolations) > 0 {
		fmt.Printf("\n[WARNING] Allowed violations (current codebase): %d violations across %d files\n",
			len(allowedViolations), uniqueFileCount(allowedViolations))
		for _, v := range allowedViolations {
			fmt.Println("  " + v.String())
		}
	}

	if len(newViolations) > 0 {
		fmt.Printf("\n[ERROR] Found %d new import violations across %d files:\n",
			len(newViolations), uniqueFileCount(newViolations))
		for _, v := range newViolations {
			fmt.Println("  " + v.String())
		}
		return fmt.Errorf("import validation failed: %d new violations found", len(newViolations))
	}

	if strictMode && len(allowedViolations) > 0 {
		fmt.Printf("\n[ERROR] Found %d total import violations (strict mode):\n", len(allowedViolations))
		for _, v := range allowedViolations {
			fmt.Println("  " + v.String())
		}
		return fmt.Errorf("import validation failed (strict mode)")
	}

	if len(allowedViolations) > 0 {
		fmt.Printf("\n[PASS] No new violations. %d allowed exceptions exist in current codebase.\n", len(allowedViolations))
	} else {
		fmt.Printf("\n[PASS] All imports in %s are valid!\n", frameworkPath)
	}
	return nil
}
