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

// Package conformance contains the core setup and execution logic
// for the Gateway API Inference Extension conformance test suite.
package conformance

import (
	"fmt"
	"io/fs"
	"os"
	"testing"

	"github.com/stretchr/testify/require"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	clientset "k8s.io/client-go/kubernetes"

	// Import runtime package for scheme creation
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/config"
	"sigs.k8s.io/yaml"

	// Import necessary types and utilities from the core Gateway API conformance suite.
	// Assumes sigs.k8s.io/gateway-api is a dependency in the go.mod.
	gatewayv1 "sigs.k8s.io/gateway-api/apis/v1"            // Import core Gateway API types
	confapis "sigs.k8s.io/gateway-api/conformance/apis/v1" // Report struct definition
	confconfig "sigs.k8s.io/gateway-api/conformance/utils/config"
	confflags "sigs.k8s.io/gateway-api/conformance/utils/flags"
	confsuite "sigs.k8s.io/gateway-api/conformance/utils/suite"
	"sigs.k8s.io/gateway-api/pkg/features" // Using core features definitions if applicable

	// Import the test definitions package to access the ConformanceTests slice
	"sigs.k8s.io/gateway-api-inference-extension/conformance/tests"

	// Import test packages using blank identifier
	// This triggers the init() functions in these packages, which register the tests
	// by appending them to the tests.ConformanceTests slice.
	_ "sigs.k8s.io/gateway-api-inference-extension/conformance/tests/basic"
	// TODO: Add blank imports for other test categories as they are created.
	// _ "sigs.k8s.io/gateway-api-inference-extension/conformance/tests/model_routing"

	// Import the Inference Extension API types
	inferencev1alpha2 "sigs.k8s.io/gateway-api-inference-extension/api/v1alpha2"
)

// GatewayLayerProfileName defines the name for the conformance profile that tests
// the Gateway API layer aspects of the Inference Extension (e.g., InferencePool, InferenceModel CRDs).
// Future profiles will cover EPP and ModelServer layers.
const GatewayLayerProfileName confsuite.ConformanceProfileName = "Gateway"

var InferenceCoreFeatures = sets.New[features.FeatureName]() // Placeholder - Populate with actual features specific to this profile or manage features per profile

// GatewayLayerProfile defines the conformance profile for the Gateway API layer
// of the Inference Extension.
// In future iterations, we will add constants and ConformanceProfile structs for
// EPPProfileName ("EPP") and ModelServerProfileName ("ModelServer")
// to cover their respective conformance layers.
var GatewayLayerProfile = confsuite.ConformanceProfile{
	Name:         GatewayLayerProfileName,
	CoreFeatures: InferenceCoreFeatures,
}

// DefaultOptions parses command line flags and sets up the suite options.
// Adapted from the core Gateway API conformance suite.
func DefaultOptions(t *testing.T) confsuite.ConformanceOptions {
	t.Helper()

	cfg, err := config.GetConfig()
	require.NoError(t, err, "error loading Kubernetes config")

	// Initialize client options. The scheme must include Gateway API types
	// and the Inference Extension types.
	clientOptions := client.Options{}
	scheme := clientOptions.Scheme
	if scheme == nil {
		// If default options don't provide a scheme, create one using runtime.NewScheme().
		scheme = runtime.NewScheme()
		clientOptions.Scheme = scheme
	}

	// Register necessary API Types
	require.NoError(t, gatewayv1.Install(scheme)) // Add core Gateway API types
	// Add the Inference Extension API types to the scheme using the correct import alias
	require.NoError(t, inferencev1alpha2.Install(scheme))
	require.NoError(t, apiextensionsv1.AddToScheme(scheme)) // Needed for CRD checks

	// Create the Kubernetes clients
	c, err := client.New(cfg, clientOptions)
	require.NoError(t, err, "error initializing Kubernetes client")
	cs, err := clientset.NewForConfig(cfg)
	require.NoError(t, err, "error initializing Kubernetes clientset")

	exemptFeatures := confsuite.ParseSupportedFeatures(*confflags.ExemptFeatures)
	skipTests := confsuite.ParseSkipTests(*confflags.SkipTests)
	// Initially, run the GatewayLayerProfile. This will expand as other profiles
	// (EPP, ModelServer) are added and can be selected via flags in future iterations.
	conformanceProfiles := sets.New(GatewayLayerProfileName)

	// Implementation details from flags
	implementation := confsuite.ParseImplementation(
		*confflags.ImplementationOrganization,
		*confflags.ImplementationProject,
		*confflags.ImplementationURL,
		*confflags.ImplementationVersion,
		*confflags.ImplementationContact,
	)

	// Inference Extension Specific Report Fields
	inferenceExtensionVersion := "v0.3.0"
	_ = inferenceExtensionVersion // Avoid unused variable error until implemented

	// Create ConformanceOptions
	opts := confsuite.ConformanceOptions{
		Client:               c,
		Clientset:            cs,
		RestConfig:           cfg,
		GatewayClassName:     *confflags.GatewayClassName,
		Debug:                *confflags.ShowDebug,
		CleanupBaseResources: *confflags.CleanupBaseResources,
		SupportedFeatures:    sets.New[features.FeatureName](), // Initialize empty, will be populated below
		TimeoutConfig:        confconfig.DefaultTimeoutConfig(),
		SkipTests:            skipTests,
		ExemptFeatures:       exemptFeatures,
		RunTest:              *confflags.RunTest,
		Mode:                 *confflags.Mode,
		Implementation:       implementation,
		ConformanceProfiles:  conformanceProfiles,
		ManifestFS:           []fs.FS{&Manifests}, // Assumes embed.go defines `Manifests`
		ReportOutputPath:     *confflags.ReportOutput,
		SkipProvisionalTests: *confflags.SkipProvisionalTests,
		// TODO: Add the inference extension specific fields to ConformanceOptions struct if needed,
		// or handle them during report generation.
		// GatewayAPIInferenceExtensionChannel: inferenceExtensionChannel,
		// GatewayAPIInferenceExtensionVersion: inferenceExtensionVersion,
	}

	// Populate SupportedFeatures based on the GatewayLayerProfile.
	// Since all features are mandatory for this profile, add all defined core features.
	if opts.ConformanceProfiles.Has(GatewayLayerProfileName) {
		for feature := range GatewayLayerProfile.CoreFeatures {
			opts.SupportedFeatures.Insert(feature)
		}
	}

	// Remove any features explicitly exempted via flags.
	for feature := range opts.ExemptFeatures {
		opts.SupportedFeatures.Delete(feature)
	}

	return opts
}

// RunConformance runs the Inference Extension conformance tests using default options.
func RunConformance(t *testing.T) {
	RunConformanceWithOptions(t, DefaultOptions(t))
}

// RunConformanceWithOptions runs the Inference Extension conformance tests with specific options.
func RunConformanceWithOptions(t *testing.T, opts confsuite.ConformanceOptions) {
	t.Logf("Running Inference Extension conformance tests with GatewayClass %s", opts.GatewayClassName)

	// Register the GatewayLayerProfile with the suite runner.
	// In the future, other profiles (EPP, ModelServer) will also be registered here,
	// and the suite runner will execute tests based on the selected profiles.
	confsuite.RegisterConformanceProfile(GatewayLayerProfile)

	// Initialize the test suite.
	cSuite, err := confsuite.NewConformanceTestSuite(opts)
	require.NoError(t, err, "error initializing conformance suite")

	t.Log("Setting up Inference Extension conformance tests")
	// Setup requires the list of tests, which is populated by the init() functions
	// triggered by the blank imports at the top of this file.
	cSuite.Setup(t, tests.ConformanceTests)

	t.Log("Running Inference Extension conformance tests")
	// Run the tests.
	err = cSuite.Run(t, tests.ConformanceTests)
	require.NoError(t, err, "error running conformance tests")

	// Generate and write the report if requested.
	if opts.ReportOutputPath != "" {
		t.Log("Generating Inference Extension conformance report")
		report, err := cSuite.Report() // Use the existing report generation logic.
		require.NoError(t, err, "error generating conformance report")

		// TODO: Modify the report struct here if channel, version need to be modified.
		// Example (requires adding fields to confapis.ConformanceReport):
		// report.GatewayAPIInferenceExtensionChannel = opts.GatewayAPIInferenceExtensionChannel
		// report.GatewayAPIInferenceExtensionVersion = opts.GatewayAPIInferenceExtensionVersion

		err = writeReport(t.Logf, *report, opts.ReportOutputPath)
		require.NoError(t, err, "error writing conformance report")
	}
}

// writeReport writes the generated conformance report to the specified output file or logs it.
// Adapted from the core Gateway API suite.
func writeReport(logf func(string, ...any), report confapis.ConformanceReport, output string) error {
	rawReport, err := yaml.Marshal(report)
	if err != nil {
		return fmt.Errorf("error marshaling report: %w", err)
	}

	if output != "" {
		if err = os.WriteFile(output, rawReport, 0o600); err != nil {
			return fmt.Errorf("error writing report file %s: %w", output, err)
		}
		logf("Conformance report written to %s", output)
	} else {
		// Log the report YAML to stdout if no output file is specified.
		logf("Conformance report:\n%s", string(rawReport))
	}
	return nil
}
