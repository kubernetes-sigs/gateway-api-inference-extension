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
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	clientsetscheme "k8s.io/client-go/kubernetes/scheme"

	// Import runtime package for scheme creation
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/client"
	k8sconfig "sigs.k8s.io/controller-runtime/pkg/client/config"
	"sigs.k8s.io/yaml"

	// Import necessary types and utilities from the core Gateway API conformance suite.
	gatewayv1 "sigs.k8s.io/gateway-api/apis/v1"            // Import core Gateway API types
	confapis "sigs.k8s.io/gateway-api/conformance/apis/v1" // Report struct definition
	confconfig "sigs.k8s.io/gateway-api/conformance/utils/config"
	confflags "sigs.k8s.io/gateway-api/conformance/utils/flags"
	apikubernetes "sigs.k8s.io/gateway-api/conformance/utils/kubernetes"
	confsuite "sigs.k8s.io/gateway-api/conformance/utils/suite"
	"sigs.k8s.io/gateway-api/pkg/features"

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

// Constants for the shared Gateway
const (
	SharedGatewayName      = "conformance-gateway"       // Name of the Gateway in manifests.yaml
	SharedGatewayNamespace = "gateway-conformance-infra" // Namespace of the Gateway
)

// GatewayLayerProfileName defines the name for the conformance profile that tests
// the Gateway API layer aspects of the Inference Extension (e.g., InferencePool, InferenceModel CRDs).
// Future profiles will cover EPP and ModelServer layers.
const GatewayLayerProfileName confsuite.ConformanceProfileName = "Gateway"

// InferenceCoreFeatures defines the core features that implementations
// of the "Gateway" profile for the Inference Extension MUST support.
var InferenceCoreFeatures = sets.New(
	features.SupportGateway, // This is needed to ensure manifest gets applied during setup.
)

var GatewayLayerProfile = confsuite.ConformanceProfile{
	Name:         GatewayLayerProfileName,
	CoreFeatures: InferenceCoreFeatures,
}

// logDebugf conditionally logs a debug message if debug mode is enabled.
func logDebugf(t *testing.T, debug bool, format string, args ...any) {
	if debug {
		t.Helper()
		t.Logf(format, args...)
	}
}

// DefaultOptions parses command line flags and sets up the suite options.
// Adapted from the core Gateway API conformance suite.
func DefaultOptions(t *testing.T) confsuite.ConformanceOptions {
	t.Helper()

	cfg, err := k8sconfig.GetConfig()
	require.NoError(t, err, "error loading Kubernetes config")

	scheme := runtime.NewScheme()

	t.Log("Registering API types with scheme...")
	// Register core K8s types (like v1.Secret for certs) to scheme, needed by client to create/manage these resources.
	require.NoError(t, clientsetscheme.AddToScheme(scheme), "failed to add core Kubernetes types to scheme")
	// Add Gateway API types
	require.NoError(t, gatewayv1.Install(scheme), "failed to install gatewayv1 types into scheme")
	// Add APIExtensions types (for CRDs)
	require.NoError(t, apiextensionsv1.AddToScheme(scheme), "failed to add apiextensionsv1 types to scheme")

	// Register Inference Extension API types
	t.Logf("Attempting to install inferencev1alpha2 types into scheme from package: %s", inferencev1alpha2.GroupName)
	require.NoError(t, inferencev1alpha2.Install(scheme), "failed to install inferencev1alpha2 types into scheme")

	clientOptions := client.Options{Scheme: scheme}
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

	baseManifestsValue := "resources/manifests/manifests.yaml"

	opts := confsuite.ConformanceOptions{
		Client:               c,
		ClientOptions:        clientOptions,
		Clientset:            cs,
		RestConfig:           cfg,
		GatewayClassName:     *confflags.GatewayClassName,
		BaseManifests:        baseManifestsValue,
		Debug:                *confflags.ShowDebug,
		CleanupBaseResources: *confflags.CleanupBaseResources,
		SupportedFeatures:    sets.New[features.FeatureName](),
		TimeoutConfig:        confconfig.DefaultTimeoutConfig(),
		SkipTests:            skipTests,
		ExemptFeatures:       exemptFeatures,
		RunTest:              *confflags.RunTest,
		Mode:                 *confflags.Mode,
		Implementation:       implementation,
		ConformanceProfiles:  conformanceProfiles,
		ManifestFS:           []fs.FS{&Manifests},
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
		logDebugf(t, opts.Debug, "Populating SupportedFeatures with GatewayLayerProfile.CoreFeatures: %v", GatewayLayerProfile.CoreFeatures.UnsortedList())
		if GatewayLayerProfile.CoreFeatures.Len() > 0 {
			opts.SupportedFeatures = opts.SupportedFeatures.Insert(GatewayLayerProfile.CoreFeatures.UnsortedList()...)
		}
	}

	// Remove any features explicitly exempted via flags.
	if opts.ExemptFeatures.Len() > 0 {
		logDebugf(t, opts.Debug, "Removing ExemptFeatures from SupportedFeatures: %v", opts.ExemptFeatures.UnsortedList())
		opts.SupportedFeatures = opts.SupportedFeatures.Delete(opts.ExemptFeatures.UnsortedList()...)
	}

	logDebugf(t, opts.Debug, "Final opts.SupportedFeatures: %v", opts.SupportedFeatures.UnsortedList())

	return opts
}

// RunConformance runs the Inference Extension conformance tests using default options.
func RunConformance(t *testing.T) {
	RunConformanceWithOptions(t, DefaultOptions(t))
}

// RunConformanceWithOptions runs the Inference Extension conformance tests with specific options.
func RunConformanceWithOptions(t *testing.T, opts confsuite.ConformanceOptions) {
	t.Helper()
	t.Logf("Running Inference Extension conformance tests with GatewayClass %s", opts.GatewayClassName)
	logDebugf(t, opts.Debug, "RunConformanceWithOptions: BaseManifests path being used by opts: %q", opts.BaseManifests)

	// Register the GatewayLayerProfile with the suite runner.
	// In the future, other profiles (EPP, ModelServer) will also be registered here,
	// and the suite runner will execute tests based on the selected profiles.
	confsuite.RegisterConformanceProfile(GatewayLayerProfile)

	// Initialize the test suite.
	cSuite, err := confsuite.NewConformanceTestSuite(opts)
	require.NoError(t, err, "error initializing conformance suite")

	cSuite.Setup(t, tests.ConformanceTests)

	sharedGwNN := types.NamespacedName{Name: SharedGatewayName, Namespace: SharedGatewayNamespace}

	// Validate Gateway setup.
	ensureGatewayAvailableAndReady(t, cSuite.Client, opts, sharedGwNN)
	t.Log("Running Inference Extension conformance tests against all registered tests")
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

// ensureGatewayAvailableAndReady polls for the specified Gateway to exist and become ready
// with an address and programmed condition.
func ensureGatewayAvailableAndReady(t *testing.T, k8sClient client.Client, opts confsuite.ConformanceOptions, gatewayNN types.NamespacedName) {
	t.Helper()

	t.Logf("Attempting to fetch Gateway %s/%s.", gatewayNN.Namespace, gatewayNN.Name)
	gw := &gatewayv1.Gateway{} // This gw instance will be populated by the poll function

	// Define polling interval
	// TODO: Make this configurable using a local TimeoutConfig (from ConformanceOptions perhaps)
	pollingInterval := 5 * time.Second
	// Use the GatewayMustHaveAddress timeout from the suite's TimeoutConfig for the Gateway object to appear
	waitForGatewayCreationTimeout := opts.TimeoutConfig.GatewayMustHaveAddress

	logDebugf(t, opts.Debug, "Waiting up to %v for Gateway object %s/%s to appear after manifest application...", waitForGatewayCreationTimeout, gatewayNN.Namespace, gatewayNN.Name)

	ctx := context.TODO()
	pollErr := wait.PollUntilContextTimeout(ctx, pollingInterval, waitForGatewayCreationTimeout, true, func(pollCtx context.Context) (bool, error) {
		fetchErr := k8sClient.Get(pollCtx, gatewayNN, gw)
		if fetchErr == nil {
			t.Logf("Successfully fetched Gateway %s/%s. Spec.GatewayClassName: %s",
				gw.Namespace, gw.Name, gw.Spec.GatewayClassName)
			return true, nil
		}
		if apierrors.IsNotFound(fetchErr) {
			logDebugf(t, opts.Debug, "Gateway %s/%s not found, still waiting...", gatewayNN.Namespace, gatewayNN.Name)
			return false, nil // Not found, continue polling
		}
		// For any other error, stop polling and return this error
		t.Logf("Error fetching Gateway %s/%s: %v. Halting polling for this attempt.", gatewayNN.Namespace, gatewayNN.Name, fetchErr)
		return false, fetchErr
	})

	// Check if polling timed out or an error occurred during polling
	if pollErr != nil {
		var failureMessage string
		if errors.Is(pollErr, context.DeadlineExceeded) {
			failureMessage = fmt.Sprintf("Timed out after %v waiting for Gateway object %s/%s to appear in the API server.",
				waitForGatewayCreationTimeout, gatewayNN.Namespace, gatewayNN.Name)
		} else {
			failureMessage = fmt.Sprintf("Error while waiting for Gateway object %s/%s to appear: %v.",
				gatewayNN.Namespace, gatewayNN.Name, pollErr)
		}
		finalMessage := failureMessage + " The Gateway object should have been created by the base manifest application."
		require.FailNow(t, finalMessage) // Use FailNow to stop if the Gateway isn't found.
	}

	logDebugf(t, opts.Debug, "Waiting for shared Gateway %s/%s to be ready", gatewayNN.Namespace, gatewayNN.Name)
	apikubernetes.GatewayMustHaveCondition(t, k8sClient, opts.TimeoutConfig, gatewayNN, metav1.Condition{
		Type:   string(gatewayv1.GatewayConditionAccepted),
		Status: metav1.ConditionTrue,
	})
	apikubernetes.GatewayMustHaveCondition(t, k8sClient, opts.TimeoutConfig, gatewayNN, metav1.Condition{
		Type:   string(gatewayv1.GatewayConditionProgrammed),
		Status: metav1.ConditionTrue,
	})
	_, err := apikubernetes.WaitForGatewayAddress(t, k8sClient, opts.TimeoutConfig, apikubernetes.NewGatewayRef(gatewayNN))
	require.NoErrorf(t, err, "shared gateway %s/%s did not get an address", gatewayNN.Namespace, gatewayNN.Name)
	t.Logf("Shared Gateway %s/%s is ready.", gatewayNN.Namespace, gatewayNN.Name)
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
