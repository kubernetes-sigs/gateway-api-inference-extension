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
	"slices"
	"testing"

	"github.com/stretchr/testify/require"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	clientsetscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client"
	k8sconfig "sigs.k8s.io/controller-runtime/pkg/client/config"
	confflags "sigs.k8s.io/gateway-api/conformance/utils/flags"
	gatewayk8utils "sigs.k8s.io/gateway-api/conformance/utils/kubernetes"
	confsuite "sigs.k8s.io/gateway-api/conformance/utils/suite"
	"sigs.k8s.io/gateway-api/conformance/utils/tlog"
	"sigs.k8s.io/gateway-api/pkg/features"

	inferencev1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	inferencev1alpha2 "sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/conformance/resources"
	"sigs.k8s.io/gateway-api-inference-extension/conformance/tests"
	inferenceconfig "sigs.k8s.io/gateway-api-inference-extension/conformance/utils/config"
	"sigs.k8s.io/gateway-api-inference-extension/conformance/utils/roundtripper"
	"sigs.k8s.io/gateway-api-inference-extension/version"
)

// GatewayLayerProfileName defines the name for the conformance profile that tests
// the Gateway API layer aspects of the Inference Extension (e.g., InferencePool, InferenceObjective CRDs).
// Future profiles will cover EPP and ModelServer layers.
const GatewayLayerProfileName confsuite.ConformanceProfileName = "Gateway"

// TODO(#863) Create a dedicated share location for feature names similar to
// sigs.k8s.io/gateway-api/pkg/features and change the tests from
// string casting the feature name to referencing the shared feature names.

// Conformance specific features
const SupportInferencePool features.FeatureName = "SupportInferencePool"

// InferenceCoreFeatures defines the core features that implementations
// of the "Gateway" profile for the Inference Extension MUST support.
var InferenceCoreFeatures = sets.New(
	features.SupportGateway, // This is needed to ensure manifest gets applied during setup.
	features.SupportHTTPRoute,
	SupportInferencePool,
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
	require.NoError(t, gatewayk8utils.InstallGatewayV1(scheme), "failed to install gatewayv1 types into scheme")
	// Add APIExtensions types (for CRDs)
	require.NoError(t, apiextensionsv1.AddToScheme(scheme), "failed to add apiextensionsv1 types to scheme")

	// Register Inference Extension API types
	t.Logf("Attempting to install inferencev1alpha2 types into scheme from package: %s", inferencev1alpha2.GroupName)
	require.NoError(t, inferencev1alpha2.Install(scheme), "failed to install inferencev1alpha2 types into scheme")
	t.Logf("Attempting to install inferencev1 types into scheme from package: %s", inferencev1.GroupName)
	require.NoError(t, inferencev1.Install(scheme), "failed to install inferencev1 types into scheme")

	clientOptions := client.Options{Scheme: scheme}
	c, err := client.New(cfg, clientOptions)
	require.NoError(t, err, "error initializing Kubernetes client")
	cs, err := clientset.NewForConfig(cfg)
	require.NoError(t, err, "error initializing Kubernetes clientset")

	exemptFeatures := confsuite.ParseSupportedFeatures(*confflags.ExemptFeatures)
	skipTests := confsuite.ParseSkipTests(*confflags.SkipTests)
	namespaceLabels := confsuite.ParseKeyValuePairs(*confflags.NamespaceLabels)
	namespaceAnnotations := confsuite.ParseKeyValuePairs(*confflags.NamespaceAnnotations)

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

	baseManifestsValue := "resources/base.yaml"
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
		TimeoutConfig:        inferenceconfig.DefaultInferenceExtensionTimeoutConfig().TimeoutConfig,
		SkipTests:            skipTests,
		ExemptFeatures:       exemptFeatures,
		RunTest:              *confflags.RunTest,
		Mode:                 *confflags.Mode,
		Implementation:       implementation,
		ConformanceProfiles:  conformanceProfiles,
		ManifestFS:           []fs.FS{&Manifests},
		ReportOutputPath:     *confflags.ReportOutput,
		SkipProvisionalTests: *confflags.SkipProvisionalTests,
		AllowCRDsMismatch:    *confflags.AllowCRDsMismatch,
		NamespaceLabels:      namespaceLabels,
		NamespaceAnnotations: namespaceAnnotations,
		// TODO: Add the inference extension specific fields to ConformanceOptions struct if needed,
		// or handle them during report generation.
		// GatewayAPIInferenceExtensionChannel: inferenceExtensionChannel,
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
	opts := DefaultOptions(t)
	tests.RoundTripper = roundtripper.DefaultRoundTripper{Debug: opts.Debug, TimeoutConfig: opts.TimeoutConfig}
	RunConformanceWithOptions(t, opts)
}

// RunConformanceWithOptions runs the Inference Extension conformance tests with specific options.
func RunConformanceWithOptions(t *testing.T, opts confsuite.ConformanceOptions) {
	t.Helper()
	ctx := context.Background()

	t.Logf("Running Inference Extension conformance tests with GatewayClass %s", opts.GatewayClassName)
	logDebugf(t, opts.Debug, "RunConformanceWithOptions: BaseManifests path being used by opts: %q", opts.BaseManifests)

	// Register the GatewayLayerProfile with the suite runner.
	// In the future, other profiles (EPP, ModelServer) will also be registered here,
	// and the suite runner will execute tests based on the selected profiles.
	confsuite.RegisterConformanceProfile(GatewayLayerProfile)

	// Initialize the test suite.
	cSuite, err := confsuite.NewConformanceTestSuite(opts)
	require.NoError(t, err, "error initializing conformance suite")

	installedCRDs := &apiextensionsv1.CustomResourceDefinitionList{}
	err = opts.Client.List(ctx, installedCRDs)
	require.NoError(t, err, "error getting installedCRDs")
	apiVersion, err := getGatewayInferenceExtensionVersion(installedCRDs.Items)
	if err != nil {
		if opts.AllowCRDsMismatch {
			apiVersion = "UNDEFINED"
		} else {
			require.NoError(t, err, "error getting the gateway ineference extension version")
		}
	}
	SetupConformanceTestSuite(ctx, t, cSuite, opts, tests.ConformanceTests)
	t.Log("Running Inference Extension conformance tests against all registered tests")
	err = cSuite.Run(t, tests.ConformanceTests)
	require.NoError(t, err, "error running conformance tests")

	if opts.ReportOutputPath != "" {
		t.Log("Generating Inference Extension conformance report")
		report, err := cSuite.Report() // Use the existing report generation logic.
		require.NoError(t, err, "error generating conformance report")
		inferenceReport := GatewayAPIInferenceExtensionConformanceReport{
			GatewayAPIInferenceExtensionVersion: apiVersion,
			ConformanceReport:                   *report,
		}
		err = inferenceReport.WriteReport(t.Logf, opts.ReportOutputPath)
		require.NoError(t, err, "error writing conformance report")
	}
}

func SetupConformanceTestSuite(ctx context.Context, t *testing.T, suite *confsuite.ConformanceTestSuite, opts confsuite.ConformanceOptions, tests []confsuite.ConformanceTest) {
	suite.Applier.ManifestFS = suite.ManifestFS
	if suite.RunTest != "" {
		idx := slices.IndexFunc(tests, func(t confsuite.ConformanceTest) bool {
			return t.ShortName == suite.RunTest
		})

		if idx == -1 {
			require.FailNow(t, fmt.Sprintf("Test %q does not exist", suite.RunTest))
		}
	}

	tlog.Logf(t, "Test Setup: Ensuring GatewayClass has been accepted")
	suite.ControllerName = gatewayk8utils.GWCMustHaveAcceptedConditionTrue(t, suite.Client, suite.TimeoutConfig, suite.GatewayClassName)

	suite.Applier.GatewayClass = suite.GatewayClassName
	suite.Applier.ControllerName = suite.ControllerName

	tlog.Logf(t, "Test Setup: Applying base manifests")
	suite.Applier.MustApplyWithCleanup(t, suite.Client, suite.TimeoutConfig, suite.BaseManifests, suite.Cleanup)

	tlog.Logf(t, "Test Setup: Ensuring Gateways and Pods from base manifests are ready")
	namespaces := []string{
		resources.InfraNamespace,
		resources.AppBackendNamespace,
	}
	gatewayk8utils.NamespacesMustBeReady(t, suite.Client, suite.TimeoutConfig, namespaces)

	ensureGatewayAvailableAndReady(ctx, t, suite.Client, opts, resources.PrimaryGatewayNN)
	ensureGatewayAvailableAndReady(ctx, t, suite.Client, opts, resources.SecondaryGatewayNN)
}

func getGatewayInferenceExtensionVersion(crds []apiextensionsv1.CustomResourceDefinition) (string, error) {
	var inferenceVersion string
	for _, crd := range crds {
		v, okv := crd.Annotations[version.BundleVersionAnnotation]
		if !okv {
			continue
		}
		if inferenceVersion != "" && v != inferenceVersion {
			return "", errors.New("multiple gateway api inference extension CRDs versions detected")
		}
		inferenceVersion = v
	}
	if inferenceVersion == "" {
		return "", errors.New("no gateway api inference extension CRDs with the proper annotations found in the cluster")
	}
	return inferenceVersion, nil
}

// ensureGatewayAvailableAndReady polls for the specified Gateway to exist and become ready
// with an address and programmed condition.
func ensureGatewayAvailableAndReady(ctx context.Context, t *testing.T, k8sClient client.Client, opts confsuite.ConformanceOptions, gatewayNN types.NamespacedName) {
	t.Helper()

	logDebugf(t, opts.Debug, "Waiting for shared Gateway %s/%s to be ready", gatewayNN.Namespace, gatewayNN.Name)
	gatewayk8utils.GatewayMustHaveCondition(t, k8sClient, opts.TimeoutConfig, gatewayNN, gatewayk8utils.GetGatewayAcceptedCondition())
	gatewayk8utils.GatewayMustHaveCondition(t, k8sClient, opts.TimeoutConfig, gatewayNN, gatewayk8utils.GetGatewayProgrammedCondition())

	_, err := gatewayk8utils.WaitForGatewayAddress(t, k8sClient, opts.TimeoutConfig, gatewayk8utils.NewGatewayRef(gatewayNN))
	require.NoErrorf(t, err, "shared gateway %s/%s did not get an address", gatewayNN.Namespace, gatewayNN.Name)
	t.Logf("Shared Gateway %s/%s is ready.", gatewayNN.Namespace, gatewayNN.Name)
}
