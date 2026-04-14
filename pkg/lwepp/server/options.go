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

package server

import (
	"errors"
	"fmt"

	"github.com/spf13/pflag"
	"k8s.io/apimachinery/pkg/util/sets"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
)

const (
	DefaultGrpcPort      = 9002
	DefaultPoolNamespace = "default" // default when pool namespace is empty (CLI flag default is empty)
)

// Options contains configuration values necessary to create and run the EPP.
type Options struct {
	//
	// ext_proc configuration.
	//
	GRPCPort             int  // gRPC port used for communicating with Envoy proxy. (TODO: uint16?)
	EnableLeaderElection bool // Enables leader election for high availability
	//
	// InferencePool.
	//
	PoolGroup     string // Kubernetes resource group of the InferencePool this Endpoint Picker is associated with.
	PoolNamespace string // Namespace of the InferencePool this Endpoint Picker is associated with.
	PoolName      string // Name of the InferencePool this Endpoint Picker is associated with.
	//
	// Endpoints (in lieu of using an InferencePool for service discovery).
	//
	EndpointSelector            string // Selector to filter model server pods on, only 'key=value' pairs are supported. (TODO: k8s.Selector, pflag.StringSlice?)
	EndpointTargetPorts         []int  // Target ports of model server pods.
	DisableEndpointSubsetFilter bool   // Disables respecting x-gateway-destination-endpoint-subset in EPP.

	//
	// Diagnostics.
	//
	logging.LoggingOptions        // Logging configuration.
	Tracing                bool   // Enables emitting traces.
	HealthChecking         bool   // Enables health checking.
	MetricsPort            int    // The metrics port exposed by EPP. (TODO: uint16)
	GRPCHealthPort         int    // The port used for gRPC liveness and readiness probes. (TODO: uint16)
	EnablePprof            bool   // Enables pprof handlers.
	CertPath               string // The path to the certificate for secure serving.
	EnableCertReload       bool   // Enables certificate reloading of the certificates specified in --cert-path.
	SecureServing          bool   // Enables secure serving.
	MetricsEndpointAuth    bool   // Enables authentication and authorization of the metrics endpoint.
	//
	// Configuration.
	//
	ConfigFile string // The path to the configuration file.
	ConfigText string // The configuration specified as text, in lieu of a file.

	// internal
	fs *pflag.FlagSet // FlagSet used in AddFlags() and consulted in Validate()
}

// NewOptions returns a new Options struct initialized with the default values.
func NewOptions() *Options {
	return &Options{ // "zero" values are no explicitly set
		GRPCPort:                    DefaultGrpcPort,
		PoolGroup:                   "inference.networking.k8s.io",
		EndpointTargetPorts:         []int{},
		DisableEndpointSubsetFilter: false,

		LoggingOptions:      *logging.NewOptions(),
		Tracing:             true,
		MetricsPort:         9090,
		GRPCHealthPort:      9003,
		EnablePprof:         true,
		SecureServing:       true,
		MetricsEndpointAuth: true,
	}
}

func (opts *Options) AddFlags(fs *pflag.FlagSet) {
	if fs == nil {
		fs = pflag.CommandLine
	}
	opts.fs = fs

	fs.IntVar(&opts.GRPCPort, "grpc-port", opts.GRPCPort, "gRPC port used for communicating with Envoy proxy.")
	fs.BoolVar(&opts.EnableLeaderElection, "ha-enable-leader-election", opts.EnableLeaderElection,
		"Enables leader election for high availability. When enabled, readiness probes will only pass on the leader.")
	fs.StringVar(&opts.PoolGroup, "pool-group", opts.PoolGroup,
		"Kubernetes resource group of the InferencePool this Endpoint Picker is associated with. Only `inference.networking.k8s.io/v1` is currently supported.")
	fs.StringVar(&opts.PoolNamespace, "pool-namespace", opts.PoolNamespace,
		"Namespace of the InferencePool this Endpoint Picker is associated with.")
	fs.StringVar(&opts.PoolName, "pool-name", opts.PoolName, "Name of the InferencePool this Endpoint Picker is associated with.")
	fs.StringVar(&opts.EndpointSelector, "endpoint-selector", opts.EndpointSelector,
		"Selector to filter model server pods on, only 'key=value' pairs are supported. "+
			"Format: a comma-separated list of key=value pairs without whitespace (e.g., 'app=vllm-qwen3-32b,env=prod').")
	fs.IntSliceVar(&opts.EndpointTargetPorts, "endpoint-target-ports", opts.EndpointTargetPorts, "Target ports of model server pods. "+
		"Format: a comma-separated list of numbers without whitespace (e.g., '3000,3001,3002').")
	fs.BoolVar(&opts.DisableEndpointSubsetFilter, "disable-endpoint-subset-filter", opts.DisableEndpointSubsetFilter,
		"Disables respecting the x-gateway-destination-endpoint-subset metadata for dispatching requests in EPP.")

	opts.LoggingOptions.AddFlags(fs) // Add logging flags.

	fs.BoolVar(&opts.Tracing, "tracing", opts.Tracing, "Enables emitting traces.")
	fs.BoolVar(&opts.HealthChecking, "health-checking", opts.HealthChecking, "Enables health checking.")
	fs.IntVar(&opts.MetricsPort, "metrics-port", opts.MetricsPort, "The metrics port exposed by EPP.")
	fs.IntVar(&opts.GRPCHealthPort, "grpc-health-port", opts.GRPCHealthPort,
		"The port used for gRPC liveness and readiness probes.")
	fs.BoolVar(&opts.EnablePprof, "enable-pprof", opts.EnablePprof,
		"Enables pprof handlers. Defaults to true. Set to false to disable pprof handlers.")
	fs.StringVar(&opts.CertPath, "cert-path", opts.CertPath,
		"The path to the certificate for secure serving. The certificate and private key files "+
			"are assumed to be named tls.crt and tls.key, respectively. If not set, and secureServing is enabled, "+
			"then a self-signed certificate is used.")
	fs.BoolVar(&opts.EnableCertReload, "enable-cert-reload", opts.EnableCertReload,
		"Enables certificate reloading of the certificates specified in --cert-path.")
	fs.BoolVar(&opts.SecureServing, "secure-serving", opts.SecureServing, "Enables secure serving.")
	fs.BoolVar(&opts.MetricsEndpointAuth, "metrics-endpoint-auth", opts.MetricsEndpointAuth,
		"Enables authentication and authorization of the metrics endpoint.")
	fs.StringVar(&opts.ConfigFile, "config-file", opts.ConfigFile, "The path to the configuration file.")
	fs.StringVar(&opts.ConfigText, "config-text", opts.ConfigText, "The configuration specified as text, in lieu of a file.")
}

func (opts *Options) Complete() error {
	// TODO: postprocessing or command line arguments. For example, convert EndpointSelector
	// from raw string to k8s.LabelSelector, load ConfigFile into ConfigText, etc.

	opts.EndpointTargetPorts = removeDuplicatePorts(opts.EndpointTargetPorts)

	// Complete logging options.
	return opts.LoggingOptions.Complete()
}

func (opts *Options) Validate() error {
	if (opts.PoolName != "" && opts.EndpointSelector != "") || (opts.PoolName == "" && opts.EndpointSelector == "") {
		return errors.New("either pool-name or endpoint-selector must be set")
	}
	if opts.EndpointSelector != "" {
		if len(opts.EndpointTargetPorts) == 0 || len(opts.EndpointTargetPorts) > 8 {
			return fmt.Errorf("flag %q should have length from 1 to 8", "endpoint-target-ports")
		}
		for _, port := range opts.EndpointTargetPorts { // valid port range
			if port < 0 || port > 65535 {
				return fmt.Errorf("invalid port number %d in %q", port, "endpoint-target-ports")
			}
		}
	}

	if opts.ConfigText != "" && opts.ConfigFile != "" {
		return fmt.Errorf("both the %q and %q flags can not be set at the same time", "configText", "configFile")
	}

	// Validate logging options.
	if err := opts.LoggingOptions.Validate(); err != nil {
		return err
	}

	return nil
}

func removeDuplicatePorts(ports []int) []int {
	seen := sets.NewInt()
	unique := make([]int, 0, len(ports))

	for _, val := range ports {
		if !seen.Has(val) {
			unique = append(unique, val)
			seen.Insert(val)
		}
	}
	return unique
}
