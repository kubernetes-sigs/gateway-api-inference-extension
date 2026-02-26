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
	"flag"
	"fmt"

	"github.com/spf13/pflag"
	uberzap "go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/config"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
)

const (
	DefaultGrpcPort       = 9004
	DefaultGrpcHealthPort = 9005
	ZapLogLevelFlagName   = "zap-log-level"
)

// Options contains the command-line configuration for the BBR server.
type Options struct {
	//
	// ext_proc configuration.
	//
	GRPCPort  int  // gRPC port for communicating with Envoy proxy.
	Streaming bool // Enables streaming support for Envoy full-duplex streaming mode.
	//
	// Diagnostics.
	//
	LogVerbosity        int         // Number for the log level verbosity.
	ZapOptions          zap.Options // Zap logging options.
	MetricsPort         int         // The metrics port exposed by BBR.
	GRPCHealthPort      int         // The port for gRPC liveness and readiness probes.
	EnablePprof         bool        // Enables pprof handlers.
	SecureServing       bool        // Enables secure serving.
	MetricsEndpointAuth bool        // Enables authentication and authorization of the metrics endpoint.
	//
	// Plugins.
	//
	PluginSpecs config.BBRPluginSpecs // Repeatable --plugin <type>:<name>[:<json>] flag values.

	// internal
	fs *pflag.FlagSet // FlagSet used in AddFlags() and consulted in Complete()
}

// NewOptions returns a new Options struct initialized with default values.
func NewOptions() *Options {
	return &Options{
		GRPCPort:            DefaultGrpcPort,
		GRPCHealthPort:      DefaultGrpcHealthPort,
		LogVerbosity:        logging.DEFAULT,
		ZapOptions:          zap.Options{Development: true},
		MetricsPort:         9090,
		EnablePprof:         true,
		SecureServing:       true,
		MetricsEndpointAuth: true,
	}
}

// AddFlags binds the Options fields to command-line flags on the given FlagSet.
func (opts *Options) AddFlags(fs *pflag.FlagSet) {
	if fs == nil {
		fs = pflag.CommandLine
	}
	opts.fs = fs

	fs.IntVar(&opts.GRPCPort, "grpc-port", opts.GRPCPort,
		"The gRPC port used for communicating with Envoy proxy.")
	fs.IntVar(&opts.GRPCHealthPort, "grpc-health-port", opts.GRPCHealthPort,
		"The port used for gRPC liveness and readiness probes.")
	fs.IntVar(&opts.MetricsPort, "metrics-port", opts.MetricsPort,
		"The metrics port exposed by BBR.")
	fs.BoolVar(&opts.MetricsEndpointAuth, "metrics-endpoint-auth", opts.MetricsEndpointAuth,
		"Enables authentication and authorization of the metrics endpoint.")
	fs.BoolVar(&opts.Streaming, "streaming", opts.Streaming,
		"Enables streaming support for Envoy full-duplex streaming mode.")
	fs.BoolVar(&opts.SecureServing, "secure-serving", opts.SecureServing,
		"Enables secure serving.")
	fs.IntVarP(&opts.LogVerbosity, "v", "v", opts.LogVerbosity,
		"Number for the log level verbosity.")
	fs.BoolVar(&opts.EnablePprof, "enable-pprof", opts.EnablePprof,
		"Enables pprof handlers. Defaults to true. Set to false to disable pprof handlers.")

	fs.Var(&opts.PluginSpecs, "plugin", `Repeatable. --plugin <type>:<name>[:<json>]`)

	// Bind zap flags (zap expects a standard Go FlagSet; pflag.FlagSet is not compatible).
	gofs := flag.NewFlagSet("zap", flag.ExitOnError)
	opts.ZapOptions.BindFlags(gofs)
	fs.AddGoFlagSet(gofs)
}

// Complete performs post-processing of parsed command-line arguments.
func (opts *Options) Complete() error {
	// Derive the zap log level from the -v flag when --zap-log-level is not set explicitly.
	zapLogLevelFlag := opts.fs.Lookup(ZapLogLevelFlagName)
	if zapLogLevelFlag != nil && !zapLogLevelFlag.Changed {
		// See https://pkg.go.dev/sigs.k8s.io/controller-runtime/pkg/log/zap#Options.Level
		lvl := -1 * (opts.LogVerbosity)
		opts.ZapOptions.Level = uberzap.NewAtomicLevelAt(zapcore.Level(int8(lvl)))
		zapLogLevelFlag.Changed = true
	}
	return nil
}

// Validate checks the Options for invalid or conflicting values.
func (opts *Options) Validate() error {
	// Validate port ranges.
	for _, pc := range []struct {
		name string
		port int
	}{
		{"grpc-port", opts.GRPCPort},
		{"grpc-health-port", opts.GRPCHealthPort},
		{"metrics-port", opts.MetricsPort},
	} {
		if pc.port < 1 || pc.port > 65535 {
			return fmt.Errorf("invalid value %d for flag %q: must be between 1 and 65535", pc.port, pc.name)
		}
	}

	// Validate that the three server ports do not collide.
	ports := map[int]string{
		opts.GRPCPort:       "grpc-port",
		opts.GRPCHealthPort: "grpc-health-port",
		opts.MetricsPort:    "metrics-port",
	}
	if len(ports) < 3 {
		return fmt.Errorf("port conflict: grpc-port (%d), grpc-health-port (%d), and metrics-port (%d) must all be different",
			opts.GRPCPort, opts.GRPCHealthPort, opts.MetricsPort)
	}

	// Validate log verbosity is non-negative.
	if opts.LogVerbosity < 0 {
		return fmt.Errorf("invalid value %d for flag %q: must be >= 0", opts.LogVerbosity, "v")
	}

	return nil
}
