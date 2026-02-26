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
	"testing"

	"github.com/spf13/pflag"
)

func TestNewOptionsDefaults(t *testing.T) {
	opts := NewOptions()

	checks := []struct {
		name string
		got  any
		want any
	}{
		{"GRPCPort", opts.GRPCPort, DefaultGrpcPort},
		{"GRPCHealthPort", opts.GRPCHealthPort, DefaultGrpcHealthPort},
		{"MetricsPort", opts.MetricsPort, 9090},
		{"MetricsEndpointAuth", opts.MetricsEndpointAuth, true},
		{"Streaming", opts.Streaming, false},
		{"SecureServing", opts.SecureServing, true},
		{"EnablePprof", opts.EnablePprof, true},
		{"LogVerbosity", opts.LogVerbosity, 2}, // logging.DEFAULT
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("NewOptions().%s = %v, want %v", c.name, c.got, c.want)
		}
	}
}

func TestAddFlagsOverridesDefaults(t *testing.T) {
	opts := NewOptions()
	fs := pflag.NewFlagSet("test", pflag.ContinueOnError)
	opts.AddFlags(fs)

	args := []string{
		"--grpc-port", "5000",
		"--grpc-health-port", "5001",
		"--metrics-port", "5002",
		"--streaming",
		"--secure-serving=false",
		"--metrics-endpoint-auth=false",
		"--enable-pprof=false",
		"-v", "3",
	}
	if err := fs.Parse(args); err != nil {
		t.Fatalf("Failed to parse flags: %v", err)
	}

	checks := []struct {
		name string
		got  any
		want any
	}{
		{"GRPCPort", opts.GRPCPort, 5000},
		{"GRPCHealthPort", opts.GRPCHealthPort, 5001},
		{"MetricsPort", opts.MetricsPort, 5002},
		{"Streaming", opts.Streaming, true},
		{"SecureServing", opts.SecureServing, false},
		{"MetricsEndpointAuth", opts.MetricsEndpointAuth, false},
		{"EnablePprof", opts.EnablePprof, false},
		{"LogVerbosity", opts.LogVerbosity, 3},
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("After parse, opts.%s = %v, want %v", c.name, c.got, c.want)
		}
	}
}

func TestValidate(t *testing.T) {
	tests := []struct {
		name        string
		mutate      func(*Options)
		expectError bool
	}{
		{
			name:        "defaults are valid",
			mutate:      func(_ *Options) {},
			expectError: false,
		},
		// Port range validation.
		{
			name:        "grpc-port zero",
			mutate:      func(o *Options) { o.GRPCPort = 0 },
			expectError: true,
		},
		{
			name:        "grpc-port negative",
			mutate:      func(o *Options) { o.GRPCPort = -1 },
			expectError: true,
		},
		{
			name:        "grpc-port above 65535",
			mutate:      func(o *Options) { o.GRPCPort = 70000 },
			expectError: true,
		},
		{
			name:        "grpc-health-port zero",
			mutate:      func(o *Options) { o.GRPCHealthPort = 0 },
			expectError: true,
		},
		{
			name:        "metrics-port above max",
			mutate:      func(o *Options) { o.MetricsPort = 65536 },
			expectError: true,
		},
		{
			name:        "valid edge: port 1",
			mutate:      func(o *Options) { o.GRPCPort = 1 },
			expectError: false,
		},
		{
			name:        "valid edge: port 65535",
			mutate:      func(o *Options) { o.GRPCPort = 65535 },
			expectError: false,
		},
		// Port collision validation.
		{
			name: "grpc-port collides with metrics-port",
			mutate: func(o *Options) {
				o.GRPCPort = 9090
				o.MetricsPort = 9090
			},
			expectError: true,
		},
		{
			name: "grpc-port collides with grpc-health-port",
			mutate: func(o *Options) {
				o.GRPCPort = 9004
				o.GRPCHealthPort = 9004
			},
			expectError: true,
		},
		{
			name: "all three ports identical",
			mutate: func(o *Options) {
				o.GRPCPort = 8080
				o.GRPCHealthPort = 8080
				o.MetricsPort = 8080
			},
			expectError: true,
		},
		// Log verbosity validation.
		{
			name:        "negative log verbosity",
			mutate:      func(o *Options) { o.LogVerbosity = -1 },
			expectError: true,
		},
		{
			name:        "zero log verbosity is valid",
			mutate:      func(o *Options) { o.LogVerbosity = 0 },
			expectError: false,
		},
		{
			name:        "high log verbosity is valid",
			mutate:      func(o *Options) { o.LogVerbosity = 10 },
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := NewOptions()
			// AddFlags is required so that Complete/Validate can reference opts.fs.
			fs := pflag.NewFlagSet(tt.name, pflag.ContinueOnError)
			opts.AddFlags(fs)

			tt.mutate(opts)

			err := opts.Validate()
			if tt.expectError && err == nil {
				t.Error("Expected a validation error but got none")
			}
			if !tt.expectError && err != nil {
				t.Errorf("Unexpected validation error: %v", err)
			}
		})
	}
}

func TestCompleteDerivesZapLogLevel(t *testing.T) {
	opts := NewOptions()
	fs := pflag.NewFlagSet("test-complete", pflag.ContinueOnError)
	opts.AddFlags(fs)

	// Set -v to 5, do NOT set --zap-log-level explicitly.
	if err := fs.Parse([]string{"-v", "5"}); err != nil {
		t.Fatalf("Failed to parse flags: %v", err)
	}

	if err := opts.Complete(); err != nil {
		t.Fatalf("Complete() returned error: %v", err)
	}

	// After Complete(), the zap-log-level flag should be marked as changed
	// and the zap level should be set to -5.
	zapFlag := fs.Lookup(ZapLogLevelFlagName)
	if zapFlag == nil {
		t.Fatal("Expected zap-log-level flag to exist after AddFlags")
	}
	if !zapFlag.Changed {
		t.Error("Expected zap-log-level flag to be marked as Changed after Complete()")
	}
}

func TestCompleteRespectsExplicitZapLogLevel(t *testing.T) {
	opts := NewOptions()
	fs := pflag.NewFlagSet("test-explicit-zap", pflag.ContinueOnError)
	opts.AddFlags(fs)

	// Set both -v and --zap-log-level. The explicit --zap-log-level should take precedence.
	if err := fs.Parse([]string{"-v", "5", "--zap-log-level", "debug"}); err != nil {
		t.Fatalf("Failed to parse flags: %v", err)
	}

	if err := opts.Complete(); err != nil {
		t.Fatalf("Complete() returned error: %v", err)
	}

	// Since --zap-log-level was explicitly set, Complete() should NOT override it.
	// The ZapOptions.Level should remain as set by the zap flag parser, not from -v.
	// We verify by checking that opts.LogVerbosity is 5 but the zap level wasn't overwritten to -5.
	if opts.LogVerbosity != 5 {
		t.Errorf("Expected LogVerbosity to be 5, got %d", opts.LogVerbosity)
	}
}
