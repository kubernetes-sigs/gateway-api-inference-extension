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

package datalayer

import (
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/go-logr/logr/funcr"
)

// NewTestRuntime creates a Runtime configured with a FakeDataSource for testing.
// It configures the runtime but does not start it.
// Returns an EndpointFactory for use in tests.
func NewTestRuntime(t *testing.T, refreshInterval time.Duration) EndpointFactory {
	return NewTestRuntimeWithConfig(t, refreshInterval, &Config{
		Sources: []DataSourceConfig{
			{
				Plugin: &FakeDataSource{},
			},
		},
	})
}

// NewTestRuntimeWithConfig creates a Runtime with the given config for testing.
// It configures the runtime but does not start it.
// Returns an EndpointFactory for use in tests.
func NewTestRuntimeWithConfig(t *testing.T, refreshInterval time.Duration, cfg *Config) EndpointFactory {
	r := NewRuntime(refreshInterval)
	logger := newTestLogger(t)
	_ = r.Configure(cfg, false, "", logger)
	return r
}

// newTestLogger creates a test logger from a *testing.T
func newTestLogger(t *testing.T) logr.Logger {
	return funcr.New(func(prefix, args string) {
		t.Logf("%s%s", prefix, args)
	}, funcr.Options{
		Verbosity: 10,
	})
}
