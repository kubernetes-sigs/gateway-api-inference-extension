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
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/runtime/schema"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
)

func TestRuntimeStateTransitions(t *testing.T) {
	logger := newTestLogger(t)

	tests := []struct {
		name       string
		ops        func(*Runtime) error
		wantErr    bool
		finalState RuntimeState
	}{
		{
			name: "configure twice returns error",
			ops: func(r *Runtime) error {
				cfg := &Config{
					Sources: []DataSourceConfig{
						{Plugin: &FakeDataSource{}},
					},
				}
				if err := r.Configure(cfg, false, "", logger); err != nil {
					return err
				}
				return r.Configure(cfg, false, "", logger)
			},
			wantErr:    true,
			finalState: StateError,
		},
		{
			name: "start before configure returns error",
			ops: func(r *Runtime) error {
				return r.Start(context.TODO(), nil)
			},
			wantErr:    true,
			finalState: StateError,
		},
		{
			name: "stop before start returns error",
			ops: func(r *Runtime) error {
				_ = r.Configure(&Config{}, false, "", logger)
				return r.Stop()
			},
			wantErr:    true,
			finalState: StateError,
		},
		{
			name: "successful configure reaches StateConfigured",
			ops: func(r *Runtime) error {
				return r.Configure(&Config{}, false, "", logger)
			},
			wantErr:    false,
			finalState: StateConfigured,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := NewRuntime(time.Second)
			err := tc.ops(r)
			if tc.wantErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
			}
			assert.Equal(t, tc.finalState, RuntimeState(r.state.Load()))
		})
	}
}

func TestRuntimeConfigureWithNilConfig(t *testing.T) {
	logger := newTestLogger(t)

	tests := []struct {
		name          string
		cfg           *Config
		enableMetrics bool
		wantErr       bool
	}{
		{
			name:    "nil config with enableNewMetrics false succeeds",
			cfg:     nil,
			wantErr: false,
		},
		{
			name:    "empty config with enableNewMetrics false succeeds",
			cfg:     &Config{},
			wantErr: false,
		},
		{
			name:          "empty config with enableNewMetrics true returns error",
			cfg:           &Config{},
			enableMetrics: true,
			wantErr:       true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := NewRuntime(time.Second)
			err := r.Configure(tc.cfg, tc.enableMetrics, "", logger)
			if tc.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, StateConfigured, RuntimeState(r.state.Load()))
			}
		})
	}
}

func TestRuntimeConfigureWithSources(t *testing.T) {
	logger := newTestLogger(t)

	tests := []struct {
		name    string
		sources []DataSourceConfig
		wantErr bool
	}{
		{
			name: "single polling source succeeds",
			sources: []DataSourceConfig{
				{Plugin: &FakeDataSource{}},
			},
			wantErr: false,
		},
		{
			name: "notification source succeeds",
			sources: []DataSourceConfig{
				{Plugin: &FakeNotificationSource{
					gvk: schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"},
				}},
			},
			wantErr: false,
		},
		{
			name: "source with extractor succeeds",
			sources: []DataSourceConfig{
				{
					Plugin:     &FakeDataSource{},
					Extractors: []fwkdl.Extractor{},
				},
			},
			wantErr: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			r := NewRuntime(time.Second)
			cfg := &Config{Sources: tc.sources}
			err := r.Configure(cfg, false, "", logger)
			if tc.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, StateConfigured, RuntimeState(r.state.Load()))
			}
		})
	}
}
