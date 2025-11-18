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

package requestcontrol

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

var _ PrepareDataPlugin = &mockPrepareRequestDataPlugin{}

type mockPrepareRequestDataPlugin struct {
	name      string
	delay     time.Duration
	returnErr error
	executed  bool
}

func (m *mockPrepareRequestDataPlugin) TypedName() plugins.TypedName {
	return plugins.TypedName{Type: "mock", Name: m.name}
}

func (m *mockPrepareRequestDataPlugin) PrepareRequestData(ctx context.Context, request *schedulingtypes.LLMRequest, pods []schedulingtypes.Pod) error {
	m.executed = true
	if m.delay > 0 {
		select {
		case <-time.After(m.delay):
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return m.returnErr
}

func (m *mockPrepareRequestDataPlugin) Produces() map[string]any {
	return nil
}

func (m *mockPrepareRequestDataPlugin) Consumes() map[string]any {
	return nil
}

func TestPrepareDataPluginsWithTimeout(t *testing.T) {
	testCases := []struct {
		name          string
		timeout       time.Duration
		plugins       []PrepareDataPlugin
		ctxFn         func() (context.Context, context.CancelFunc)
		expectErrStr  string
		checkPlugins  func(t *testing.T, plugins []PrepareDataPlugin)
		expectSuccess bool
	}{
		{
			name:    "success with one plugin",
			timeout: 100 * time.Millisecond,
			plugins: []PrepareDataPlugin{
				&mockPrepareRequestDataPlugin{name: "p1"},
			},
			ctxFn: func() (context.Context, context.CancelFunc) {
				return context.Background(), func() {}
			},
			expectSuccess: true,
			checkPlugins: func(t *testing.T, plugins []PrepareDataPlugin) {
				assert.True(t, plugins[0].(*mockPrepareRequestDataPlugin).executed)
			},
		},
		{
			name:    "plugin returns error",
			timeout: 100 * time.Millisecond,
			plugins: []PrepareDataPlugin{
				&mockPrepareRequestDataPlugin{name: "p1", returnErr: errors.New("plugin failed")},
			},
			ctxFn: func() (context.Context, context.CancelFunc) {
				return context.Background(), func() {}
			},
			expectErrStr: "prepare data plugin p1/mock failed: plugin failed",
		},
		{
			name:    "plugins time out",
			timeout: 50 * time.Millisecond,
			plugins: []PrepareDataPlugin{
				&mockPrepareRequestDataPlugin{name: "p1", delay: 100 * time.Millisecond},
			},
			ctxFn: func() (context.Context, context.CancelFunc) {
				return context.Background(), func() {}
			},
			expectErrStr: "prepare data plugin timed out",
		},
		{
			name:    "context cancelled",
			timeout: 200 * time.Millisecond,
			plugins: []PrepareDataPlugin{
				&mockPrepareRequestDataPlugin{name: "p1", delay: 100 * time.Millisecond},
			},
			ctxFn: func() (context.Context, context.CancelFunc) {
				ctx, cancel := context.WithCancel(context.Background())
				time.AfterFunc(50*time.Millisecond, cancel)
				return ctx, cancel
			},
			expectErrStr: "context canceled",
		},
		{
			name:    "multiple plugins success",
			timeout: 100 * time.Millisecond,
			plugins: []PrepareDataPlugin{
				&mockPrepareRequestDataPlugin{name: "p1"},
				&mockPrepareRequestDataPlugin{name: "p2"},
			},
			ctxFn: func() (context.Context, context.CancelFunc) {
				return context.Background(), func() {}
			},
			expectSuccess: true,
			checkPlugins: func(t *testing.T, plugins []PrepareDataPlugin) {
				assert.True(t, plugins[0].(*mockPrepareRequestDataPlugin).executed)
				assert.True(t, plugins[1].(*mockPrepareRequestDataPlugin).executed)
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cancel := tc.ctxFn()
			defer cancel()

			err := prepareDataPluginsWithTimeout(tc.timeout, tc.plugins, ctx, &schedulingtypes.LLMRequest{}, nil)

			if tc.expectSuccess {
				assert.NoError(t, err)
			} else {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tc.expectErrStr)
			}

			if tc.checkPlugins != nil {
				tc.checkPlugins(t, tc.plugins)
			}
		})
	}
}
