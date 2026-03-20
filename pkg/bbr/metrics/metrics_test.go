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

package metrics

import (
	"os"
	"testing"
	"time"

	"k8s.io/component-base/metrics/testutil"
	crmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"
)

func TestPluginProcessingLatency(t *testing.T) {
	Register()

	type pluginLatency struct {
		extensionPoint string
		pluginType     string
		pluginName     string
		duration       time.Duration
	}

	scenarios := []struct {
		name      string
		latencies []pluginLatency
	}{
		{
			name: "multiple plugins",
			latencies: []pluginLatency{
				{
					extensionPoint: "Request",
					pluginType:     "TestPluginA",
					pluginName:     "PluginA",
					duration:       5 * time.Millisecond,
				},
				{
					extensionPoint: "Request",
					pluginType:     "TestPluginB",
					pluginName:     "PluginB",
					duration:       10 * time.Microsecond,
				},
			},
		},
	}
	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			for _, latency := range scenario.latencies {
				RecordPluginProcessingLatency(latency.extensionPoint, latency.pluginType, latency.pluginName, latency.duration)
			}

			wantPluginLatencies, err := os.Open("testdata/plugin_processing_latencies_metric")
			defer func() {
				if err := wantPluginLatencies.Close(); err != nil {
					t.Error(err)
				}
			}()
			if err != nil {
				t.Fatal(err)
			}
			if err := testutil.GatherAndCompare(crmetrics.Registry, wantPluginLatencies, "bbr_plugin_duration_seconds"); err != nil {
				t.Error(err)
			}
		})
	}
}
