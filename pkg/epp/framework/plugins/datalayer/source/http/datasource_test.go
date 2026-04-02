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

package http

import (
	"io"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/types"
)

// fakeAddressable is a test double for the Addressable interface.
type fakeAddressable struct {
	metricsHost string
}

func (f *fakeAddressable) GetIPAddress() string                    { return "" }
func (f *fakeAddressable) GetPort() string                         { return "" }
func (f *fakeAddressable) GetMetricsHost() string                  { return f.metricsHost }
func (f *fakeAddressable) GetNamespacedName() types.NamespacedName { return types.NamespacedName{Name: "pod", Namespace: "test"} }

func noopParser(r io.Reader) (any, error) { return nil, nil }

func TestGetEndpoint(t *testing.T) {
	tests := []struct {
		name        string
		metricsHost string
		metricsPort int
		wantHost    string
	}{
		{
			name:        "metricsPort=0 preserves MetricsHost unchanged",
			metricsHost: "1.2.3.4:8000",
			metricsPort: 0,
			wantHost:    "1.2.3.4:8000",
		},
		{
			name:        "metricsPort overrides port in MetricsHost",
			metricsHost: "1.2.3.4:8000",
			metricsPort: 9090,
			wantHost:    "1.2.3.4:9090",
		},
		{
			name:        "metricsPort with IPv6 address",
			metricsHost: "[::1]:8000",
			metricsPort: 9090,
			wantHost:    "[::1]:9090",
		},
		{
			name:        "metricsPort with IPv6 address, no override when port=0",
			metricsHost: "[::1]:8000",
			metricsPort: 0,
			wantHost:    "[::1]:8000",
		},
		{
			name:        "malformed host falls back to original MetricsHost",
			metricsHost: "not-a-host-with-port",
			metricsPort: 9090,
			wantHost:    "not-a-host-with-port",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ds, err := NewHTTPDataSource("http", "/metrics", false, tc.metricsPort,
				"test-type", "test-name", noopParser, reflect.TypeOf(""))
			if err != nil {
				t.Fatalf("NewHTTPDataSource() error = %v", err)
			}

			got := ds.getEndpoint(&fakeAddressable{metricsHost: tc.metricsHost})

			if diff := cmp.Diff(tc.wantHost, got.Host); diff != "" {
				t.Errorf("getEndpoint() host mismatch (-want +got):\n%s", diff)
			}
		})
	}
}
