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

package runner

import (
	"context"
	"sync/atomic"
	"testing"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr/testr"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"

	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
)

// Mock Datastore
type mockDatastore struct {
	datastore.Datastore
	hasSynced bool
	pool      *datalayer.EndpointPool
	err       error
}

func (m *mockDatastore) PoolHasSynced() bool {
	return m.hasSynced
}

func (m *mockDatastore) PoolGet() (*datalayer.EndpointPool, error) {
	return m.pool, m.err
}

// Mock AppProtocolSupporter
type mockSupporter struct {
	protocols []v1.AppProtocol
}

func (m *mockSupporter) SupportedAppProtocols() []v1.AppProtocol {
	return m.protocols
}

func TestHealthServer_Check(t *testing.T) {
	tests := []struct {
		name                  string
		leaderElectionEnabled bool
		isLeader              bool
		hasSynced             bool
		pool                  *datalayer.EndpointPool
		poolErr               error
		supporter             *mockSupporter
		service               string
		wantStatus            healthPb.HealthCheckResponse_ServingStatus
	}{
		{
			name:                  "LeaderElectionDisabled_Live_ProtocolMatches",
			leaderElectionEnabled: false,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolHTTP},
			wantStatus:            healthPb.HealthCheckResponse_SERVING,
		},
		{
			name:                  "LeaderElectionDisabled_NotLive",
			leaderElectionEnabled: false,
			hasSynced:             false,
			wantStatus:            healthPb.HealthCheckResponse_NOT_SERVING,
		},
		{
			name:                  "LeaderElectionDisabled_ProtocolMismatch",
			leaderElectionEnabled: false,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolH2C},
			supporter:             &mockSupporter{protocols: []v1.AppProtocol{v1.AppProtocolHTTP}},
			wantStatus:            healthPb.HealthCheckResponse_NOT_SERVING,
		},
		{
			name:                  "LeaderElectionEnabled_Liveness_AlwaysServing",
			leaderElectionEnabled: true,
			service:               LivenessCheckService,
			wantStatus:            healthPb.HealthCheckResponse_SERVING,
		},
		{
			name:                  "LeaderElectionEnabled_Readiness_Live_Leader_ProtocolMatches",
			leaderElectionEnabled: true,
			isLeader:              true,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolHTTP},
			service:               ReadinessCheckService,
			wantStatus:            healthPb.HealthCheckResponse_SERVING,
		},
		{
			name:                  "LeaderElectionEnabled_Readiness_NotLive",
			leaderElectionEnabled: true,
			isLeader:              true,
			hasSynced:             false,
			service:               ReadinessCheckService,
			wantStatus:            healthPb.HealthCheckResponse_NOT_SERVING,
		},
		{
			name:                  "LeaderElectionEnabled_Readiness_NotLeader",
			leaderElectionEnabled: true,
			isLeader:              false,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolHTTP},
			service:               ReadinessCheckService,
			wantStatus:            healthPb.HealthCheckResponse_NOT_SERVING,
		},
		{
			name:                  "LeaderElectionEnabled_Readiness_ProtocolMismatch",
			leaderElectionEnabled: true,
			isLeader:              true,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolH2C},
			supporter:             &mockSupporter{protocols: []v1.AppProtocol{v1.AppProtocolHTTP}},
			service:               ReadinessCheckService,
			wantStatus:            healthPb.HealthCheckResponse_NOT_SERVING,
		},
		{
			name:                  "LeaderElectionEnabled_EmptyService_ReflectsReadiness_Serving",
			leaderElectionEnabled: true,
			isLeader:              true,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolHTTP},
			service:               "",
			wantStatus:            healthPb.HealthCheckResponse_SERVING,
		},
		{
			name:                  "LeaderElectionEnabled_ExtProc_ReflectsReadiness_Serving",
			leaderElectionEnabled: true,
			isLeader:              true,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolHTTP},
			service:               extProcPb.ExternalProcessor_ServiceDesc.ServiceName,
			wantStatus:            healthPb.HealthCheckResponse_SERVING,
		},
		{
			name:                  "LeaderElectionEnabled_UnknownService",
			leaderElectionEnabled: true,
			service:               "unknown",
			wantStatus:            healthPb.HealthCheckResponse_SERVICE_UNKNOWN,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger := testr.New(t)
			ds := &mockDatastore{
				hasSynced: tt.hasSynced,
				pool:      tt.pool,
				err:       tt.poolErr,
			}
			var isLeader atomic.Bool
			isLeader.Store(tt.isLeader)

			var supporter appProtocolSupporter
			if tt.supporter != nil {
				supporter = tt.supporter
			}

			s := &healthServer{
				logger:                logger,
				datastore:             ds,
				isLeader:              &isLeader,
				leaderElectionEnabled: tt.leaderElectionEnabled,
				supporter:             supporter,
			}

			resp, err := s.Check(context.Background(), &healthPb.HealthCheckRequest{Service: tt.service})
			if err != nil {
				t.Fatalf("Check failed: %v", err)
			}
			if resp.Status != tt.wantStatus {
				t.Errorf("Check() status = %v, want %v", resp.Status, tt.wantStatus)
			}
		})
	}
}
