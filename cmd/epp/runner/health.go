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
	"fmt"
	"sync/atomic"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	"google.golang.org/grpc/codes"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

type healthServer struct {
	logger                logr.Logger
	datastore             datastore.Datastore
	isLeader              *atomic.Bool
	leaderElectionEnabled bool
}

const (
	LivenessCheckService  = "liveness"
	ReadinessCheckService = "readiness"
)

func (s *healthServer) Check(ctx context.Context, in *healthPb.HealthCheckRequest) (*healthPb.HealthCheckResponse, error) {
	isLive := s.datastore.PoolHasSynced()

	// If leader election is disabled, use current logic: all checks are based on whether the pool has synced.
	if !s.leaderElectionEnabled {
		if !isLive {
			s.logger.V(logutil.DEFAULT).Info("gRPC health check not serving (leader election disabled)", "service", in.Service)
			return &healthPb.HealthCheckResponse{Status: healthPb.HealthCheckResponse_NOT_SERVING}, nil
		}
		s.logger.V(logutil.TRACE).Info("gRPC health check serving (leader election disabled)", "service", in.Service)
		return &healthPb.HealthCheckResponse{Status: healthPb.HealthCheckResponse_SERVING}, nil
	}

	// When leader election is enabled, differentiate between liveness and readiness.
	// The service name in the request determines which check to perform.
	var checkName string
	var isPassing bool

	switch in.Service {
	case ReadinessCheckService:
		checkName = "readiness"
		isPassing = isLive && s.isLeader.Load()
	case "": // Handle overall server health for load balancers that use an empty service name.
		checkName = "empty service name (considered as overall health)"
		// The overall health for a load balancer should reflect readiness to accept traffic,
		// which is true only for the leader pod that has synced its data.
		isPassing = isLive && s.isLeader.Load()
	case LivenessCheckService:
		checkName = "liveness"
		// Any pod that is running and can respond to this gRPC check is considered "live".
		// The datastore sync status should not affect liveness, only readiness.
		// This is to prevent the non-leader node from continuous restarts
		isPassing = true
	case extProcPb.ExternalProcessor_ServiceDesc.ServiceName:
		// The main service is considered ready only on the leader.
		checkName = "ext_proc"
		isPassing = isLive && s.isLeader.Load()
	default:
		s.logger.V(logutil.DEFAULT).Info("gRPC health check requested unknown service", "available-services", []string{LivenessCheckService, ReadinessCheckService, extProcPb.ExternalProcessor_ServiceDesc.ServiceName}, "requested-service", in.Service)
		return &healthPb.HealthCheckResponse{Status: healthPb.HealthCheckResponse_SERVICE_UNKNOWN}, nil
	}

	if !isPassing {
		s.logger.V(logutil.DEFAULT).Info(fmt.Sprintf("gRPC %s check not serving", checkName), "service", in.Service, "isLive", isLive, "isLeader", s.isLeader.Load())
		return &healthPb.HealthCheckResponse{Status: healthPb.HealthCheckResponse_NOT_SERVING}, nil
	}

	s.logger.V(logutil.TRACE).Info(fmt.Sprintf("gRPC %s check serving", checkName), "service", in.Service)
	return &healthPb.HealthCheckResponse{Status: healthPb.HealthCheckResponse_SERVING}, nil
}

func (s *healthServer) List(ctx context.Context, _ *healthPb.HealthListRequest) (*healthPb.HealthListResponse, error) {
	statuses := make(map[string]*healthPb.HealthCheckResponse)

	services := []string{extProcPb.ExternalProcessor_ServiceDesc.ServiceName}
	if s.leaderElectionEnabled {
		services = append(services, LivenessCheckService, ReadinessCheckService)
	}

	for _, service := range services {
		resp, err := s.Check(ctx, &healthPb.HealthCheckRequest{Service: service})
		if err != nil {
			// Check can return an error for unknown services, but here we are iterating known services.
			// If another error occurs, we should probably return it.
			return nil, err
		}
		statuses[service] = resp
	}

	return &healthPb.HealthListResponse{
		Statuses: statuses,
	}, nil
}

func (s *healthServer) Watch(in *healthPb.HealthCheckRequest, srv healthPb.Health_WatchServer) error {
	return status.Error(codes.Unimplemented, "Watch is not implemented")
}
