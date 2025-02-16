package main

import (
	"context"

	"github.com/go-logr/logr"
	"google.golang.org/grpc/codes"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/backend"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/logging"
)

type healthServer struct {
	logger    logr.Logger
	datastore backend.Datastore
}

func (s *healthServer) Check(ctx context.Context, in *healthPb.HealthCheckRequest) (*healthPb.HealthCheckResponse, error) {
	if !s.datastore.PoolHasSynced() {
		s.logger.V(logutil.VERBOSE).Info("gRPC health check not serving", "service", in.Service)
		return &healthPb.HealthCheckResponse{Status: healthPb.HealthCheckResponse_NOT_SERVING}, nil
	}
	s.logger.V(logutil.VERBOSE).Info("gRPC health check serving", "service", in.Service)
	return &healthPb.HealthCheckResponse{Status: healthPb.HealthCheckResponse_SERVING}, nil
}

func (s *healthServer) Watch(in *healthPb.HealthCheckRequest, srv healthPb.Health_WatchServer) error {
	return status.Error(codes.Unimplemented, "Watch is not implemented")
}
