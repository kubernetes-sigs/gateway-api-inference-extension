package main

import (
	"context"

	"google.golang.org/grpc/codes"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/backend"
	klog "k8s.io/klog/v2"
)

type healthServer struct {
	datastore *backend.K8sDatastore
}

func (s *healthServer) Check(ctx context.Context, in *healthPb.HealthCheckRequest) (*healthPb.HealthCheckResponse, error) {
	if !s.datastore.HasSynced() {
		klog.Infof("gRPC health check not serving: %s", in.String())
		return &healthPb.HealthCheckResponse{Status: healthPb.HealthCheckResponse_NOT_SERVING}, nil
	}
	klog.Infof("gRPC health check serving: %s", in.String())
	return &healthPb.HealthCheckResponse{Status: healthPb.HealthCheckResponse_SERVING}, nil
}

func (s *healthServer) Watch(in *healthPb.HealthCheckRequest, srv healthPb.Health_WatchServer) error {
	return status.Error(codes.Unimplemented, "Watch is not implemented")
}
