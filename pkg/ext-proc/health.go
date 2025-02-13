package main

import (
	"context"

	"google.golang.org/grpc/codes"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"
	klog "k8s.io/klog/v2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/backend"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/logging"
)

type healthServer struct {
	datastore *backend.K8sDatastore
}

func (s *healthServer) Check(ctx context.Context, in *healthPb.HealthCheckRequest) (*healthPb.HealthCheckResponse, error) {
	if !s.datastore.HasSynced() {
		klog.V(logutil.VERBOSE).InfoS("gRPC health check not serving", "service", in.Service)
		return &healthPb.HealthCheckResponse{Status: healthPb.HealthCheckResponse_NOT_SERVING}, nil
	}
	klog.V(logutil.VERBOSE).InfoS("gRPC health check serving", "service", in.Service)
	return &healthPb.HealthCheckResponse{Status: healthPb.HealthCheckResponse_SERVING}, nil
}

func (s *healthServer) Watch(in *healthPb.HealthCheckRequest, srv healthPb.Health_WatchServer) error {
	return status.Error(codes.Unimplemented, "Watch is not implemented")
}
