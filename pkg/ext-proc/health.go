package main

import (
	"context"
	"fmt"

	"google.golang.org/grpc/codes"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/api/v1alpha1"
	klog "k8s.io/klog/v2"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

type healthServer struct {
	client.Client
}

func (s *healthServer) Check(ctx context.Context, in *healthPb.HealthCheckRequest) (*healthPb.HealthCheckResponse, error) {
	if err := s.checkResources(); err != nil {
		klog.Infof("gRPC health check not serving: %s", in.String())
		return &healthPb.HealthCheckResponse{Status: healthPb.HealthCheckResponse_NOT_SERVING}, nil
	}
	klog.Infof("gRPC health check serving: %s", in.String())
	return &healthPb.HealthCheckResponse{Status: healthPb.HealthCheckResponse_SERVING}, nil
}

func (s *healthServer) Watch(in *healthPb.HealthCheckRequest, srv healthPb.Health_WatchServer) error {
	return status.Error(codes.Unimplemented, "Watch is not implemented")
}

// checkResources uses a client to list all InferenceModels in the configured namespace
// and gets the configured InferencePool by name and namespace. If any client calls fail,
// no InferenceModels exist, or the InferencePool does not exist, an error is returned.
func (s *healthServer) checkResources() error {
	ctx := context.Background()
	var infPool v1alpha1.InferencePool
	if err := s.Client.Get(
		ctx,
		client.ObjectKey{Name: *poolName, Namespace: *poolNamespace},
		&infPool,
	); err != nil {
		return fmt.Errorf("failed to get InferencePool %s/%s: %v", *poolNamespace, *poolName, err)
	}
	klog.Infof("Successfully retrieved InferencePool %s/%s", *poolNamespace, *poolName)

	var modelList v1alpha1.InferenceModelList
	if err := s.Client.List(ctx, &modelList, client.InNamespace(*poolNamespace)); err != nil {
		return fmt.Errorf("failed to list InferenceModels in namespace %s: %v", *poolNamespace, err)
	}

	// Ensure at least 1 InferenceModel
	if len(modelList.Items) == 0 {
		return fmt.Errorf("no InferenceModels exist in namespace %s", *poolNamespace)
	}
	klog.Infof("Found %d InferenceModels in namespace %s", len(modelList.Items), *poolNamespace)

	return nil
}
