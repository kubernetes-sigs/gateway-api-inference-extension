// Package test contains e2e tests for the ext proc while faking the backend pods.
package test

import (
	"context"
	"fmt"
	"testing"
	"time"

	configPb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/google/go-cmp/cmp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/testing/protocmp"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/api/v1alpha1"
	"inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/backend"
)

const (
	port = 9002
)

func TestHandleRequestBody(t *testing.T) {
	tests := []struct {
		name        string
		req         *extProcPb.ProcessingRequest
		pods        []*backend.PodMetrics
		models      map[string]*v1alpha1.InferenceModel
		wantHeaders []*configPb.HeaderValueOption
		wantBody    []byte
		wantErr     bool
	}{
		{
			name: "success",
			req:  GenerateRequest("my-model"),
			models: map[string]*v1alpha1.InferenceModel{
				"my-model": {
					Spec: v1alpha1.InferenceModelSpec{
						ModelName: "my-model",
						TargetModels: []v1alpha1.TargetModel{
							{
								Name:   "my-model-v1",
								Weight: 100,
							},
						},
					},
				},
			},
			// pod-1 will be picked because it has relatively low queue size, with the requested
			// model being active, and has low KV cache.
			pods: []*backend.PodMetrics{
				{
					Pod: FakePod(0),
					Metrics: backend.Metrics{
						WaitingQueueSize:    0,
						KVCacheUsagePercent: 0.2,
						ActiveModels: map[string]int{
							"foo": 1,
							"bar": 1,
						},
					},
				},
				{
					Pod: FakePod(1),
					Metrics: backend.Metrics{
						WaitingQueueSize:    0,
						KVCacheUsagePercent: 0.1,
						ActiveModels: map[string]int{
							"foo":         1,
							"my-model-v1": 1,
						},
					},
				},
				{
					Pod: FakePod(2),
					Metrics: backend.Metrics{
						WaitingQueueSize:    10,
						KVCacheUsagePercent: 0.2,
						ActiveModels: map[string]int{
							"foo": 1,
						},
					},
				},
			},
			wantHeaders: []*configPb.HeaderValueOption{
				{
					Header: &configPb.HeaderValue{
						Key:      "target-pod",
						RawValue: []byte("address-1"),
					},
				},
				{
					Header: &configPb.HeaderValue{
						Key:      "Content-Length",
						RawValue: []byte("73"),
					},
				},
			},
			wantBody: []byte("{\"max_tokens\":100,\"model\":\"my-model-v1\",\"prompt\":\"hello\",\"temperature\":0}"),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			client, cleanup := setUpServer(t, test.pods, test.models)
			t.Cleanup(cleanup)
			want := &extProcPb.ProcessingResponse{
				Response: &extProcPb.ProcessingResponse_RequestBody{
					RequestBody: &extProcPb.BodyResponse{
						Response: &extProcPb.CommonResponse{
							HeaderMutation: &extProcPb.HeaderMutation{
								SetHeaders: test.wantHeaders,
							},
							BodyMutation: &extProcPb.BodyMutation{
								Mutation: &extProcPb.BodyMutation_Body{
									Body: test.wantBody,
								},
							},
						},
					},
				},
			}
			res, err := sendRequest(t, client, test.req)

			if (err != nil) != test.wantErr {
				t.Fatalf("Unexpected error, got %v, want %v", err, test.wantErr)
			}

			if diff := cmp.Diff(want, res, protocmp.Transform()); diff != "" {
				t.Errorf("Unexpected response, (-want +got): %v", diff)
			}
		})
	}

}

func setUpServer(t *testing.T, pods []*backend.PodMetrics, models map[string]*v1alpha1.InferenceModel) (client extProcPb.ExternalProcessor_ProcessClient, cleanup func()) {
	server := StartExtProc(port, time.Second, time.Second, pods, models)

	address := fmt.Sprintf("localhost:%v", port)
	// Create a grpc connection
	conn, err := grpc.NewClient(address, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("Failed to connect to %v: %v", address, err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	client, err = extProcPb.NewExternalProcessorClient(conn).Process(ctx)
	if err != nil {
		t.Fatalf("Failed to create client: %v", err)
	}
	return client, func() {
		cancel()
		conn.Close()
		server.GracefulStop()
	}
}

func sendRequest(t *testing.T, client extProcPb.ExternalProcessor_ProcessClient, req *extProcPb.ProcessingRequest) (*extProcPb.ProcessingResponse, error) {
	t.Logf("Sending request: %v", req)
	if err := client.Send(req); err != nil {
		t.Logf("Failed to send request %+v: %v", req, err)
		return nil, err
	}

	res, err := client.Recv()
	if err != nil {
		t.Logf("Failed to receive: %v", err)
		return nil, err
	}
	t.Logf("Received request %+v", res)
	return res, err
}
