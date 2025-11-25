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

package utils

import (
	"context"
	"net"
	"testing"
	"time"

	corev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	pb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"

	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"
	"sigs.k8s.io/gateway-api-inference-extension/apix/v1alpha2"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/datastore"
	pooltuil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/pool"
	testutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/testing"
)

const bufSize = 1024 * 1024

var testListener *bufconn.Listener

func PrepareForTestStreamingServer(objectives []*v1alpha2.InferenceObjective, pods []*corev1.Pod, poolName string, namespace string,
	poolPort int32) (context.Context, context.CancelFunc, datastore.Datastore, *metrics.FakePodMetricsClient) {
	ctx, cancel := context.WithCancel(context.Background())

	pmc := &metrics.FakePodMetricsClient{}
	pmf := metrics.NewPodMetricsFactory(pmc, time.Second)
	ds := datastore.NewDatastore(ctx, pmf, 0)

	initObjs := []client.Object{}
	for _, objective := range objectives {
		initObjs = append(initObjs, objective)
		ds.ObjectiveSet(objective)
	}
	for _, pod := range pods {
		initObjs = append(initObjs, pod)
		ds.PodUpdateOrAddIfNotExist(pod)
	}

	scheme := runtime.NewScheme()
	_ = clientgoscheme.AddToScheme(scheme)
	_ = v1alpha2.Install(scheme)
	_ = v1.Install(scheme)
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(initObjs...).
		Build()
	pool := testutil.MakeInferencePool(poolName).Namespace(namespace).ObjRef()
	pool.Spec.TargetPorts = []v1.Port{{Number: v1.PortNumber(poolPort)}}
	_ = ds.PoolSet(context.Background(), fakeClient, pooltuil.InferencePoolToEndpointPool(pool))

	return ctx, cancel, ds, pmc
}

func SetupTestStreamingServer(t *testing.T, ctx context.Context, ds datastore.Datastore,
	streamingServer pb.ExternalProcessorServer) (*bufconn.Listener, chan error) {
	testListener = bufconn.Listen(bufSize)

	errChan := make(chan error)
	go func() {
		err := LaunchTestGRPCServer(streamingServer, ctx, testListener)
		if err != nil {
			t.Error("Error launching listener", err)
		}
		errChan <- err
	}()

	time.Sleep(2 * time.Second)
	return testListener, errChan
}

func testDialer(context.Context, string) (net.Conn, error) {
	return testListener.Dial()
}

func GetStreamingServerClient(ctx context.Context, t *testing.T) (pb.ExternalProcessor_ProcessClient, *grpc.ClientConn) {
	opts := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(testDialer),
	}
	conn, err := grpc.NewClient("passthrough://bufconn", opts...)
	if err != nil {
		t.Error(err)
		return nil, nil
	}

	extProcClient := pb.NewExternalProcessorClient(conn)
	process, err := extProcClient.Process(ctx)
	if err != nil {
		t.Error(err)
		return nil, nil
	}

	return process, conn
}

// LaunchTestGRPCServer actually starts the server (enables testing)
func LaunchTestGRPCServer(s pb.ExternalProcessorServer, ctx context.Context, listener net.Listener) error {
	grpcServer := grpc.NewServer()

	pb.RegisterExternalProcessorServer(grpcServer, s)

	// Shutdown on context closed.
	// Terminate the server on context closed.
	go func() {
		<-ctx.Done()
		grpcServer.GracefulStop()
	}()

	if err := grpcServer.Serve(listener); err != nil {
		return err
	}

	return nil
}

func CheckEnvoyGRPCHeaders(t *testing.T, response *pb.CommonResponse, expectedHeaders map[string]string) bool {
	headers := response.HeaderMutation.SetHeaders
	for expectedKey, expectedValue := range expectedHeaders {
		found := false
		for _, header := range headers {
			if header.Header.Key == expectedKey {
				if expectedValue != string(header.Header.RawValue) {
					t.Errorf("Incorrect value for header %s, want %s got %s", expectedKey, expectedValue,
						string(header.Header.RawValue))
					return false
				}
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Missing header %s", expectedKey)
			return false
		}
	}

	for _, header := range headers {
		expectedValue, ok := expectedHeaders[header.Header.Key]
		if !ok {
			t.Errorf("Unexpected header %s", header.Header.Key)
			return false
		} else if expectedValue != string(header.Header.RawValue) {
			t.Errorf("Incorrect value for header %s, want %s got %s", header.Header.Key, expectedValue,
				string(header.Header.RawValue))
			return false
		}
	}
	return true
}

func BuildEnvoyGRPCHeaders(headers map[string]string, rawValue bool) *pb.HttpHeaders {
	headerValues := make([]*corev3.HeaderValue, 0)
	for key, value := range headers {
		header := &corev3.HeaderValue{Key: key}
		if rawValue {
			header.RawValue = []byte(value)
		} else {
			header.Value = value
		}
		headerValues = append(headerValues, header)
	}
	return &pb.HttpHeaders{
		Headers: &corev3.HeaderMap{
			Headers: headerValues,
		},
	}
}
