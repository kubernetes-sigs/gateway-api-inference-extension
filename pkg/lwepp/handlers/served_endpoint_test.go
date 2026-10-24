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

package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/zapr"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"google.golang.org/protobuf/types/known/structpb"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/lwepp/datastore"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/lwepp/metadata"
)

func TestProcessServedEndpointObservation(t *testing.T) {
	for _, tc := range []struct{ name, id, served, expected string }{
		{"actual endpoint differs from selected", "first", "10.0.0.2:3000", "10.0.0.2:3000"},
		{"separate request", "second", "10.0.0.3:3000", "10.0.0.3:3000"},
		{"missing metadata", "third", "", "fail: missing envoy lb metadata"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			var logs bytes.Buffer
			encoder := zapcore.NewJSONEncoder(zap.NewProductionEncoderConfig())
			logger := zapr.NewLogger(zap.New(zapcore.NewCore(encoder, zapcore.AddSync(&logs), zap.InfoLevel)))
			response := &ext.ProcessingRequest{Request: &ext.ProcessingRequest_ResponseHeaders{ResponseHeaders: &ext.HttpHeaders{}}}
			if tc.served != "" {
				response.MetadataContext = &core.Metadata{FilterMetadata: map[string]*structpb.Struct{
					metadata.DestinationEndpointNamespace: {Fields: map[string]*structpb.Value{metadata.DestinationEndpointServedKey: structpb.NewStringValue(tc.served)}},
				}}
			}
			stream := &mockProcessServer{ctx: log.IntoContext(context.Background(), logger), recvMessages: []*ext.ProcessingRequest{
				{Request: &ext.ProcessingRequest_RequestHeaders{RequestHeaders: &ext.HttpHeaders{EndOfStream: true, Headers: &core.HeaderMap{Headers: []*core.HeaderValue{{Key: metadata.ConformanceRequestIDHeader, Value: tc.id}}}}}},
				response,
			}}
			server := NewStreamingServer(&mockDatastore{pods: []*datastore.Endpoint{{Address: "10.0.0.1", Port: "3000"}}})
			require.NoError(t, server.Process(stream))
			found := false
			for _, line := range strings.Split(logs.String(), "\n") {
				var event map[string]any
				if json.Unmarshal([]byte(line), &event) != nil || event["msg"] != metadata.ServedEndpointObservation {
					continue
				}
				require.Equal(t, tc.id, event["conformanceRequestID"])
				require.Equal(t, tc.expected, event["servedEndpoint"])
				found = true
			}
			require.True(t, found, "response metadata must be observable without a client response header")
		})
	}
}
