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

package handlers

import (
	"context"
	"testing"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/lwepp/metadata"
)

func TestHandleResponseHeaders(t *testing.T) {
	server := &StreamingServer{}
	reqCtx := &RequestContext{
		SelectedPodIP:  "10.0.0.2",
		TargetEndpoint: "10.0.0.2:3000",
	}
	req := &extProcPb.ProcessingRequest_ResponseHeaders{}

	resp, err := server.handleResponseHeaders(context.Background(), reqCtx, req)
	assert.NoError(t, err)
	assert.NotNil(t, resp)

	// Check if the header is set correctly
	headerMutation := resp.GetResponseHeaders().GetResponse().GetHeaderMutation()
	assert.NotNil(t, headerMutation)

	setHeaders := headerMutation.GetSetHeaders()
	assert.Len(t, setHeaders, 1)

	assert.Equal(t, metadata.ConformanceTestResultHeader, setHeaders[0].GetHeader().GetKey())
	assert.Equal(t, "10.0.0.2:3000", string(setHeaders[0].GetHeader().GetRawValue()))
}
