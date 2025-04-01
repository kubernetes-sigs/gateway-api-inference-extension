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

package integration

import (
	"encoding/json"
	"io"
	"testing"
	"time"

	envoyCorev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
)

func SendRequest(t *testing.T, client extProcPb.ExternalProcessor_ProcessClient, req *extProcPb.ProcessingRequest) (*extProcPb.ProcessingResponse, error) {
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
	t.Logf("Received response %+v", res)
	return res, err
}

func StreamedRequest(t *testing.T, client extProcPb.ExternalProcessor_ProcessClient, requests []*extProcPb.ProcessingRequest, expectedResponses int) ([]*extProcPb.ProcessingResponse, error) {
	for _, req := range requests {
		t.Logf("Sending request: %v", req)
		if err := client.Send(req); err != nil {
			t.Logf("Failed to send request %+v: %v", req, err)
			return nil, err
		}
	}
	responses := []*extProcPb.ProcessingResponse{}

	// Make an incredible simple timeout func in the case where
	// there is less than the expected amount of responses; bail and fail.
	var simpleTimeout bool
	go func() {
		time.Sleep(10 * time.Second)
		simpleTimeout = true
	}()

	for range expectedResponses {
		if simpleTimeout {
			break
		}
		res, err := client.Recv()
		if err != nil && err != io.EOF {
			t.Logf("Failed to receive: %v", err)
			return nil, err
		}
		t.Logf("Received response %+v", res)
		responses = append(responses, res)
	}
	return responses, nil
}

func GenerateRequest(logger logr.Logger, prompt, model string) *extProcPb.ProcessingRequest {
	j := map[string]interface{}{
		"prompt":      prompt,
		"max_tokens":  100,
		"temperature": 0,
	}
	if model != "" {
		j["model"] = model
	}

	llmReq, err := json.Marshal(j)
	if err != nil {
		logutil.Fatal(logger, err, "Failed to unmarshal LLM request")
	}
	req := &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestBody{
			RequestBody: &extProcPb.HttpBody{Body: llmReq, EndOfStream: true},
		},
	}
	return req
}

func GenerateStreamedRequestSet(logger logr.Logger, prompt, model string) []*extProcPb.ProcessingRequest {
	requests := []*extProcPb.ProcessingRequest{}
	headerReq := &extProcPb.ProcessingRequest{
		Request: &extProcPb.ProcessingRequest_RequestHeaders{
			RequestHeaders: &extProcPb.HttpHeaders{
				Headers: &envoyCorev3.HeaderMap{
					Headers: []*envoyCorev3.HeaderValue{
						{
							Key:   "hi",
							Value: "mom",
						},
					},
				},
			},
		},
	}
	requests = append(requests, headerReq)
	requests = append(requests, GenerateRequest(logger, prompt, model))
	return requests
}
