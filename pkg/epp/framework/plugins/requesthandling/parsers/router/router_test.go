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

package router

import (
	"context"
	"testing"

	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkrh "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/requesthandling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/metadata"
)

func TestRouterParser_ParseRequest_MultipartTranscriptions(t *testing.T) {
	ctx := context.Background()
	delegate := &mockParser{}
	p := NewRouterParser(delegate)

	headers := map[string]string{
		"content-type":         "multipart/form-data; boundary=----boundary",
		":path":                metadata.AudioTranscriptionsPathPrefix,
		metadata.ModelNameKey:  "whisper-1",
	}
	body := []byte("raw multipart body")

	got, err := p.ParseRequest(ctx, body, headers)
	if err != nil {
		t.Fatalf("ParseRequest: %v", err)
	}
	if got == nil {
		t.Fatal("expected non-nil LLMRequestBody")
	}
	if got.AudioTranscriptions == nil || got.AudioTranscriptions.ModelName != "whisper-1" {
		t.Errorf("AudioTranscriptions.ModelName = %v, want whisper-1", got.AudioTranscriptions)
	}
	if delegate.parseRequestCalled {
		t.Error("delegate ParseRequest should not be called for multipart transcriptions")
	}
}

func TestRouterParser_ParseRequest_MultipartMissingModel(t *testing.T) {
	ctx := context.Background()
	delegate := &mockParser{}
	p := NewRouterParser(delegate)

	headers := map[string]string{
		"content-type": "multipart/form-data; boundary=----boundary",
		":path":        metadata.AudioTranscriptionsPathPrefix,
	}
	body := []byte("raw multipart body")

	got, err := p.ParseRequest(ctx, body, headers)
	if err == nil {
		t.Fatal("expected error when model header missing")
	}
	if got != nil {
		t.Errorf("expected nil body, got %v", got)
	}
}

func TestRouterParser_ParseRequest_DelegatesNonMultipart(t *testing.T) {
	ctx := context.Background()
	delegate := &mockParser{}
	p := NewRouterParser(delegate)

	headers := map[string]string{
		"content-type": "application/json",
		":path":        "/v1/chat/completions",
	}
	body := []byte(`{"model":"gpt-4","messages":[]}`)

	_, _ = p.ParseRequest(ctx, body, headers)
	if !delegate.parseRequestCalled {
		t.Error("delegate ParseRequest should be called for non-multipart")
	}
}

type mockParser struct {
	parseRequestCalled bool
}

var _ fwkrh.Parser = (*mockParser)(nil)

func (m *mockParser) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "mock", Name: "mock"}
}

func (m *mockParser) ParseRequest(ctx context.Context, body []byte, headers map[string]string) (*scheduling.LLMRequestBody, error) {
	m.parseRequestCalled = true
	return nil, nil
}

func (m *mockParser) ParseResponse(ctx context.Context, body []byte, headers map[string]string, endOfStream bool) (*fwkrh.ParsedResponse, error) {
	return nil, nil
}
