package handlers

import (
	"testing"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/google/go-cmp/cmp"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/logging"
)

const (
	body = `
	{
		"id": "cmpl-573498d260f2423f9e42817bbba3743a",
		"object": "text_completion",
		"created": 1732563765,
		"model": "meta-llama/Llama-2-7b-hf",
		"choices": [
			{
				"index": 0,
				"text": " Chronicle\nThe San Francisco Chronicle has a new book review section, and it's a good one. The reviews are short, but they're well-written and well-informed. The Chronicle's book review section is a good place to start if you're looking for a good book review.\nThe Chronicle's book review section is a good place to start if you're looking for a good book review. The Chronicle's book review section",
				"logprobs": null,
				"finish_reason": "length",
				"stop_reason": null,
				"prompt_logprobs": null
			}
		],
		"usage": {
			"prompt_tokens": 11,
			"total_tokens": 111,
			"completion_tokens": 100
		}
	}
	`
)

func TestHandleResponseBody(t *testing.T) {
	logger := logutil.NewTestLogger()

	tests := []struct {
		name    string
		req     *extProcPb.ProcessingRequest_ResponseBody
		want    Response
		wantErr bool
	}{
		{
			name: "success",
			req: &extProcPb.ProcessingRequest_ResponseBody{
				ResponseBody: &extProcPb.HttpBody{
					Body: []byte(body),
				},
			},
			want: Response{
				Usage: Usage{
					PromptTokens:     11,
					TotalTokens:      111,
					CompletionTokens: 100,
				},
			},
		},
		{
			name: "malformed response",
			req: &extProcPb.ProcessingRequest_ResponseBody{
				ResponseBody: &extProcPb.HttpBody{
					Body: []byte("malformed json"),
				},
			},
			wantErr: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := &Server{}
			reqCtx := &RequestContext{}
			_, err := server.HandleResponseBody(logger, reqCtx, &extProcPb.ProcessingRequest{Request: test.req})
			if err != nil {
				if !test.wantErr {
					t.Fatalf("HandleResponseBody returned unexpected error: %v, want %v", err, test.wantErr)
				}
				return
			}

			if diff := cmp.Diff(test.want, reqCtx.Response); diff != "" {
				t.Errorf("HandleResponseBody returned unexpected response, diff(-want, +got): %v", diff)
			}
		})
	}
}
