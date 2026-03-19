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

package error

import (
	"errors"
	"strings"
	"testing"

	envoyTypePb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
)

func TestError_Error(t *testing.T) {
	tests := []struct {
		name string
		err  Error
		want string
	}{
		{
			name: "BadRequest error",
			err: Error{
				Code: BadRequest,
				Msg:  "invalid model name",
			},
			want: "inference error: BadRequest - invalid model name",
		},
		{
			name: "Internal error",
			err: Error{
				Code: Internal,
				Msg:  "unexpected condition",
			},
			want: "inference error: Internal - unexpected condition",
		},
		{
			name: "ServiceUnavailable error",
			err: Error{
				Code: ServiceUnavailable,
				Msg:  "service unavailable",
			},
			want: "inference error: ServiceUnavailable - service unavailable",
		},
		{
			name: "ModelServerError",
			err: Error{
				Code: ModelServerError,
				Msg:  "connection timeout",
			},
			want: "inference error: ModelServerError - connection timeout",
		},
		{
			name: "ResourceExhausted error",
			err: Error{
				Code: ResourceExhausted,
				Msg:  "no available pods",
			},
			want: "inference error: ResourceExhausted - no available pods",
		},
		{
			name: "Unknown error",
			err: Error{
				Code: Unknown,
				Msg:  "something went wrong",
			},
			want: "inference error: Unknown - something went wrong",
		},
		{
			name: "Empty message",
			err: Error{
				Code: BadRequest,
				Msg:  "",
			},
			want: "inference error: BadRequest - ",
		},
		{
			name: "Empty code",
			err: Error{
				Code: "",
				Msg:  "error occurred",
			},
			want: "inference error:  - error occurred",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.err.Error(); got != tt.want {
				t.Errorf("Error.Error() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCanonicalCode(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want string
	}{
		{
			name: "Error type with BadRequest code",
			err: Error{
				Code: BadRequest,
				Msg:  "invalid input",
			},
			want: BadRequest,
		},
		{
			name: "Error type with Internal code",
			err: Error{
				Code: Internal,
				Msg:  "server error",
			},
			want: Internal,
		},
		{
			name: "Error type with ServiceUnavailable code",
			err: Error{
				Code: ServiceUnavailable,
				Msg:  "Service unavailable error",
			},
			want: ServiceUnavailable,
		},
		{
			name: "Error type with ModelServerError code",
			err: Error{
				Code: ModelServerError,
				Msg:  "model unavailable",
			},
			want: ModelServerError,
		},
		{
			name: "Error type with ResourceExhausted code",
			err: Error{
				Code: ResourceExhausted,
				Msg:  "no resources",
			},
			want: ResourceExhausted,
		},
		{
			name: "Error type with Unknown code",
			err: Error{
				Code: Unknown,
				Msg:  "unknown error",
			},
			want: Unknown,
		},
		{
			name: "Error type with empty code",
			err: Error{
				Code: "",
				Msg:  "no code provided",
			},
			want: "",
		},
		{
			name: "Non-Error type",
			err:  errors.New("standard go error"),
			want: Unknown,
		},
		{
			name: "Nil error",
			err:  nil,
			want: Unknown,
		},
		{
			name: "Custom error type that is not Error",
			err:  customError{msg: "custom error"},
			want: Unknown,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := CanonicalCode(tt.err); got != tt.want {
				t.Errorf("CanonicalCode() = %v, want %v", got, tt.want)
			}
		})
	}
}

// customError is a helper type for testing non-Error error types
type customError struct {
	msg string
}

func (e customError) Error() string {
	return e.msg
}

func TestErrorConstants(t *testing.T) {
	// Verify that error constants match their expected string values
	tests := map[string]string{
		Unknown:            "Unknown",
		BadRequest:         "BadRequest",
		Internal:           "Internal",
		ServiceUnavailable: "ServiceUnavailable",
		ModelServerError:   "ModelServerError",
		ResourceExhausted:  "ResourceExhausted",
	}

	for constant, expected := range tests {
		if constant != expected {
			t.Errorf("Constant value %q != expected %q", constant, expected)
		}
	}
}

func TestBuildErrResponse(t *testing.T) {
	tests := []struct {
		name             string
		err              error
		wantHTTPStatus   envoyTypePb.StatusCode
		wantBodyContains string
		wantGRPCErr      bool
	}{
		{
			name:             "BadRequest returns 400",
			err:              Error{Code: BadRequest, Msg: "invalid model name"},
			wantHTTPStatus:   envoyTypePb.StatusCode_BadRequest,
			wantBodyContains: "invalid model name",
		},
		{
			name:             "Unauthorized returns 401",
			err:              Error{Code: Unauthorized, Msg: "missing token"},
			wantHTTPStatus:   envoyTypePb.StatusCode_Unauthorized,
			wantBodyContains: "missing token",
		},
		{
			name:             "Forbidden returns 403",
			err:              Error{Code: Forbidden, Msg: "unsafe content blocked"},
			wantHTTPStatus:   envoyTypePb.StatusCode_Forbidden,
			wantBodyContains: "unsafe content blocked",
		},
		{
			name:             "NotFound returns 404",
			err:              Error{Code: NotFound, Msg: "model not found"},
			wantHTTPStatus:   envoyTypePb.StatusCode_NotFound,
			wantBodyContains: "model not found",
		},
		{
			name:             "ResourceExhausted returns 429",
			err:              Error{Code: ResourceExhausted, Msg: "no capacity"},
			wantHTTPStatus:   envoyTypePb.StatusCode_TooManyRequests,
			wantBodyContains: "no capacity",
		},
		{
			name:             "Internal returns 500",
			err:              Error{Code: Internal, Msg: "unexpected failure"},
			wantHTTPStatus:   envoyTypePb.StatusCode_InternalServerError,
			wantBodyContains: "unexpected failure",
		},
		{
			name:             "ServiceUnavailable returns 503",
			err:              Error{Code: ServiceUnavailable, Msg: "no endpoints"},
			wantHTTPStatus:   envoyTypePb.StatusCode_ServiceUnavailable,
			wantBodyContains: "no endpoints",
		},
		{
			name:        "plain error returns gRPC error",
			err:         errors.New("unknown problem"),
			wantGRPCErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resp, err := BuildErrResponse(tt.err)

			if tt.wantGRPCErr {
				if err == nil {
					t.Fatal("expected gRPC error, got nil")
				}
				if resp != nil {
					t.Errorf("expected nil response on gRPC error, got %v", resp)
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if resp == nil {
				t.Fatal("expected response, got nil")
			}

			ir := resp.GetImmediateResponse()
			if ir == nil {
				t.Fatal("expected ImmediateResponse, got nil")
			}
			if ir.GetStatus().GetCode() != tt.wantHTTPStatus {
				t.Errorf("HTTP status = %v, want %v", ir.GetStatus().GetCode(), tt.wantHTTPStatus)
			}
			if tt.wantBodyContains != "" && !strings.Contains(string(ir.GetBody()), tt.wantBodyContains) {
				t.Errorf("body %q should contain %q", string(ir.GetBody()), tt.wantBodyContains)
			}
		})
	}
}
