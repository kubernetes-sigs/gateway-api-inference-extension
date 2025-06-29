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
	"testing"
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
			want: "inference gateway: BadRequest - invalid model name",
		},
		{
			name: "Internal error",
			err: Error{
				Code: Internal,
				Msg:  "unexpected condition",
			},
			want: "inference gateway: Internal - unexpected condition",
		},
		{
			name: "ServiceUnavailable error",
			err: Error{
				Code: ServiceUnavailable,
				Msg:  "service unavailable",
			},
			want: "inference gateway: ServiceUnavailable - service unavailable",
		},
		{
			name: "ModelServerError",
			err: Error{
				Code: ModelServerError,
				Msg:  "connection timeout",
			},
			want: "inference gateway: ModelServerError - connection timeout",
		},
		{
			name: "BadConfiguration error",
			err: Error{
				Code: BadConfiguration,
				Msg:  "missing required field",
			},
			want: "inference gateway: BadConfiguration - missing required field",
		},
		{
			name: "InferencePoolResourceExhausted error",
			err: Error{
				Code: InferencePoolResourceExhausted,
				Msg:  "no available pods",
			},
			want: "inference gateway: InferencePoolResourceExhausted - no available pods",
		},
		{
			name: "Unknown error",
			err: Error{
				Code: Unknown,
				Msg:  "something went wrong",
			},
			want: "inference gateway: Unknown - something went wrong",
		},
		{
			name: "Empty message",
			err: Error{
				Code: BadRequest,
				Msg:  "",
			},
			want: "inference gateway: BadRequest - ",
		},
		{
			name: "Empty code",
			err: Error{
				Code: "",
				Msg:  "error occurred",
			},
			want: "inference gateway:  - error occurred",
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
			name: "Error type with BadConfiguration code",
			err: Error{
				Code: BadConfiguration,
				Msg:  "invalid config",
			},
			want: BadConfiguration,
		},
		{
			name: "Error type with InferencePoolResourceExhausted code",
			err: Error{
				Code: InferencePoolResourceExhausted,
				Msg:  "no resources",
			},
			want: InferencePoolResourceExhausted,
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
		Unknown:                        "Unknown",
		BadRequest:                     "BadRequest",
		Internal:                       "Internal",
		ServiceUnavailable:             "ServiceUnavailable",
		ModelServerError:               "ModelServerError",
		BadConfiguration:               "BadConfiguration",
		InferencePoolResourceExhausted: "InferencePoolResourceExhausted",
	}

	for constant, expected := range tests {
		if constant != expected {
			t.Errorf("Constant value %q != expected %q", constant, expected)
		}
	}
}
