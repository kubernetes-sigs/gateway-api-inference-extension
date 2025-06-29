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
	"fmt"
)

// Error is an error struct for errors returned by the epp server.
type Error struct {
	Code string
	Msg  string
}

const (
	Unknown                        = "Unknown"
	BadRequest                     = "BadRequest"
	Internal                       = "Internal"
	ServiceUnavailable             = "ServiceUnavailable"
	ModelServerError               = "ModelServerError"
	BadConfiguration               = "BadConfiguration"
	InferencePoolResourceExhausted = "InferencePoolResourceExhausted"
)

// Error returns a string version of the error.
func (e Error) Error() string {
	return fmt.Sprintf("inference gateway: %s - %s", e.Code, e.Msg)
}

// CanonicalCode returns the error's ErrorCode.
func CanonicalCode(err error) string {
	e, ok := err.(Error)
	if ok {
		return e.Code
	}
	return Unknown
}
