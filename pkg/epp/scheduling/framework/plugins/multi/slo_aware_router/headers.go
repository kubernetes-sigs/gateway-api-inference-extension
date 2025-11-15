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

// Package requestcontrol contains helpers to decouple latency-predictor logic.
package slo_aware_router

import (
	"fmt"
	"strconv"

	schedulingtypes "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	errutil "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/error"
)

// parseFloatHeader retrieves a header by name, parses it as a float64,
// and returns the value or an error if the header is missing or invalid.
func parseFloatHeader(request schedulingtypes.LLMRequest, headerName string) (float64, bool, error) {
	// 1. Get header value from the map
	headerValue, ok := request.Headers[headerName]
	if !ok {
		return 0, false, nil // Header not found, return 0 and false
	}

	// 2. Parse the header value to a float64
	parsedFloat, err := strconv.ParseFloat(headerValue, 64)
	if err != nil {
		return 0, false, errutil.Error{
			Code: errutil.BadRequest,
			Msg:  fmt.Sprintf("%s must be a float", headerName),
		}
	}

	// 3. Return the successfully parsed value
	return parsedFloat, true, nil
}

// parseFloatHeader retrieves a header by name, parses it as a bool,
// and returns the value or an error if the header is missing or invalid.
func parseBoolHeader(request schedulingtypes.LLMRequest, headerName string) (bool, error) {
	// 1. Get header value from the map
	headerValue, ok := request.Headers[headerName]
	if !ok {
		return false, nil // Header not found, return 0 and false
	}

	// 2. Parse the header value to a bool
	parsedBool, err := strconv.ParseBool(headerValue)
	if err != nil {
		return false, errutil.Error{
			Code: errutil.BadRequest,
			Msg:  fmt.Sprintf("%s must be a bool", headerName),
		}
	}

	// 3. Return the successfully parsed value
	return parsedBool, nil
}
