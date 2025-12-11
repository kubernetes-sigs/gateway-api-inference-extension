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
	"fmt"
	"strings"

	"google.golang.org/protobuf/encoding/protojson"

	eppb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"encoding/json"
)

// PrettyPrintResponses returns a human-readable string with:
// - Full JSON representation of each response
// - Decoded headers and body content (pretty-printed if JSON)
func PrettyPrintResponses(responses []*eppb.ProcessingResponse) string {
	var builder strings.Builder

	for i, resp := range responses {
		// Marshal protobuf to JSON
		jsonBytes, err := protojson.MarshalOptions{
			Multiline: true,
			Indent:    "  ",
		}.Marshal(resp)
		if err != nil {
			builder.WriteString(fmt.Sprintf("Error marshaling response %d: %v\n", i, err))
			continue
		}

		builder.WriteString(fmt.Sprintf("\n=== Response %d ===\n", i))
		builder.WriteString(string(jsonBytes))
		builder.WriteString("\n")

		// Decode headers
		if headers := resp.GetRequestHeaders(); headers != nil {
			builder.WriteString("\nDecoded Headers:\n")
			for _, h := range headers.GetResponse().GetHeaderMutation().GetSetHeaders() {
				key := h.Header.Key
				raw := h.Header.RawValue
				if len(raw) > 0 {
					decoded := string(raw) // RawValue is []byte, safe to convert
					builder.WriteString(fmt.Sprintf("  %s: %s\n", key, decoded))
				}
			}
		}

		// Decode body
		if body := resp.GetRequestBody(); body != nil {
			mutation := body.GetResponse().GetBodyMutation()
			if mutation != nil {
				if streamed := mutation.GetStreamedResponse(); streamed != nil {
					builder.WriteString("\nDecoded Streamed Body:\n")
					builder.WriteString(prettyIfJSON(streamed.Body))
				} else if raw := mutation.GetBody(); len(raw) > 0 {
					builder.WriteString("\nDecoded Body:\n")
					builder.WriteString(prettyIfJSON(raw))
				}
			}
		}
		builder.WriteString("\n====================\n")
	}

	return builder.String()
}

// prettyIfJSON tries to pretty-print JSON if valid, else returns raw text
func prettyIfJSON(data []byte) string {
	var obj interface{}
	if err := json.Unmarshal(data, &obj); err == nil {
		pretty, _ := json.MarshalIndent(obj, "  ", "  ")
		return string(pretty) + "\n"
	}
	return string(data) + "\n"
}
