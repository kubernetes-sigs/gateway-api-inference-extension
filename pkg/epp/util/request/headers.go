package request

import (
	"strings"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

const (
	RequestIdHeaderKey = "x-request-id"
)

func ExtractHeaderValue(req *extProcPb.ProcessingRequest_RequestHeaders, headerKey string) string {
	// header key should be case insensitive
	headerKeyInLower := strings.ToLower(headerKey)
	if req != nil && req.RequestHeaders != nil && req.RequestHeaders.Headers != nil {
		for _, headerKv := range req.RequestHeaders.Headers.Headers {
			if strings.ToLower(headerKv.Key) == headerKeyInLower {
				return string(headerKv.RawValue)
			}
		}
	}
	return ""
}
