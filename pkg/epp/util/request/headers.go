package request

import (
	"strings"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

const (
	RequestIdHeaderKey = "x-request-id"
)

func ExtractRequestId(req *extProcPb.ProcessingRequest_RequestHeaders) string {
	if req != nil && req.RequestHeaders != nil && req.RequestHeaders.Headers != nil {
		for _, headerKv := range req.RequestHeaders.Headers.Headers {
			if strings.ToLower(headerKv.Key) == RequestIdHeaderKey {
				return string(headerKv.RawValue)
			}
		}
	}
	return ""
}
