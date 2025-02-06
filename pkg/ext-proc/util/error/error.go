package error

import (
	"fmt"

	"k8s.io/klog/v2"

	logutil "inference.networking.x-k8s.io/gateway-api-inference-extension/pkg/ext-proc/util/logging"
)

// Error is an error struct for errors returned by the epp server.
type Error struct {
	Code ErrorCode
	Msg  string
}

// ErrorCode is the Error code type.
type ErrorCode int

// error types
const (
	Unknown ErrorCode = iota
	InvalidRequest
	ResourceExhausted
	Internal
	ModelServerError
)

// Error returns a string version of the error.
func (e Error) Error() string {
	return fmt.Sprintf("inference gateway: %d - %s", e.Code, e.Msg)
}

func (e ErrorCode) String() string {
	switch e {
	case InvalidRequest:
		return "InvalidRequest"
	case Internal:
		return "Internal"
	case ModelServerError:
		return "ModelServer"
	case ResourceExhausted:
		return "ResourceExhausted"
	default:
		return "Unknown"
	}
}

// CanonicalCode returns the error's ErrorCode.
func CanonicalCode(err error) ErrorCode {
	e, ok := err.(Error)
	if ok {
		return e.Code
	}
	klog.V(logutil.VERBOSE).Infof("failed to convert to internal error")
	return Unknown
}
