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
