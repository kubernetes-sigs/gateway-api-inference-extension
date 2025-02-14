package logging

import (
	"os"

	"github.com/go-logr/logr"
)

// Fatal calls logger.Error followed by os.Exit(1).
//
// This is a utility function and should not be used in production code!
func Fatal(logger logr.Logger, err error, msg string, keysAndValues ...interface{}) {
	logger.Error(err, msg, keysAndValues...)
	os.Exit(1)
}
