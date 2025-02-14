package logging

import (
	"github.com/go-logr/logr"
	uberzap "go.uber.org/zap"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
)

// NewTestLogger creates a new Zap logger using the dev mode.
func NewTestLogger() logr.Logger {
	return zap.New(zap.UseDevMode(true), zap.RawZapOpts(uberzap.AddCaller()))
}
