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

package logging

import (
	"context"
	"os"

	"github.com/go-logr/logr"
	uberzap "go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
)

const (
	DEFAULT = 2
	VERBOSE = 3
	DEBUG   = 4
	TRACE   = 5
)

// NewTestLogger creates a new Zap logger using the dev mode.
func NewTestLogger() logr.Logger {
	return zap.New(zap.UseDevMode(true), zap.RawZapOpts(uberzap.AddCaller()))
}

// NewTestLoggerIntoContext creates a new Zap logger using the dev mode and inserts it into the given context.
func NewTestLoggerIntoContext(ctx context.Context) context.Context {
	return log.IntoContext(ctx, zap.New(zap.UseDevMode(true), zap.RawZapOpts(uberzap.AddCaller())))
}

// Fatal calls logger.Error followed by os.Exit(1).
//
// This is a utility function and should not be used in production code!
func Fatal(logger logr.Logger, err error, msg string, keysAndValues ...any) {
	logger.Error(err, msg, keysAndValues...)
	os.Exit(1)
}

func InitLogging(logVerbosity int, development bool) logr.Logger {
	// See https://pkg.go.dev/sigs.k8s.io/controller-runtime/pkg/log/zap#Options.Level
	opts := &zap.Options{
		Development: development,
		Level:       uberzap.NewAtomicLevelAt(zapcore.Level(int8(-1 * logVerbosity))),
	}
	logger := zap.New(zap.UseFlagOptions(opts), zap.RawZapOpts(uberzap.AddCaller()))

	return logger
}
