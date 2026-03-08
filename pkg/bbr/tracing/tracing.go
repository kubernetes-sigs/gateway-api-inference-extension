/*
Copyright 2026 The Kubernetes Authors.

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

package tracing

import (
	"context"
	"os"

	"github.com/go-logr/logr"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
	tracingutils "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/tracing"
	"sigs.k8s.io/gateway-api-inference-extension/version"
)

const (
	defaultServiceName         = "gateway-api-inference-extension/bbr"
	extprocInstrumentationName = defaultServiceName + "/extproc"
)

func InitTracing(ctx context.Context, logger logr.Logger) error {
	_, ok := os.LookupEnv("OTEL_SERVICE_NAME")
	if !ok {
		os.Setenv("OTEL_SERVICE_NAME", defaultServiceName)
	}

	return tracingutils.InitTracing(ctx, logger)
}

func GetExtProcTracer() trace.Tracer {
	return otel.Tracer(
		extprocInstrumentationName,
		trace.WithInstrumentationVersion(version.BuildRef),
		trace.WithInstrumentationAttributes(
			attribute.String("commit-sha", version.CommitSHA),
		),
	)
}
