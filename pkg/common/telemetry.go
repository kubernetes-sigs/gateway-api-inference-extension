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

package common

import (
	"context"
	"fmt"
	"os"
	"strconv"

	"github.com/go-logr/logr"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.37.0"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"
	"sigs.k8s.io/gateway-api-inference-extension/version"
)

type errorHandler struct {
	logger logr.Logger
}

func (h *errorHandler) Handle(err error) {
	h.logger.V(logging.DEFAULT).Error(err, "trace error occurred")
}

func InitTracing(ctx context.Context, logger logr.Logger) error {
	logger = logger.WithName("trace")
	loggerWrap := &errorHandler{logger: logger}

	_, ok := os.LookupEnv("OTEL_SERVICE_NAME")
	if !ok {
		os.Setenv("OTEL_SERVICE_NAME", "gateway-api-inference-extension")
	}

	_, ok = os.LookupEnv("OTEL_EXPORTER_OTLP_ENDPOINT")
	if !ok {
		os.Setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
	}

	traceExporter, err := initTraceExporter(ctx, logger)
	if err != nil {
		loggerWrap.Handle(fmt.Errorf("%s: %v", "init trace exporter failed", err))
		return err
	}

	// Go SDK doesn't have an automatic sampler, handle manually
	samplerType, ok := os.LookupEnv("OTEL_TRACES_SAMPLER")
	if !ok {
		samplerType = "parentbased_traceidratio"
	}
	samplerARG, ok := os.LookupEnv("OTEL_TRACES_SAMPLER_ARG")
	if !ok {
		samplerARG = "0.1"
	}

	sampler := sdktrace.ParentBased(sdktrace.TraceIDRatioBased(0.1))
	if samplerType == "parentbased_traceidratio" {
		fraction, err := strconv.ParseFloat(samplerARG, 64)
		if err != nil {
			fraction = 0.1
		}

		sampler = sdktrace.ParentBased(sdktrace.TraceIDRatioBased(fraction))
	} else {
		loggerWrap.Handle(fmt.Errorf("unsupported sampler type: %s, fallback to parentbased_traceidratio with 0.1 Ratio", samplerType))
	}

	opt := []sdktrace.TracerProviderOption{
		sdktrace.WithBatcher(traceExporter),
		sdktrace.WithSampler(sampler),
		sdktrace.WithResource(resource.NewWithAttributes(
			semconv.SchemaURL,
			semconv.ServiceVersionKey.String(version.BuildRef),
		)),
	}

	tracerProvider := sdktrace.NewTracerProvider(opt...)
	otel.SetTracerProvider(tracerProvider)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(propagation.TraceContext{}, propagation.Baggage{}))
	otel.SetErrorHandler(loggerWrap)

	go func() {
		<-ctx.Done()
		err := tracerProvider.Shutdown(context.Background())
		if err != nil {
			loggerWrap.Handle(fmt.Errorf("%s: %v", "failed to shutdown TraceProvider", err))
		}

		logger.V(logging.DEFAULT).Info("trace provider shutting down")
	}()

	return nil
}

// initTraceExporter create a SpanExporter
// support exporter type
// - console: export spans in console for development use case
// - otlp: export spans through gRPC to an opentelemetry collector
func initTraceExporter(ctx context.Context, logger logr.Logger) (sdktrace.SpanExporter, error) {
	var traceExporter sdktrace.SpanExporter
	traceExporter, err := stdouttrace.New(stdouttrace.WithPrettyPrint())
	if err != nil {
		return nil, fmt.Errorf("failed to create stdouttrace exporter: %w", err)
	}

	exporterType, ok := os.LookupEnv("OTEL_TRACES_EXPORTER")
	if !ok {
		exporterType = "console"
	}

	logger.Info("init OTel trace exporter", "type", exporterType)
	if exporterType == "otlp" {
		traceExporter, err = otlptracegrpc.New(ctx, otlptracegrpc.WithInsecure())
		if err != nil {
			return nil, fmt.Errorf("failed to create otlp-grcp exporter: %w", err)
		}
	}

	return traceExporter, nil
}
