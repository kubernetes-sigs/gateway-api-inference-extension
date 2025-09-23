package common

import (
	"context"
	"fmt"
	"os"

	"github.com/go-logr/logr"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.37.0"

	"sigs.k8s.io/gateway-api-inference-extension/version"
)

type errorHandler struct {
	logger logr.Logger
}

func (h *errorHandler) Handle(err error) {
	h.logger.Error(err, "trace error occurred")
}

func InitTracing(ctx context.Context, logger logr.Logger) error {
	logger = logger.WithName("trace")
	loggerWrap := &errorHandler{logger: logger}

	serviceName, ok := os.LookupEnv("OTEL_SERVICE_NAME")
	if !ok {
		serviceName = "gateway-api-inference-extension"
		os.Setenv("OTEL_SERVICE_NAME", serviceName)
	}

	collectorAddr, ok := os.LookupEnv("OTEL_EXPORTER_OTLP_ENDPOINT")
	if !ok {
		collectorAddr = "http://localhost:4317"
		os.Setenv("OTEL_EXPORTER_OTLP_ENDPOINT", collectorAddr)
	}

	traceExporter, err := otlptracegrpc.New(ctx, otlptracegrpc.WithInsecure())
	if err != nil {
		loggerWrap.Handle(fmt.Errorf("%s: %v", "new OTel trace gRPC exporter fail", err))
		return nil
	}

	logger.Info(fmt.Sprintf("OTel trace exporter connect to: %s with service name: %s", collectorAddr, serviceName))
	opt := []sdktrace.TracerProviderOption{
		sdktrace.WithBatcher(traceExporter),
		sdktrace.WithSampler(sdktrace.ParentBased(sdktrace.AlwaysSample())),
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
			loggerWrap.Handle(fmt.Errorf("%s: %v", "failed to shutdown MeterProvider", err))
		}

		logger.Info("trace provider shutting down")
	}()

	return nil
}
