package metrics

import (
	"net"
	"net/http"
	"strconv"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/klog/v2"
)

func StartMetricsHandler(port int) {
	klog.Info("Starting metrics HTTP handler ...")

	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(
		legacyregistry.DefaultGatherer,
		promhttp.HandlerOpts{},
	))

	server := &http.Server{
		Addr:    net.JoinHostPort("", strconv.Itoa(port)),
		Handler: mux,
	}
	if err := server.ListenAndServe(); err != http.ErrServerClosed {
		klog.Fatalf("failed to start metrics HTTP handler: %v", err)
	}
}
